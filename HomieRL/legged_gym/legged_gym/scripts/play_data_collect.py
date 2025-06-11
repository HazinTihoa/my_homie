import math
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import pygame
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import time
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
import torch.nn.functional as F


def draw_target_cross(env, viewer, target_pos):
    tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    pts = np.array(
        [
            [tx - 0.1, ty, tz],
            [tx + 0.1, ty, tz],
            [tx, ty - 0.1, tz],
            [tx, ty + 0.1, tz],
        ],
        dtype=np.float32,
    )
    cols = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    count = 2
    for env_handle in env.envs:
        env.gym.add_lines(viewer, env_handle, count, pts, cols)


# IK func
def solve_right_arm_ik_jacobian(
    env,
    env_idx,
    actor_handle,#single
    wrist_body_index,
    arm_joint_indices,
    target_pos,
    q_init,
):

    device = env.device  
    # rb_states_np = env.gym.get_actor_rigid_body_states(
    #      env.envs[env_idx], actor_handle, gymapi.STATE_ALL
    # )
    # pos_w = rb_states_np[wrist_body_index][0][0]  # (px, py, pz)
    rb_state_tensor = env.gym.acquire_rigid_body_state_tensor(env.sim)   # 原始 GPU 缓冲区
    env.gym.refresh_rigid_body_state_tensor(env.sim)                    # VERY IMPORTANT!
    rb_states = gymtorch.wrap_tensor(rb_state_tensor)           # 形状: (num_envs * num_bodies, 13)
    num_envs   = env.num_envs            # 例如 32
    rb_states = rb_states.view(num_envs, -1, 13)
    pos_w = rb_states[env_idx, wrist_body_index, 0:3].cpu()
    cur_wrist_pos = np.array([pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    pos_err = target_pos - cur_wrist_pos  # numpy (3,) current - target error vector
    err_norm = np.linalg.norm(pos_err) # error norm


    actor_jacobian = env.gym.acquire_jacobian_tensor(
        env.sim, env.cfg.asset.name
    )  # 拿到当前全身雅可比，并提取末端对机械臂关节的 6×7 子雅可比
    env.gym.refresh_jacobian_tensors(env.sim)
    whole_jac = gymtorch.wrap_tensor(actor_jacobian)
    j_eef = whole_jac[env_idx, wrist_body_index, :6, arm_joint_indices]  # [6,7]，Tensorprin
    # print("j_eef:", j_eef.shape, j_eef)
    dpos = torch.zeros(6, device=device)  # 构建 dpose = [pos_err; zeros(3)]，并转成 Tensor (6,1)
    dpos[0:3] = torch.from_numpy(pos_err).to(device).float()
    dpose = dpos.unsqueeze(-1)  # [6,1]     # orientation 误差置 0（不关心姿态）

    #  Δq = J^T (J J^T + λ² I)^(-1) dpose
    J = j_eef  # [6,7]
    JJT = J @ J.T  # [6,6]
    lambda_sq = 0.05**2
    reg = torch.eye(6, device=device) * lambda_sq  # [6,6]
    inv_term = torch.inverse(JJT + reg)  # [6,6]
    dq = (J.T @ inv_term @ dpose).squeeze(-1)  # [7]
    q_new = q_init.cpu() + dq.cpu().numpy() * 0.2  # numpy (7,)
    return q_new, err_norm


def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # env_cfg.env.num_envs = 1
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_body_displacement = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.use_random = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.asset.self_collision = 0
    env_cfg.env.upper_teleop = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Acquire rigid body state tensor
    # This tensor is populated by the simulation state
    _rb_states = env.gym.acquire_rigid_body_state_tensor(env.sim)
    # Wrap the tensor so we can use it with PyTorch
    rb_states_pt = gymtorch.wrap_tensor(_rb_states) # Shape: (num_envs * num_bodies_per_env, 13)
    # Get the number of rigid bodies per robot asset
    num_links_per_robot = env.num_bodies 

    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    env.commands[:, 4] = height
    env.action_curriculum_ratio = 0
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(
        device=env.device
    )  # Use this to load from trained pt file

    # Initialize rigid body state tensor
    _rb_states = env.gym.acquire_rigid_body_state_tensor(env.sim)
    rb_states_pt = gymtorch.wrap_tensor(_rb_states) # (num_envs * num_links_per_robot, 13)
    num_links_per_robot = env.num_bodies # Assuming env.num_bodies gives the number of links per robot

    # init upper body joint
    waist_yaw_joint = torch.zeros(env.num_envs, 1, device=env.device)  # (B,1)
    left_arm_joint = torch.zeros(env.num_envs, 7, device=env.device)  # (B,7)
    right_arm_joint = torch.zeros(env.num_envs, 7, device=env.device)  # (B,7)

    ##check shoulder pitch link
    actor_handle = env.actor_handles[0]
    rb_names = env.gym.get_actor_rigid_body_names(
        env.envs[0], actor_handle
    )  # link names
    rb_shoulder_index = rb_names.index("right_shoulder_pitch_link")
    rb_wrist_index = rb_names.index("right_wrist_yaw_link")



    device = env.device
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    # cpu_inds = torch.arange(env.num_envs, dtype=torch.int32)
    # env.reset_idx(cpu_inds)


    arrive_tollerance = 0.3  # pos tollerance
    heading_thresh = 0.32  # yaw 30 degree tollerance
    max_speed = 1  # max linear speed
    arm_end_target_err = 0.1

    viewer = env.viewer
    env.gym.clear_lines(viewer)
    B      = env.num_envs

    torch.manual_seed(42)  # 这里设置种子为42
    target_pos = torch.empty(B, 3).to(device)
    root = env.root_states          # shape = (B, 13)，B = 并行环境数
    # pos_xyz     = root[:, 0:3].clone()      # (B, 3)  所有机器人的 (x,y,z)
    pos_xyz = root[:, 0:3].clone().to(device)
    target_pos[:, :2] = torch.rand(B, 2) * 3 - 1.5# x,y
    #每个机器人中心作为目标范围的中心
    target_pos[:, 0] += pos_xyz[:, 0]  # x
    target_pos[:, 1] += pos_xyz[:, 1]  # y
    target_pos[:, 2] = torch.rand(B) + 0.3  # height

    arm_joint_indices = torch.arange(26, 33, dtype=torch.long, device=env.device)

    clock = pygame.time.Clock()  # Initialize the clock for FPS calculation
    # Draw a cross at each target position for every environment
    err_norms = torch.zeros(B, device=env.device, dtype=torch.float32)
    need_reset = torch.zeros(B, device=env.device, dtype=torch.bool)
    for i in range(B):
        draw_target_cross(env, viewer, target_pos[i,:])

    # --- 数据采集：初始化 ---
    # 用于存储每个时间帧收集到的数据字典列表
    collected_frames_data = [] 
    # --- 数据采集：结束初始化 ---
    try:
        for _ in range(30 * int(env.max_episode_length)):
            gym = env.gym
            env_handle = env.envs[0]  # 这里只取第 0 个并行环境
            actor = actor_handle  # 你之前已拿到的 actor_handle
            # The following lines for single actor rb_states are not needed for batched collection
            # rb_states = gym.get_actor_rigid_body_states(env_handle, actor, gymapi.STATE_ALL)
            # shoulder_state = rb_states[rb_shoulder_index]
            # pos_w = shoulder_state[0][0]
            # rot_w = shoulder_state[0][1]
            fps = clock.get_fps()
            print("fps: ",fps)
            current_pos_xy = env.root_states[:, :2]  # 只取 x,y
            qx, qy, qz, qw = env.root_states[:, 3:7].unbind(dim=1)  # 快速拆 4 列，各 (B,)
            # cal_yaw
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw_currrent = torch.atan2(siny_cosp, cosy_cosp)

            target_pos_xy = target_pos[:, :2]  # 只取 x,y
            dx = target_pos_xy[:, 0].to(device) - current_pos_xy[:, 0].to(device)
            dy = target_pos_xy[:, 1].to(device) - current_pos_xy[:, 1].to(device)

            yaw_target = torch.atan2(dy, dx)
            dtheta = torch.atan2(torch.sin(yaw_target - yaw_currrent), torch.cos(yaw_target - yaw_currrent))
            dist = torch.linalg.norm(target_pos_xy - current_pos_xy, dim=1)


            dist_ratio = dist / (dist + 1.0)
            vx_nominal = dist_ratio * max_speed                    # (B,)
            yaw      = torch.clamp(dtheta * 3.0, -1.0, 1.0)
            height_tar = torch.clamp(target_pos[:, 2], 0.3, 0.8)   # (B,)
            idle_height = torch.full((B,), 0.75, device=device)

            need_turn     = (torch.abs(dtheta) > heading_thresh)      # (B,)
            need_approach = (dist > arrive_tollerance)                       # (B,)
            no_need_turn   = torch.abs(dtheta) < heading_thresh+0.2
            no_need_approach = ~need_approach

            yaw_cmd  = torch.where(need_turn, yaw, torch.zeros_like(yaw))
            vx_cmd  = torch.where(need_approach, vx_nominal,torch.zeros_like(vx_nominal))
            height_cmd  = torch.where(no_need_turn&no_need_approach,  height_tar, idle_height)
            
            pos_z     = env.root_states[:, 2]      # (B, 1)所有机器人的 (z)
            eps          = 5e-2
            no_need_height =torch.abs(pos_z - height_cmd) < eps

            need_arm_ik=torch.where(no_need_turn&no_need_approach&no_need_height,True,False) 

            ik_cal_idx  = torch.nonzero(need_arm_ik).squeeze(1)  # (k,)

            if ik_cal_idx.numel():
                n= ik_cal_idx.numel()
                for env_idx in ik_cal_idx:
                    target_pos_ik = target_pos[env_idx, :].cpu().numpy()  # 之后转为gpu计算
        
                    dof_states = gym.acquire_dof_state_tensor(env.sim)
                    dof_state_tensor = gymtorch.wrap_tensor(dof_states)
                    gym.refresh_dof_state_tensor(env.sim)          # ← 一定别忘了刷新！
                    dof_state_tensor = dof_state_tensor.view(env.num_envs,-1 ,2)
                    q_init_7dof = dof_state_tensor[env_idx, 20:27, 0] 

                    # rigid_body_states = gym.acquire_rigid_body_state_tensor(env.sim)
   

                    # dof_states = gym.get_actor_dof_states(env.envs[env_idx], env.actor_handles[env_idx], gymapi.STATE_POS)
                    # right_arm_pos = dof_states["pos"][20:27]
                    # q_init_7dof = right_arm_pos #current arm joint position

                    q_new, err_norm = solve_right_arm_ik_jacobian(
                            env,
                            env_idx,
                            env.actor_handles[env_idx],
                            rb_wrist_index,
                            arm_joint_indices,
                            target_pos_ik,
                            q_init_7dof,
                        )  #  arm joint get
                    # print("q_new:", q_new)
                    right_arm_joint[env_idx,:] = torch.tensor(q_new, dtype=torch.float32) * 4
                    err_norms[env_idx]= err_norm
                    if err_norms[env_idx] < arm_end_target_err:
                        need_reset[env_idx] = True
            else:
                err_norms[:] = float('inf')

            env.commands[:, 0] = vx_cmd
            env.commands[:, 1] = torch.zeros(B, dtype=torch.float32)
            env.commands[:, 2] = yaw_cmd
            env.commands[:, 4] = height_cmd  # height

            actions = policy(obs.detach())
            left_arm_joint = left_arm_joint.view(B, -1)
            right_arm_joint = right_arm_joint.view(B, -1)
            waist_yaw_joint = waist_yaw_joint.view(B, -1)
            actions = torch.cat([actions, waist_yaw_joint, left_arm_joint, right_arm_joint], dim=1)
            obs, reward, _, done, *_ = env.step(actions.detach())  # reset中也会调用

            # Refresh rigid body state tensor
            env.gym.refresh_rigid_body_state_tensor(env.sim)

            reset_ids  = torch.nonzero(need_reset).squeeze(1)  # (k,)
            if reset_ids.numel():
                print("resetting:")
                # env.reset_idx(reset_ids.cpu())        # 物理重置
                n= reset_ids.numel()
                right_arm_joint[reset_ids,:] = torch.zeros(7, dtype=torch.float32,device=device)
                height_cmd[reset_ids]     = 0.75
                target_pos[reset_ids, 2] = torch.rand(n, device=device)*0.2+0.9  # height
                target_pos[reset_ids, :2] = torch.rand(n, 2, device=device) * 3 - 1.5# x,y
                target_pos[reset_ids, 0] += pos_xyz[reset_ids, 0]  # x
                target_pos[reset_ids, 1] += pos_xyz[reset_ids, 1]  # y
                need_reset[reset_ids] = False    # 重置后不再满足“到达”条件   
                env.gym.clear_lines(viewer)
                for i in range(B):
                    draw_target_cross(env, viewer, target_pos[i,:])
            clock.tick(60)
            
            # ------- 数据采集与处理（在 env.step() 之后）--------
            # --- 数据采集：获取时间戳 ---
            wall_time_current = time.time()  # Unix 时间戳
            sim_time_current = env.gym.get_sim_time(env.sim)  # 仿真时间
            # --- 数据采集：结束获取时间戳 ---

            # --- 数据采集：存储当前帧数据 ---
            # 注意：所有PyTorch张量在存入前都通过 .clone().cpu().numpy() 转换为NumPy数组
            frame_data = {
                'sim_time': sim_time_current,  # 当前仿真时间 (标量)
                'wall_time': wall_time_current, # 当前墙上时间 (标量)
                'root_states': env.root_states.clone().cpu().numpy(),      # 基座状态 (B, 13)
                'dof_pos': env.dof_pos.clone().cpu().numpy(),              # 关节角度 (B, num_dof)
                'dof_vel': env.dof_vel.clone().cpu().numpy(),              # 关节速度 (B, num_dof)
                # MODIFICATION: Detach actions tensor before converting to numpy
                'applied_actions': actions.clone().detach().cpu().numpy(),          # 应用的动作 (B, num_actions_total)
                'current_target_pos': target_pos.clone().cpu().numpy(),    # 当前目标位置 (B, 3)
                'ik_err_norms': err_norms.clone().cpu().numpy(),           # IK误差范数 (B,)
                'base_commands': env.commands.clone().cpu().numpy(),       # 基座指令 (B, num_base_commands)
                'done_flags': done.clone().cpu().numpy(),                  # 环境done标志 (B,)
                'need_reset_flags_eval': need_reset.clone().cpu().numpy(),  # 评估时的need_reset标志 (B,)
                # ADDED: Collect all link states
                'link_positions': rb_states_pt.view(env.num_envs, num_links_per_robot, 13)[:, :, 0:3].clone().detach().cpu().numpy(), # (B, num_links, 3)
                'link_linear_velocities': rb_states_pt.view(env.num_envs, num_links_per_robot, 13)[:, :, 7:10].clone().detach().cpu().numpy(), # (B, num_links, 3)
                'link_angular_velocities': rb_states_pt.view(env.num_envs, num_links_per_robot, 13)[:, :, 10:13].clone().detach().cpu().numpy() # (B, num_links, 3)
            }
            collected_frames_data.append(frame_data)
            # --- 数据采集：结束存储当前帧数据 ---
        # ------- 数据采集与处理结束 --------
    # 仿真主循环结束
    finally:
        # --- 数据采集：处理并保存所有收集的数据 ---
        if collected_frames_data:
            collated_data = {}
            # 从第一帧获取键名，假设所有帧结构相同
            # 这些键对应每帧变化的标量数据
            data_keys_per_frame_scalar = ['sim_time', 'wall_time']
            # 这些键对应每帧变化的、带批量维度B的数据
            # MODIFICATION: Add new keys for link states
            data_keys_per_frame_batched = ['root_states', 'dof_pos', 'dof_vel', 'applied_actions', 
                                        'current_target_pos', 'ik_err_norms', 'base_commands', 
                                        'done_flags', 'need_reset_flags_eval',
                                        'link_positions', 'link_linear_velocities', 'link_angular_velocities']

            for key in data_keys_per_frame_scalar:
                collated_data[key] = np.array([frame[key] for frame in collected_frames_data])
                # 结果形状: (num_timesteps,)

            for key in data_keys_per_frame_batched:
                # 沿新的时间维度 (axis 0) 堆叠
                collated_data[key] = np.stack([frame[key] for frame in collected_frames_data], axis=0)
                # 结果形状: (num_timesteps, B, num_features_for_key)

            # --- 数据采集：定义文件名并保存 ---
            # LEGGED_GYM_ROOT_DIR, os, time, np 应该已在文件顶部导入
            output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'collected_data_play')
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            data_filename = os.path.join(output_dir, f"homie_play_data_{timestamp}.npz")
            
            np.savez_compressed(data_filename, **collated_data)
            print(f"收集的数据已保存到: {data_filename}")
            # --- 数据采集：结束定义文件名并保存 ---
        else:
            print("没有收集到数据可供保存。")
    # --- 数据采集：处理并保存所有收集的数据结束 ---

if __name__ == "__main__":
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.75)

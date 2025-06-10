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
    rb_states_np = env.gym.get_actor_rigid_body_states(
         env.envs[env_idx], actor_handle, gymapi.STATE_ALL
    )
    pos_w = rb_states_np[wrist_body_index][0][0]  # (px, py, pz)
    cur_wrist_pos = np.array([pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    pos_err = target_pos - cur_wrist_pos  # numpy (3,) current - target error vector
    err_norm = np.linalg.norm(pos_err) # error norm


    actor_jacobian = env.gym.acquire_jacobian_tensor(
        env.sim, env.cfg.asset.name
    )  # 拿到当前全身雅可比，并提取末端对机械臂关节的 6×7 子雅可比
    env.gym.refresh_jacobian_tensors(env.sim)
    whole_jac = gymtorch.wrap_tensor(actor_jacobian)
    j_eef = whole_jac[env_idx, wrist_body_index, :6, arm_joint_indices]  # [6,7]，Tensorprin

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
    q_new = q_init + dq.cpu().numpy() * 1  # numpy (7,)
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



    # env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    cpu_inds = torch.arange(env.num_envs, dtype=torch.int32)
    env.reset_idx(cpu_inds)


    arrive_tollerance = 0.3  # pos tollerance
    heading_thresh = 0.32  # yaw 30 degree tollerance
    max_speed = 1  # max linear speed
    arm_end_target_err = 0.1

    viewer = env.viewer
    env.gym.clear_lines(viewer)
    B      = env.num_envs

    torch.manual_seed(42)  # 这里设置种子为42
    target_pos = torch.empty(B, 3)

    root = env.root_states          # shape = (B, 13)，B = 并行环境数
    pos_xyz     = root[:, 0:3].clone()      # (B, 3)  所有机器人的 (x,y,z)
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

    for _ in range(30 * int(env.max_episode_length)):
        gym = env.gym
        env_handle = env.envs[0]  # 这里只取第 0 个并行环境
        actor = actor_handle  # 你之前已拿到的 actor_handle
        rb_states = gym.get_actor_rigid_body_states(env_handle, actor, gymapi.STATE_ALL)
        shoulder_state = rb_states[rb_shoulder_index]
        pos_w = shoulder_state[0][0]
        rot_w = shoulder_state[0][1]



        # Calculate and display FPS
        fps = clock.get_fps()

        reset = False
        current_pos_xy = env.root_states[:, :2]  # 只取 x,y
        qx, qy, qz, qw = env.root_states[:, 3:7].unbind(dim=1)  # 快速拆 4 列，各 (B,)
        # cal_yaw
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw_currrent = torch.atan2(siny_cosp, cosy_cosp)

        target_pos_xy = target_pos[:, :2]  # 只取 x,y
        dx = target_pos_xy[:, 0] - current_pos_xy[:, 0]
        dy = target_pos_xy[:, 1] - current_pos_xy[:, 1]

        yaw_target = torch.atan2(dy, dx)

        dtheta = torch.atan2(
            torch.sin(yaw_target - yaw_currrent), torch.cos(yaw_target - yaw_currrent)
        )

        dist = torch.linalg.norm(target_pos_xy - current_pos_xy, dim=1)



        device = env.device
        dist_ratio = dist / (dist + 1.0)
        vx_nominal = dist_ratio * max_speed                    # (B,)
        yaw      = torch.clamp(dtheta * 3.0, -1.0, 1.0)
        height_tar = torch.clamp(target_pos[:, 2], 0.3, 0.8)   # (B,)
        idle_height = torch.full((B,), 0.75, device=device)

        # --- masks ---
        need_turn     = (torch.abs(dtheta) > heading_thresh)      # (B,)
        need_approach = (dist > arrive_tollerance)                       # (B,)
        no_need_turn   = torch.abs(dtheta) < heading_thresh+0.2
        no_need_approach = ~need_approach
        # --- yaw ---

        #if need turn：
        yaw_cmd  = torch.where(need_turn, yaw, torch.zeros_like(yaw))
        # if need turn and need approach：

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

                dof_states = gym.get_actor_dof_states(env.envs[0], env.actor_handles[0], gymapi.STATE_POS)
                right_arm_pos = dof_states["pos"][20:27]
                q_init_7dof = right_arm_pos #current arm joint position

                q_new, err_norm = solve_right_arm_ik_jacobian(
                        env,
                        env_idx,
                        env.actor_handles[env_idx],
                        rb_wrist_index,
                        arm_joint_indices,
                        target_pos_ik,
                        q_init_7dof,
                    )  #  arm joint get
                right_arm_joint[env_idx,:] = torch.tensor(q_new, dtype=torch.float32) * 4
                err_norms[env_idx]= err_norm
                if err_norms[env_idx] < arm_end_target_err:
                    need_reset[env_idx] = True
        else:
            err_norms[:] = float('inf')

    

        
        # if abs(dtheta) > heading_thresh:
        #     yaw_cmd = max(-1.0, min(1.0, dtheta * 2))
        #     if dist > arrive_tollerance:
        #         vx_cmd = dist / (dist + 1) * max_speed
        #     else:
        #         vx_cmd = np.zeros(1, dtype=np.float32)  # 到了就停
        # else:
        #     vx_cmd = dist / (dist + 1) * max_speed

        # if dist < 0.3:
        #     vx_cmd = 0
        #     if dtheta < 0.2:
        #         height_cmd = max(0.4, min(0.9, target_pos[2]))
        #         yaw_cmd = 0
        #         q_new, err_norm = solve_right_arm_ik_jacobian(
        #             env,
        #             env_handle,
        #             actor_handle,
        #             rb_wrist_index,
        #             arm_joint_indices,
        #             target_pos,
        #             q_init_7dof,
        #         )  #  arm joint get
        #         right_arm_joint = torch.tensor(q_new, dtype=torch.float32) * 4
        #         if err_norm < 0.1:
        #             reset = True
        #             print("arrive at target")
        #     else:
        #         yaw_cmd = max(-1.0, min(1.0, dtheta * 2))

        # right_arm_joint = torch.zeros(B,7,dtype=torch.float32)


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



        reset_ids  = torch.nonzero(need_reset).squeeze(1)  # (k,)

        if reset_ids.numel():
            print("resetting====================================:")
            # env.reset_idx(reset_ids.cpu())        # 物理重置
            n= reset_ids.numel()
            right_arm_joint[reset_ids,:] = torch.zeros(7, dtype=torch.float32)
            height_cmd[reset_ids]     = 0.75

            target_pos[reset_ids, 2] = torch.rand(n, device=device)+0.3  # height
            target_pos[reset_ids, :2] = torch.rand(n, 2, device=device) * 3 - 1.5# x,y
            target_pos[reset_ids, 0] += pos_xyz[reset_ids, 0]  # x
            target_pos[reset_ids, 1] += pos_xyz[reset_ids, 1]  # y

            need_reset[reset_ids] = False    # 重置后不再满足“到达”条件   
            env.gym.clear_lines(viewer)
            for i in range(B):
                draw_target_cross(env, viewer, target_pos[i,:])
                

        clock.tick(60)
        # ------- 数据采样与打印（插在 env.step() 之后）---------------------------------------------------------------------------------------------
        wall_time = time.time()  # Unix 时间戳
        sim_time = env.gym.get_sim_time(env.sim)  # 仿真时间
        fps_val = clock.get_fps()  # 实时 FPS
        # print("fps:",fps_val)

        # base（机体）13 维：pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # 全批量张量保留在 GPU（建议）
        # root = env.root_states          # shape = (B, 13)，B = 并行环境数

        # pos_xyz     = root[:, 0:3]      # (B, 3)  所有机器人的 (x,y,z)
        # quat_xyzw   = root[:, 3:7]      # (B, 4)  (qx,qy,qz,qw)
        # lin_vel_xyz = root[:, 7:10]     # (B, 3)  线速度
        # ang_vel_xyz = root[:, 10:13]    # (B, 3)  角速度


        # 关节 v w
        dof_states = env.gym.get_actor_dof_states(
            env_handle, actor_handle, gymapi.STATE_ALL
        )

        q = dof_states["pos"]  # 关节角
        qd = dof_states["vel"]  # 关节速度

        link_states = env.gym.get_actor_rigid_body_states(
            env_handle, actor_handle, gymapi.STATE_POS | gymapi.STATE_VEL
        )
        # positions = link_states["pos"]                  # 关节位置
        # velocities = link_states["vel"]                 # 关节速度



if __name__ == "__main__":
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.75)

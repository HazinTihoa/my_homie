import math
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import time
import torch

from isaacgym import gymapi
from isaacgym import gymtorch
import torch.nn.functional as F
import h5py
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


def _save_episode(buffer: dict, epi_id: int, out_dir: str):
    fname = os.path.join(out_dir, f"episode_{epi_id}.hdf5")
    with h5py.File(fname, "w") as h5file:
        obs_grp = h5file.require_group("observations")
        h5file.create_dataset("upper_actions", data=np.stack(buffer["upper_actions"], 0), compression="gzip")
        h5file.create_dataset("cmd_vel_and_height", data=np.stack(buffer["cmd_vel_and_height"], 0), compression="gzip")
        obs_grp.create_dataset("qpos",         data=np.stack(buffer["qpos"], 0), compression="gzip")
        obs_grp.create_dataset("qvel",         data=np.stack(buffer["qvel"], 0), compression="gzip")
        obs_grp.create_dataset("eef_to_goal",  data=np.stack(buffer["eef_to_goal"], 0), compression="gzip")
        obs_grp.create_dataset("obs_vel_and_height",  data=np.stack(buffer["obs_vel_and_height"], 0), compression="gzip")
    print(f"wrote episode to {fname}")

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


    arrive_tollerance = 0.2  # pos tollerance
    heading_thresh = 0.16  # yaw 30 degree tollerance
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

    tar_range=6
    hei_coe= 1.0
    hei_bias= 0.3
    # 生成目标位置
    target_pos[:, :2] = torch.rand(B, 2) * tar_range -tar_range/2# x,y
    target_pos[:, 0] += pos_xyz[:, 0] # x
    target_pos[:, 1] += pos_xyz[:, 1]# y
    target_pos[:, 2] = torch.rand(B)*hei_coe + hei_bias  # height

    arm_joint_indices = torch.arange(26, 33, dtype=torch.long, device=env.device)
    # Draw a cross at each target position for every environment
    err_norms = torch.zeros(B, device=env.device, dtype=torch.float32)
    need_reset = torch.zeros(B, device=env.device, dtype=torch.bool)
    for i in range(B):
        draw_target_cross(env, viewer, target_pos[i,:])

    
    _dof  = env.gym.acquire_dof_state_tensor(env.sim)
    _rb   = env.gym.acquire_rigid_body_state_tensor(env.sim)
    _jac  = env.gym.acquire_jacobian_tensor(env.sim, env.cfg.asset.name)


   # -------- 2. 打开 HDF5 文件 --------
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "act_dataset")
    os.makedirs(out_dir, exist_ok=True)
    # h5_path = os.path.join(out_dir, time.strftime("%Y%m%d-%H%M%S_act.hdf5"))
    # h5f = h5py.File(h5_path, "w")
    # print(f"[ACT‑dataset] writing to {h5_path}")

    # 每个并行 env 对应一个缓冲区 & 计数器
    epi_buf = [dict(upper_actions=[], cmd_vel_and_height=[], qpos=[], eef_to_goal=[],qvel=[],obs_vel_and_height=[]) for _ in range(B)]
    global_epi = 0  # 全局 episode id

    # indices helpers
    rb_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    wrist_idx = rb_names.index("right_wrist_yaw_link")


    obs = env.get_observations()
    env.reset_idx(torch.arange(B, device=device))

    # acquire tensors once
    _rb = env.gym.acquire_rigid_body_state_tensor(env.sim)
     # -------- 2. 打开 HDF5 文件 --------
    current_time = time.time()
    try:
        for _ in range(30 * int(env.max_episode_length)):
            flush_start_time1 = time.time()

            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_jacobian_tensors(env.sim)
            t_sim0 = env.gym.get_sim_time(env.sim)
            dof_tensor = gymtorch.wrap_tensor(_dof).view(B, -1, 2)         # (B, D, 2)
            rb_tensor  = gymtorch.wrap_tensor(_rb ).view(B, -1, 13)      # (B, L, 13)
            jacobian   = gymtorch.wrap_tensor(_jac).view(B, env.num_bodies, 6, -1)  # (B, L, 6, D)




            current_pos_xy = env.root_states[:, :2]  # 只取 x,y
            qx, qy, qz, qw = env.root_states[:, 3:7].unbind(dim=1)  # 快速拆 4 列，各 (B,)
            # cal_yaw
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw_currrent = torch.atan2(siny_cosp, cosy_cosp)

            target_pos_xy = target_pos[:, :2]  # 只取 x,y
            dx = target_pos_xy[:, 0].to(device) - current_pos_xy[:, 0].to(device)
            dy = target_pos_xy[:, 1].to(device) - current_pos_xy[:, 1].to(device)
            
            pos_z     = env.root_states[:, 2]      # (B, 1)所有机器人的 (z)
            eps          = 5e-2
            

            yaw_target = torch.atan2(dy, dx)
            dtheta = torch.atan2(torch.sin(yaw_target - yaw_currrent), torch.cos(yaw_target - yaw_currrent))
            dist = torch.linalg.norm(target_pos_xy - current_pos_xy, dim=1)

            dist_ratio = dist / (dist + 1.0)
            vx_nominal = dist_ratio * max_speed                    # (B,)
            yaw      = torch.clamp(dtheta * 3.0, -1.0, 1.0)
            height_tar = torch.clamp(target_pos[:, 2], 0.2, 0.8)   # (B,)
            idle_height = torch.full((B,), 0.75, device=device)

            need_turn     = (torch.abs(dtheta) >= heading_thresh)      # (B,)
            need_approach = (dist >= arrive_tollerance)                       # (B,)
            no_need_turn   = torch.abs(dtheta) <= heading_thresh+0.3
            no_need_approach = dist <= arrive_tollerance+0.1

            yaw_cmd  = torch.where(need_turn, yaw, torch.zeros_like(yaw))

            vx_cmd  = torch.where(need_approach, vx_nominal,torch.zeros_like(vx_nominal))
            height_cmd  = torch.where(no_need_turn&no_need_approach,  height_tar, pos_z)
            

            no_need_height =torch.abs(pos_z - height_cmd) < eps

            need_arm_ik=torch.where(no_need_turn&no_need_approach&no_need_height,True,False) 

            ik_cal_idx  = torch.nonzero(need_arm_ik).squeeze(1)  # (k,)

            # 打印第一个机器人的状态信息
            # state_msg = []
            # if need_turn[0]:
            #     state_msg.append("需要转向")
            # else:
            #     state_msg.append("无需转向")
            # if need_approach[0]:
            #     state_msg.append("需要靠近")
            # else:
            #     state_msg.append("无需靠近")
            # if not no_need_height[0]:
            #     state_msg.append("需要高度调整")
            # else:
            #     state_msg.append("高度已达标")
            # print(f"[调试] 机器人0 状态: {', '.join(state_msg)} yaw差: {dtheta[0]:.3f}, 距离: {dist[0]:.3f}, 高度: {pos_z[0]:.3f} 目标高度: {height_cmd[0]:.3f}")

            
            flag=True
            if flag:
                if ik_cal_idx.numel():
                    n= ik_cal_idx.numel()
                    for env_idx in ik_cal_idx:
                        target_pos_ik = target_pos[env_idx, :].cpu().numpy()  # 之后转为gpu计算
            
                        dof_states = env.gym.acquire_dof_state_tensor(env.sim)
                        dof_state_tensor = gymtorch.wrap_tensor(dof_states)
                        env.gym.refresh_dof_state_tensor(env.sim)          # ← 一定别忘了刷新！
                        dof_state_tensor = dof_state_tensor.view(env.num_envs,-1 ,2)
                        q_init_7dof = dof_state_tensor[env_idx, 20:27, 0] 

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
            else:

                if ik_cal_idx.numel():
                    k = ik_cal_idx.numel()

                    # --- (1) 收集状态 ----------------------------------------------------
                    q_init_7dof = dof_tensor[ik_cal_idx, 20:27, 0]               # (k, 7)
                    wrist_pos   = rb_tensor[ik_cal_idx, rb_wrist_index, 0:3]     # (k, 3)
                    target_xyz  = target_pos[ik_cal_idx]                         # (k, 3)
                    pos_err     = target_xyz - wrist_pos                         # (k, 3)
                    err_norms[ik_cal_idx] = torch.linalg.norm(pos_err, dim=1)

                    # --- (2) 构造 6×7 Jacobian -----------------------------------------
                    J = jacobian[ik_cal_idx, rb_wrist_index, :6]                 # (k, 6, dof)
                    J = J[:, :, arm_joint_indices].contiguous()                  # (k, 6, 7)

                    # --- (3) Damped Least Squares 求 Δq -------------------------------
                    lambda_sq = 0.05 ** 2
                    JJT       = torch.matmul(J, J.transpose(-1, -2))             # (k, 6, 6)
                    reg_eye   = torch.eye(6, device=device).expand(k, 6, 6) * lambda_sq
                    inv_term  = torch.linalg.inv(JJT + reg_eye)                  # (k, 6, 6)

                    dpose6    = torch.zeros(k, 6, 1, device=device)
                    dpose6[:, 0:3, 0] = pos_err                                  # 位置误差
                    dq        = torch.matmul(J.transpose(-1, -2),
                                            torch.matmul(inv_term, dpose6)).squeeze(-1)  # (k, 7)

                    # --- (4) 写回动作 ---------------------------------------------------
                    right_arm_joint[ik_cal_idx] = (q_init_7dof + 0.2 * dq) * 4

                    # --- (5) 更新 need_reset ------------------------------------------
                    need_reset[ik_cal_idx] = err_norms[ik_cal_idx] < arm_end_target_err
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
            upper_actions = torch.cat([waist_yaw_joint, left_arm_joint, right_arm_joint], dim=1)

            # ---- 2.2 收集数据到缓冲区 ----
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            rb_tensor = gymtorch.wrap_tensor(_rb).view(B, -1, 13)
            s_vx = env.root_states[:, 7]  # (B,)
            s_wz = env.root_states[:, 12] # (B,)
            s_height = env.root_states[:, 2]   # (B,)

            wrist_pos = rb_tensor[0, wrist_idx, 0:3].cpu().numpy()
            goal_pos = target_pos[0].cpu().numpy()
            goal=goal_pos-wrist_pos
            # print("goal:",torch.norm(torch.tensor(goal))) 
            for i in range(B):
                # 动作 & qpos
                # cmd
                epi_buf[i]["upper_actions"].append(upper_actions[i].detach().cpu().numpy())
                cmd_vec = np.stack([vx_cmd[i].cpu().numpy(), yaw_cmd[i].cpu().numpy(), height_cmd[i].cpu().numpy()], axis=-1)
                epi_buf[i]["cmd_vel_and_height"].append(cmd_vec)
                # obs
                epi_buf[i]["qpos"   ].append(env.dof_pos[i].cpu().numpy())
                epi_buf[i]["qvel"   ].append(env.dof_vel[i].cpu().numpy())
                obs_vec = np.stack([s_vx[i].cpu().numpy(), s_wz[i].cpu().numpy(), s_height[i].cpu().numpy()], axis=-1)
                epi_buf[i]["obs_vel_and_height"].append(obs_vec)
                wrist_pos = rb_tensor[i, wrist_idx, 0:3].cpu().numpy()
                goal_pos = target_pos[i].cpu().numpy()
                epi_buf[i]["eef_to_goal"].append(goal_pos - wrist_pos)
           
            # ---- 步进 ----
            obs, reward, _, reset_buf, *_ = env.step(actions.detach())  # reset中也会调用

            # break 条件（采够 N episode 或 Ctrl‑C）
            if global_epi >= 5000:
                break
            
            # print("right_arm_joint:", right_arm_joint[0,:])
            reset_ids  = torch.nonzero(need_reset).squeeze(1)
            reset_buf = torch.nonzero(reset_buf).squeeze(1)
            
            if reset_buf.numel()>0:
                # env.reset_idx(reset_buf.cuda())        # 物理重置

                n= reset_buf.numel()
                right_arm_joint[reset_buf,:] = torch.zeros(7, dtype=torch.float32,device=device)
                height_cmd[reset_buf]     = 0.75
                target_pos[reset_buf, 2] = torch.rand(n, device=device)*hei_coe + hei_bias  # height
                target_pos[reset_buf, :2] = torch.rand(n, 2, device=device) * tar_range - tar_range/2# x,y
                target_pos[reset_buf, 0] += pos_xyz[reset_buf, 0]  # x
                target_pos[reset_buf, 1] += pos_xyz[reset_buf, 1]  # y


                env.gym.clear_lines(viewer)
                for i in range(B):
                    draw_target_cross(env, viewer, target_pos[i,:])

            if reset_ids.numel():
                # 1) 写盘 + 清空缓冲
                for idx in reset_ids.cpu().tolist():
                    if 0 < len(epi_buf[idx]['upper_actions']) <=1000:               # 防止空 episode
                        flush_start_time = time.time()
                        _save_episode(epi_buf[idx], global_epi, out_dir)                
                        consume_time = time.time()-flush_start_time
                
                        print(f"[ACT‑dataset] wrote episode {global_epi} with {len(epi_buf[idx]['upper_actions'])} steps,time cost: {consume_time:.6f} seconds")
                        global_epi += 1
                        for key in epi_buf[idx]:
                            epi_buf[idx][key] = []  
                        epi_buf[idx] = dict(upper_actions=[], cmd_vel_and_height=[], qpos=[], eef_to_goal=[],qvel=[],obs_vel_and_height=[])


                env.reset_idx(reset_ids.cuda())        # 物理重置

                n= reset_ids.numel()
                right_arm_joint[reset_ids,:] = torch.zeros(7, dtype=torch.float32,device=device)
                height_cmd[reset_ids]     = 0.75
                target_pos[reset_ids, 2] = torch.rand(n, device=device)*hei_coe + hei_bias  # height
                target_pos[reset_ids, :2] = torch.rand(n, 2, device=device) * tar_range - tar_range/2 # x,y
                target_pos[reset_ids, 0] += pos_xyz[reset_ids, 0]  # x
                target_pos[reset_ids, 1] += pos_xyz[reset_ids, 1]  # y

                need_reset[reset_ids] = False    # 重置后不再满足“到达”条件   
                env.gym.clear_lines(viewer)
                # for i in range(B):
                #     draw_target_cross(env, viewer, target_pos[i,:])

    finally:
 
        print(f"=========================")

    
if __name__ == "__main__":
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.75)
'''
python legged_gym/legged_gym/scripts/play_and_record.py --num_envs 1024 --headless
'''
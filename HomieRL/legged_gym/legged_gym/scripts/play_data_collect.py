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
    env_handle,
    actor_handle,
    wrist_body_index,
    arm_joint_indices,
    target_pos,
    q_init,
):

    device = env.device  # e.g. "cuda:0"
    rb_states_np = env.gym.get_actor_rigid_body_states(
        env_handle, actor_handle, gymapi.STATE_ALL
    )
    pos_w = rb_states_np[wrist_body_index][0][0]  # (px, py, pz)
    cur_wrist_pos = np.array([pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    pos_err = target_pos - cur_wrist_pos  # numpy (3,)
    err_norm = np.linalg.norm(pos_err)
    actor_jacobian = env.gym.acquire_jacobian_tensor(
        env.sim, env.cfg.asset.name
    )  # 拿到当前全身雅可比，并提取末端对机械臂关节的 6×7 子雅可比
    env.gym.refresh_jacobian_tensors(env.sim)
    whole_jac = gymtorch.wrap_tensor(actor_jacobian)
    j_eef = whole_jac[0, wrist_body_index, :6, arm_joint_indices]  # [6,7]，Tensorprin
    dpos = torch.zeros(
        6, device=device
    )  # 构建 dpose = [pos_err; zeros(3)]，并转成 Tensor (6,1)
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
    vx = vy = yaw = 0.0

    arrive_tollerance = 0.3  # pos tollerance
    heading_thresh = 0.1  # yaw tollerance
    max_speed = 1  # max linear speed
    arm_end_target_err = 0.1

    viewer = env.viewer
    env.gym.clear_lines(viewer)
    B      = env.num_envs

    root = env.root_states          # shape = (B, 13)，B = 并行环境数
    pos_xyz     = root[:, 0:3]      # (B, 3)  所有机器人的 (x,y,z)
    torch.manual_seed(42)  # 这里设置种子为42
    target_pos = torch.empty(B, 3)
    target_pos[:, :2] = torch.rand(B, 2) * 3 - 1.5# x,y
    
    #每个机器人中心作为目标范围的中心
    target_pos[:, 0] += pos_xyz[:, 0]  # x
    target_pos[:, 1] += pos_xyz[:, 1]  # y

    target_pos[:, 2] = torch.rand(B) + 0.3  # height

    q_init_7dof = np.zeros(7)
    arm_joint_indices = torch.arange(26, 33, dtype=torch.long, device=env.device)

    sim_step = 0
    clock = pygame.time.Clock()  # Initialize the clock for FPS calculation
    # Draw a cross at each target position for every environment
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

        dof_states = gym.get_actor_dof_states(
            env_handle, actor_handle, gymapi.STATE_POS
        )
        all_pos = dof_states["pos"]
        right_arm_pos = all_pos[20:27]
        q_init_7dof = right_arm_pos




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

        # --- masks ---
        need_turn     = (torch.abs(dtheta) > heading_thresh)      # (B,)
        need_approach = (dist > arrive_tollerance)                       # (B,)

        # --- yaw ---
        yaw      = torch.clamp(dtheta * 2.0, -1.0, 1.0)
        yaw_cmd  = torch.where(need_turn, yaw, torch.zeros_like(yaw))

        # --- vx ---
        vx_nominal = dist_ratio * max_speed                    # (B,)
        vx_cmd     = torch.where(need_approach, vx_nominal, torch.zeros_like(vx_nominal))

        # --- height ---
        height_tar = torch.clamp(target_pos[:, 2], 0.4, 0.9)   # (B,)
        # 若只在“近 + 已对正”时才让高度→目标，其余保持 0.75
        idle_height = torch.full((B,), 0.75, device=device)
        height_cmd  = torch.where(need_turn & need_approach, idle_height, height_tar)


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

        right_arm_joint = torch.zeros(B,7,dtype=torch.float32)
        vx = vx_cmd
        yaw = yaw_cmd

        env.commands[:, 0] = torch.tensor(vx, dtype=torch.float32)
        env.commands[:, 1] = torch.zeros(B, dtype=torch.float32)
        env.commands[:, 2] = torch.tensor(yaw, dtype=torch.float32)
        env.commands[:, 4] = torch.tensor(height_cmd, dtype=torch.float32)  # height

        actions = policy(obs.detach())

        left_arm_joint = left_arm_joint.view(B, -1)
        right_arm_joint = right_arm_joint.view(B, -1)
        waist_yaw_joint = waist_yaw_joint.view(B, -1)
        actions = torch.cat([actions, waist_yaw_joint, left_arm_joint, right_arm_joint], dim=1)

        obs, reward, _, done, *_ = env.step(actions.detach())  # reset中也会调用

        arrived_mask = (need_turn & need_approach)
        need_reset = arrived_mask | done          # (B,) bool
        reset_ids  = torch.nonzero(need_reset).squeeze(1)  # (k,)

        if reset_ids.numel():
            # env.reset_idx(reset_ids.cpu())        # 物理重置

            env.gym.clear_lines(viewer)
            right_arm_joint[reset_ids,:] = torch.zeros(7, dtype=torch.float32)
            height_cmd[reset_ids]     = 0.75

            # ④ 给这些 env 随机新目标
            n = reset_ids.numel()
            target_pos[reset_ids, :2] = torch.rand(n, 2, device=device) * 3 - 1.5
            target_pos[reset_ids,  2] = torch.rand(n,   device=device)   + 0.3

            arrived_mask[reset_ids] = False    # 重置后不再满足“到达”条件   
                    
            # Draw a cross at each target position for every environment
            for i in range(B):
                draw_target_cross(env, viewer, target_pos[i,:])


        # if done or reset:
        #     env.gym.clear_lines(viewer)
        #     right_arm_joint = torch.zeros(7, dtype=torch.float32)  # 重置右臂关节角度
        #     height_cmd = 0.75  # reset height

        #     target_pos[:, :2] = torch.rand(B, 2) * 3 - 1.5  # x,y
        #     target_pos[:, 2] = torch.rand(B) + 0.3  # height


        clock.tick(60)
        # ------- 数据采样与打印（插在 env.step() 之后）---------------------------------------------------------------------------------------------
        wall_time = time.time()  # Unix 时间戳
        sim_time = env.gym.get_sim_time(env.sim)  # 仿真时间
        fps_val = clock.get_fps()  # 实时 FPS

        # base（机体）13 维：pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        # 全批量张量保留在 GPU（建议）
        root = env.root_states          # shape = (B, 13)，B = 并行环境数

        pos_xyz     = root[:, 0:3]      # (B, 3)  所有机器人的 (x,y,z)
        quat_xyzw   = root[:, 3:7]      # (B, 4)  (qx,qy,qz,qw)
        lin_vel_xyz = root[:, 7:10]     # (B, 3)  线速度
        ang_vel_xyz = root[:, 10:13]    # (B, 3)  角速度


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

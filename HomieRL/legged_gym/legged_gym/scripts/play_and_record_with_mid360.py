import math
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import time
import torch
from concurrent.futures import ThreadPoolExecutor
import threading

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import torch.nn.functional as F
import h5py

# --- add LidarSensor to sys.path (manual path) ---
import sys, os
LIDAR_PATH = "/home/zhihaot/OmniPerception/LidarSensor"
sys.path.insert(0, LIDAR_PATH)
# === Lidar 相关 ===
import warp as wp
import trimesh
from LidarSensor.lidar_sensor import LidarSensor
from LidarSensor.sensor_config.lidar_sensor_config import LidarConfig
from LidarSensor import RESOURCES_DIR as LS_RESOURCES_DIR
from isaacgym.torch_utils import quat_apply, quat_mul
import numpy as np
from legged_gym.utils.PathLibNavigator import PathLibNavigator, NavigatorCfg
PATHS_PLY = "/home/zhihaot/my_homie/HomieRL/legged_gym/legged_gym/utils/path/paths.ply"
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

def draw_target_cross(env, viewer, target_pos, sphere_radius=0.08, sphere_color=(1.0, 1.0, 0.0), 
                      segments=8, rings=8):
    """
    在目标位置绘制球体标记
    
    Args:
        env: 环境对象
        viewer: 查看器对象
        target_pos: 目标位置 [x, y, z]
        sphere_radius: 球体半径 (默认0.08m)
        sphere_color: 球体颜色 RGB (默认红色)
        segments: 球体经线段数 (默认8)
        rings: 球体纬线环数 (默认8)
    """
    tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    
    # 创建球体几何体
    sphere_geom = gymutil.WireframeSphereGeometry(
        sphere_radius, segments, rings, None, color=sphere_color
    )
    
    # 创建球体位置
    sphere_pose = gymapi.Transform(gymapi.Vec3(tx, ty, tz), r=None)
    
    # 为每个环境绘制球体
    for env_handle in env.envs:
        gymutil.draw_lines(sphere_geom, env.gym, viewer, env_handle, sphere_pose)

def _save_episode(buffer: dict, epi_id: int, out_dir: str):
    """同步保存episode数据（在线程中调用）"""
    try:
        fname = os.path.join(out_dir, f"episode_{epi_id}.hdf5")
        t_start = time.time()

        with h5py.File(fname, "w") as h5file:
            obs_grp = h5file.require_group("observations")
            h5file.create_dataset("upper_actions", data=np.stack(buffer["upper_actions"], 0), compression="lzf")
            h5file.create_dataset("cmd_vel_and_height", data=np.stack(buffer["cmd_vel_and_height"], 0), compression="lzf")
            obs_grp.create_dataset("qpos",         data=np.stack(buffer["qpos"], 0), compression="lzf")
            obs_grp.create_dataset("qvel",         data=np.stack(buffer["qvel"], 0), compression="lzf")
            obs_grp.create_dataset("eef_to_goal",  data=np.stack(buffer["eef_to_goal"], 0), compression="lzf")
            obs_grp.create_dataset("obs_vel_and_height",  data=np.stack(buffer["obs_vel_and_height"], 0), compression="lzf")

            # 添加点云数据
            if "point_cloud" in buffer and len(buffer["point_cloud"]) > 0:
                obs_grp.create_dataset("point_cloud", data=np.stack(buffer["point_cloud"], 0).astype(np.float16), compression="lzf")

        elapsed = time.time() - t_start
        print(f"[ACT-dataset] Saved episode {epi_id} with {len(buffer['upper_actions'])} steps to {fname} (took {elapsed:.2f}s)")
    except Exception as e:
        print(f"[ERROR] Failed to save episode {epi_id}: {e}")

def world2base(vec_world, base_yaw):
    c, s = np.cos(-base_yaw), np.sin(-base_yaw)
    R = np.array([[c, -s], [s, c]])
    vec_xy = R @ vec_world[:2]
    vec_z = vec_world[2]
    vec_base = np.array([vec_xy[0], vec_xy[1], vec_z])  # 合成3维向量
    return vec_base

def make_esdf_query_fn(manager, device):
    H, W = manager.esdf.shape          # 网格高(H, 对应 y)/宽(W, 对应 x)
    res = manager.map_resolution       # 分辨率(米/格)
    ox, oy = manager.origin            # 左下角原点(世界坐标)
    def query(pts_bm3: torch.Tensor) -> torch.Tensor:
        x = pts_bm3[..., 0].detach().cpu().numpy()
        y = pts_bm3[..., 1].detach().cpu().numpy()
        # 世界坐标 → 栅格索引 (最近邻)
        i = np.floor((x - ox) / res).astype(np.int32)  # 列索引，对应 x
        j = np.floor((y - oy) / res).astype(np.int32)  # 行索引，对应 y
        # 出界裁剪到边界（等同 grid_sample 的 'border'）
        i = np.clip(i, 0, W - 1); j = np.clip(j, 0, H - 1)
        d = manager.esdf[j, i]                         # 查距离（正/负）
        # 回到原来的设备/精度
        return torch.from_numpy(d).to(pts_bm3.device, dtype=pts_bm3.dtype)
    return query

def make_esdf_query_fn_torch(manager, device):
    esdf = torch.from_numpy(manager.esdf).to(device)   # [H, W]
    H, W = esdf.shape
    res = manager.map_resolution
    ox, oy = map(float, manager.origin)

    def query(pts_bm3: torch.Tensor) -> torch.Tensor:
        # 连续栅格坐标（以像素左上角为 0）
        x = (pts_bm3[..., 0] - ox) / res
        y = (pts_bm3[..., 1] - oy) / res

        i0 = torch.floor(x).long().clamp_(0, W - 1)
        j0 = torch.floor(y).long().clamp_(0, H - 1)
        i1 = (i0 + 1).clamp_(0, W - 1)
        j1 = (j0 + 1).clamp_(0, H - 1)

        wx = (x - i0.float()).clamp_(0, 1)
        wy = (y - j0.float()).clamp_(0, 1)

        # 取四邻域
        d00 = esdf[j0, i0]
        d10 = esdf[j0, i1]
        d01 = esdf[j1, i0]
        d11 = esdf[j1, i1]

        # x 方向线性，再 y 方向线性
        d0 = d00 * (1 - wx) + d10 * wx
        d1 = d01 * (1 - wx) + d11 * wx
        d  = d0 * (1 - wy) + d1 * wy

        return d.to(dtype=pts_bm3.dtype)
    return query

def make_esdf_projector(manager, device, safe_clearance=0.2, step_gain=0.9, max_iters=30,z_range=None):
    """
    基于 ESDF 的目标点安全投影器：
    - 给定任意世界坐标点（B,3），沿 ESDF 梯度上升把点推到 clearance >= safe_clearance
    - 只移动 x,y；z 保持/可选夹紧
    参数:
        manager: 含 esdf(np.ndarray[H,W])、map_resolution(float)、origin(tuple[x0,y0]) 的对象
        device:  torch.device
        safe_clearance: 期望最小安全距离（米）
        step_gain: 每步的“目标增量比例”，取 (0,1]，越大收敛越快，但可能振荡
        max_iters: 迭代次数上限
    返回:
        projector(points_b3, z_range=None) -> (safe_points_b3, final_clearance_b)
    """
    esdf = torch.from_numpy(manager.esdf).to(device)  # [H, W]
    H, W = esdf.shape
    res = float(manager.map_resolution)
    ox, oy = map(float, manager.origin)

    # 世界->像素坐标（浮点）
    def world_to_pix(x, y):
        px = (x - ox) / res
        py = (y - oy) / res
        return px, py

    # 双线性采样 ESDF 值 & 像素坐标梯度（∂d/∂px, ∂d/∂py）
    def bilinear_value_and_pixgrad(px, py):
        # 邻域索引
        i0 = torch.floor(px).long().clamp_(0, W - 1)
        j0 = torch.floor(py).long().clamp_(0, H - 1)
        i1 = (i0 + 1).clamp_(0, W - 1)
        j1 = (j0 + 1).clamp_(0, H - 1)

        # 权重
        wx = (px - i0.float()).clamp_(0, 1)
        wy = (py - j0.float()).clamp_(0, 1)

        # 四邻域值
        d00 = esdf[j0, i0]
        d10 = esdf[j0, i1]
        d01 = esdf[j1, i0]
        d11 = esdf[j1, i1]

        # 双线性插值值
        d0 = d00 * (1 - wx) + d10 * wx
        d1 = d01 * (1 - wx) + d11 * wx
        d  = d0  * (1 - wy) + d1  * wy

        # 双线性函数的像素梯度（注意是对 px/py 的偏导）
        # ∂d/∂px = (1-wy)*(d10-d00) + wy*(d11-d01)
        # ∂d/∂py = (1-wx)*(d01-d00) + wx*(d11-d10)
        dd_dpx = (1 - wy) * (d10 - d00) + wy * (d11 - d01)
        dd_dpy = (1 - wx) * (d01 - d00) + wx * (d11 - d10)

        return d, dd_dpx, dd_dpy

    # 像素梯度 -> 世界坐标梯度（米^-1）
    # px = (x-ox)/res → ∂/∂x = (1/res)*∂/∂px
    def to_world_grad(dd_dpx, dd_dpy):
        inv_res = 1.0 / res
        gx = dd_dpx * inv_res
        gy = dd_dpy * inv_res
        return gx, gy

    # 将世界坐标裁剪到地图边界（避免越界）
    def clamp_to_map(x, y, margin_pix=1.0):
        # 允许留一点像素余量，默认 1 像素
        x_min = ox + margin_pix * res
        y_min = oy + margin_pix * res
        x_max = ox + (W - 1 - margin_pix) * res
        y_max = oy + (H - 1 - margin_pix) * res
        return x.clamp(x_min, x_max), y.clamp(y_min, y_max)

    def projector(points_b3: torch.Tensor, z_range=None):
        """
        参数:
            points_b3: [B,3] 世界坐标
            z_range: Optional[min_z, max_z]，若给定则把 z 夹紧到范围
        返回:
            safe_points_b3: [B,3]
            final_clearance_b: [B]
        """
        assert points_b3.dim() == 2 and points_b3.size(-1) == 3
        pts = points_b3.clone().to(device)

        if z_range is not None:
            z_min, z_max = float(z_range[0]), float(z_range[1])
            pts[:, 2] = pts[:, 2].clamp(z_min, z_max)

        for _ in range(max_iters):
            px, py = world_to_pix(pts[:, 0], pts[:, 1])
            d, dd_dpx, dd_dpy = bilinear_value_and_pixgrad(px, py)
            gx, gy = to_world_grad(dd_dpx, dd_dpy)  # 世界坐标梯度

            need = d < safe_clearance
            if not need.any():
                break  # 全部满足

            # 梯度方向（上升）
            g = torch.stack([gx, gy], dim=-1)  # [B,2]
            g_norm = torch.linalg.norm(g, dim=-1, keepdim=True).clamp_min(1e-8)

            # 对于需要移动的点，计算位移步长（按缺口比例）
            # 目标：本步大约提升 clearance 到 safe_clearance 的 step_gain 比例
            step = (safe_clearance - d).clamp_min(0.0) * step_gain  # [B]
            delta = (step.view(-1, 1) / g_norm) * g                 # [B,2]

            # 梯度过小（平坦区）时，给一个微小随机抖动方向以脱离平台
            flat = (g_norm.squeeze(-1) < 1e-6) & need
            if flat.any():
                jitter = torch.randn(flat.sum(), 2, device=device)
                jitter = jitter / torch.linalg.norm(jitter, dim=-1, keepdim=True).clamp_min(1e-6)
                delta[flat] = 0.05 * jitter  # 5cm 小抖动

            # 只更新不合格的点
            pts[:, 0] = torch.where(need, pts[:, 0] + delta[:, 0], pts[:, 0])
            pts[:, 1] = torch.where(need, pts[:, 1] + delta[:, 1], pts[:, 1])

            # 防越界裁剪
            pts[:, 0], pts[:, 1] = clamp_to_map(pts[:, 0], pts[:, 1])

            # 可选：若提供 z_range，继续夹紧 z
            if z_range is not None:
                pts[:, 2] = pts[:, 2].clamp(z_min, z_max)

        # 返回最终 clearance
        px, py = world_to_pix(pts[:, 0], pts[:, 1])
        d_final, _, _ = bilinear_value_and_pixgrad(px, py)
        return pts, d_final

    return projector

def Target_generate(pos_xyz, tar_range, z_range=(0.3, 1.3), manager=None, device=None, reset_indices=None):
    """
    生成目标点位置

    参数:
        pos_xyz: [B, 3]，起始位置
        tar_range: float，目标点xy的采样范围
        z_range: tuple (min_hei, max_hei)，高度范围
        manager: ESDF管理器，用于安全投影
        device: torch设备
        reset_indices: 需要重置的索引，如果为None则生成所有环境的目标点

    返回:
        target_pos: [B, 3]，目标点位置
    """
    B = pos_xyz.shape[0]
    min_hei, max_hei = z_range

    if reset_indices is not None:
        # 只为指定的环境生成新目标点
        n = reset_indices.numel()
        new_targets = torch.zeros(n, 3, device=device)
        new_targets[:, :2] = torch.rand(n, 2, device=device) * tar_range - tar_range / 2  # x, y
        new_targets[:, 0] += pos_xyz[reset_indices, 0]  # x
        new_targets[:, 1] += pos_xyz[reset_indices, 1]  # y
        new_targets[:, 2] = torch.rand(n, device=device) * (max_hei - min_hei) + min_hei  # height

        if manager is not None:
            projector = make_esdf_projector(manager, device, z_range=z_range)
            safe_targets, _ = projector(new_targets, z_range=z_range)
            return safe_targets
        else:
            return new_targets
    else:
        # 为所有环境生成目标点
        target_pos = torch.zeros(B, 3, device=device)
        target_pos[:, :2] = torch.rand(B, 2, device=device) * tar_range - tar_range / 2  # x, y
        target_pos[:, 0] += pos_xyz[:, 0]  # x
        target_pos[:, 1] += pos_xyz[:, 1]  # y
        target_pos[:, 2] = torch.rand(B, device=device) * (max_hei - min_hei) + min_hei  # height

        projector = make_esdf_projector(manager, device, z_range=z_range)
        safe_points, _ = projector(target_pos, z_range=z_range)
        return safe_points

     

    
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
    
    _rb_states = env.gym.acquire_rigid_body_state_tensor(env.sim)


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
 

    device = env.device
    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    # cpu_inds = torch.arange(env.num_envs, dtype=torch.int32)
    # env.reset_idx(cpu_inds)




    viewer = env.viewer
    B      = env.num_envs

    torch.manual_seed(42)  # 这里设置种子为42
    root = env.root_states          # shape = (B, 13)，B = 并行环境数
    pos_xyz = root[:, 0:3].clone().to(device)

    #== parameters ==
    arrive_tollerance = 0.3  # pos tollerance

    pos_bias=7.0  # 目标点距离机器人的基准距离
    z_range=(0.2, 1.0)
    tar_range=3.0  # 在基准距离基础上的随机范围

    # 在机器人周围随机方向生成偏移（可以是前后左右任意方向）
    random_angles = torch.rand(B, device=env.device) * 2 * 3.14159  # 0到2π随机角度
    random_radius = pos_bias  
    bias_xy = torch.stack([
        random_radius * torch.cos(random_angles),  # x偏移
        random_radius * torch.sin(random_angles),  # y偏移
        torch.zeros(B, device=env.device)           # z不偏移
    ], dim=1)

    safe_target_pos = Target_generate(pos_xyz + bias_xy, tar_range=tar_range, z_range=z_range, manager=env.esdf_manager, device=env.device)

    
    need_reset = torch.zeros(B, device=env.device, dtype=torch.bool)
    
    # env.gym.clear_lines(viewer)
    # for i in range(B):
    #     draw_target_cross(env, viewer, safe_target_pos[i,:])

    _dof  = env.gym.acquire_dof_state_tensor(env.sim)
    _rb   = env.gym.acquire_rigid_body_state_tensor(env.sim)
    _jac  = env.gym.acquire_jacobian_tensor(env.sim, env.cfg.asset.name)

   # -------- 2. 创建异步保存线程池 --------
    out_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "act_dataset")
    os.makedirs(out_dir, exist_ok=True)

    # 创建线程池用于异步保存（最多4个并发保存任务）
    save_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="EpisodeSaver")
    save_futures = []  # 跟踪正在进行的保存任务

    # 每个并行 env 对应一个缓冲区 & 计数器
    epi_buf = [dict(upper_actions=[], cmd_vel_and_height=[], qpos=[], eef_to_goal=[], qvel=[], obs_vel_and_height=[], point_cloud=[]) for _ in range(B)]
    global_epi = 0  # 全局 episode id（成功收集的）
    total_reset_count = 0  # 总重置次数（包括成功和失败）
    max_num_episodes = 1000  # 最大收集episode数
    total_start_time = time.time()  # 记录总开始时间

    print(f"[ACT-dataset] Async saving enabled with ThreadPoolExecutor (max_workers=4)")
    print(f"[ACT-dataset] Target: collect {max_num_episodes} episodes")

    # indices helpers
    rb_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
    wrist_idx = rb_names.index("right_wrist_yaw_link")

    obs = env.get_observations()
    env.reset_idx(torch.arange(B, device=device))

    # acquire tensors once
    _rb = env.gym.acquire_rigid_body_state_tensor(env.sim)

    nav_cfg = NavigatorCfg(
        resample_N=60,
        rot_deg=tuple(range(-90, 90, 15)),   # 先用 18 个角度，算力轻
        scales=(1.0,),                       # 先只用 1 个尺度
        lookahead=0.5, v_max=1.0, w_gain=3.0, w_max=1.0,
        rs_chunk=8, k_chunk=128
    )
    navigator = PathLibNavigator(PATHS_PLY, device=env.device, cfg=nav_cfg)
    # esdf_query_fn=make_esdf_query_fn(manager=env.esdf_manager, device=env.device)
    esdf_query_fn=make_esdf_query_fn_torch(manager=env.esdf_manager, device=env.device)
    # env.visualize_esdf()
    #super_paremeter


    try:
        for _ in range(30 * int(env.max_episode_length)):
            start_time = time.time()
            env.gym.refresh_dof_state_tensor(env.sim)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
            env.gym.refresh_jacobian_tensors(env.sim)

            dof_tensor = gymtorch.wrap_tensor(_dof).view(B, -1, 2)         # (B, D, 2)
            rb_tensor  = gymtorch.wrap_tensor(_rb ).view(B, -1, 13)      # (B, L, 13)
            jacobian   = gymtorch.wrap_tensor(_jac).view(B, env.num_bodies, 6, -1)  # (B, L, 6, D)

            # collision_free_nav
            current_pos_xy = env.root_states[:, :2]
            current_height = env.root_states[:, 2]
            qx, qy, qz, qw = env.root_states[:, 3:7].unbind(-1)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw_current = torch.atan2(siny_cosp, cosy_cosp)
            
            nav_out = navigator.select_and_track(
                base_pos_xy=current_pos_xy,
                base_yaw=yaw_current,
                target_pos_xy=safe_target_pos[:, :2],
                esdf_query_fn=esdf_query_fn
            )

            vx_cmd = nav_out['vx']  # (B,)
            yaw_cmd = nav_out['wz']  # (B,)
            height_cmd = torch.full((B,), 0.75, device=device)


            dist_to_goal = torch.linalg.norm(safe_target_pos[:, :2] - current_pos_xy, dim=1)

            # 找到到达目标位置的环境
            arrived_mask = dist_to_goal < arrive_tollerance
            arrived_ids = torch.nonzero(arrived_mask).squeeze(1)
            env.commands[:, 0] = vx_cmd      # vx
            env.commands[:, 1] = torch.zeros(B, dtype=torch.float32) # vy all zeros
            env.commands[:, 2] = yaw_cmd      # wz
            env.commands[:, 4] = height_cmd         # height

            if arrived_ids.numel() > 0:
                # 只对到达的环境设置命令
                vx_cmd[arrived_ids] = 0.0
                yaw_cmd[arrived_ids] = 0.0
                height_cmd[arrived_ids] = safe_target_pos[arrived_ids, 2].clamp(0.2, 0.8)

                env.commands[arrived_ids, 0] = vx_cmd[arrived_ids]  # vx = 0
                env.commands[arrived_ids, 2] = yaw_cmd[arrived_ids]  # wz = 0
                env.commands[arrived_ids, 4] = height_cmd[arrived_ids]  # height
                # 检查哪些环境的高度也到达了
                current_height = env.root_states[:, 2]  # 当前高度
                height_diff = torch.abs(current_height[arrived_ids] - env.commands[arrived_ids, 4])
                height_arrived_mask = height_diff < 0.08

                # 只对同时满足位置和高度条件的环境置为需要重置
                final_arrived_ids = arrived_ids[height_arrived_mask]
                if final_arrived_ids.numel() > 0:
                    need_reset[final_arrived_ids] = True


            actions = policy(obs.detach())
            left_arm_joint = left_arm_joint.view(B, -1)
            right_arm_joint = right_arm_joint.view(B, -1)
            waist_yaw_joint = waist_yaw_joint.view(B, -1)

            actions = torch.cat([actions, waist_yaw_joint, left_arm_joint, right_arm_joint], dim=1)
            upper_actions = torch.cat([waist_yaw_joint, left_arm_joint, right_arm_joint], dim=1)

            env.gym.refresh_rigid_body_state_tensor(env.sim)
            rb_tensor = gymtorch.wrap_tensor(_rb).view(B, -1, 13)
            s_vx = env.root_states[:, 7]  # (B,)
            s_wz = env.root_states[:, 12] # (B,)
            s_height = env.root_states[:, 2]   # (B,)

            wrist_pos = rb_tensor[0, wrist_idx, 0:3].cpu().numpy()
            goal_pos = safe_target_pos[0].cpu().numpy()
            goal=goal_pos-wrist_pos

            # 获取点云数据（如果环境支持）
            point_clouds = None
            if env.downsampled_cloud is not None:
                # 获取激光雷达点云数据
                # point_clouds = env.lidar_tensor.view(B,-1,3)  # 假设返回 (B, N, 3) 格式的点云
                point_clouds= env.downsampled_cloud  # (B, M, 3)，下采样后的点云
                # print(f"down pc shape{point_clouds.shape}")

            # env.gym.clear_lines(env.viewer)
            # env._draw_lidar_vis()

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
                goal_pos = safe_target_pos[i].cpu().numpy()
                world_goal_vector=goal_pos - wrist_pos
                base_goal_vector=world2base(world_goal_vector,yaw_current[i].item())
                epi_buf[i]["eef_to_goal"].append(base_goal_vector)

                # 添加点云数据
                if point_clouds is not None:
                    pc_data = point_clouds[i].detach().cpu().numpy()
                    epi_buf[i]["point_cloud"].append(pc_data)
                else:
                    # 如果没有点云数据，添加空数组占位
                    epi_buf[i]["point_cloud"].append(np.array([]))
            t0=time.perf_counter()
            obs, reward, _, reset_buf, *_ = env.step(actions.detach())  # reset中也会调用
            step_time_ms = (time.perf_counter() - t0) * 1000

            # 在同一行更新，带颜色高亮
            total_elapsed_min = (time.time() - total_start_time) / 60
            success_rate = (global_epi / total_reset_count * 100) if total_reset_count > 0 else 0.0
            print(f"\r\033[1;36m📊 Episodes: {global_epi:4d}/{max_num_episodes}\033[0m | \033[1;33m✓ Success: {success_rate:5.1f}%\033[0m | \033[1;35m⏱ Time: {total_elapsed_min:6.1f}min\033[0m | \033[1;32m⚡ Step: {step_time_ms:6.2f}ms\033[0m", end='', flush=True)

            # 检查是否达到最大episode数
            if global_epi >= max_num_episodes:
                print(f"\n[ACT-dataset] Reached target of {max_num_episodes} episodes. Exiting...")
                break
            reset_ids  = torch.nonzero(need_reset).squeeze(1)
            reset_buf = torch.nonzero(reset_buf).squeeze(1)
            # Print number of steps in env 0's current episode buffer
            env0_steps = len(epi_buf[0]["upper_actions"])
            # print(f"Env 0 current episode steps: {env0_steps}")
            if reset_buf.numel()>0:
                # 统计失败/超时的重置次数
                total_reset_count += reset_buf.numel()

                # 清空失败/超时环境的buffer（不保存）
                for idx in reset_buf.cpu().tolist():
                    epi_buf[idx] = dict(upper_actions=[], cmd_vel_and_height=[], qpos=[],
                                       eef_to_goal=[], qvel=[], obs_vel_and_height=[], point_cloud=[])

                env.reset_idx(reset_buf.cuda())        # 物理重置
                right_arm_joint[reset_buf,:] = torch.zeros(7, dtype=torch.float32,device=device) #reset arm
                height_cmd[reset_buf]     = 0.75 #reset height
                root = env.root_states          # shape = (B, 13)，B = 并行环境数
                pos_xyz = root[:, 0:3].clone().to(device)

                # 在机器人周围随机方向生成偏移（可以是前后左右任意方向）
                random_angles = torch.rand(B, device=env.device) * 2 * 3.14159  # 0到2π随机角度
                random_radius = pos_bias  
                bias_xy = torch.stack([
                    random_radius * torch.cos(random_angles),  # x偏移
                    random_radius * torch.sin(random_angles),  # y偏移
                    torch.zeros(B, device=env.device)           # z不偏移
                ], dim=1)

                new_safe_target_pos = Target_generate(pos_xyz+bias_xy,tar_range=tar_range,z_range=z_range,manager=env.esdf_manager,device=env.device,reset_indices=reset_buf)
                safe_target_pos[reset_buf]=new_safe_target_pos

                # env.gym.clear_lines(viewer)
                # for i in range(B):
                #     draw_target_cross(env, viewer, safe_target_pos[i,:])

            if reset_ids.numel():
                # 统计成功完成的重置次数
                num_success_resets = 0

                for idx in reset_ids.cpu().tolist():
                    # 异步保存 episode 数据
                    if 0 < len(epi_buf[idx]['upper_actions']) :               # 防止空 episode
                        # 深拷贝buffer数据（避免后续修改影响保存）
                        buffer_to_save = {
                            'upper_actions': epi_buf[idx]['upper_actions'][:],
                            'cmd_vel_and_height': epi_buf[idx]['cmd_vel_and_height'][:],
                            'qpos': epi_buf[idx]['qpos'][:],
                            'qvel': epi_buf[idx]['qvel'][:],
                            'eef_to_goal': epi_buf[idx]['eef_to_goal'][:],
                            'obs_vel_and_height': epi_buf[idx]['obs_vel_and_height'][:],
                            'point_cloud': epi_buf[idx]['point_cloud'][:]
                        }

                        # 提交到线程池异步保存
                        future = save_executor.submit(_save_episode, buffer_to_save, global_epi, out_dir)
                        save_futures.append(future)
                        print(f"[ACT-dataset] Submitted episode {global_epi} with {len(buffer_to_save['upper_actions'])} steps for async saving")
                        global_epi += 1
                        num_success_resets += 1

                        # 清空当前环境的buffer
                        epi_buf[idx] = dict(upper_actions=[], cmd_vel_and_height=[], qpos=[], eef_to_goal=[], qvel=[], obs_vel_and_height=[], point_cloud=[])

                # 更新总重置次数（成功的）
                total_reset_count += num_success_resets

                env.reset_idx(reset_ids.cuda())        # 物理重置
                height_cmd[reset_ids]     = 0.75 #reset height
                need_reset[reset_ids] = False    # 重置后不再满足“到达”条件 


                root = env.root_states
                pos_xyz = root[:, 0:3].clone().to(device)

                            # 在机器人周围随机方向生成偏移（可以是前后左右任意方向）
                random_angles = torch.rand(B, device=env.device) * 2 * 3.14159  # 0到2π随机角度
                random_radius = pos_bias  
                bias_xy = torch.stack([
                    random_radius * torch.cos(random_angles),  # x偏移
                    random_radius * torch.sin(random_angles),  # y偏移
                    torch.zeros(B, device=env.device)           # z不偏移
                ], dim=1)
                new_safe_target_pos = Target_generate(pos_xyz+bias_xy,tar_range=tar_range,z_range=z_range,manager=env.esdf_manager,device=env.device,reset_indices=reset_ids)
                safe_target_pos[reset_ids]=new_safe_target_pos

                # env.gym.clear_lines(viewer)
                # for i in range(B):
                #     draw_target_cross(env, viewer, safe_target_pos[i,:])
            for i in range(B):
                draw_target_cross(env, viewer, safe_target_pos[i,:])

    finally:
        # 等待所有保存任务完成
        print(f"\n[ACT-dataset] Waiting for {len(save_futures)} pending save tasks to complete...")
        for i, future in enumerate(save_futures):
            try:
                future.result(timeout=300)  # 每个任务最多等5分钟
            except Exception as e:
                print(f"[ERROR] Save task {i} failed: {e}")

        # 关闭线程池
        save_executor.shutdown(wait=True)
        final_success_rate = (global_epi / total_reset_count * 100) if total_reset_count > 0 else 0.0
        print(f"[ACT-dataset] All episodes saved. Total: {global_epi}")
        print(f"[ACT-dataset] Success rate: {global_epi}/{total_reset_count} = {final_success_rate:.2f}%")
        print(f"=========================")

if __name__ == "__main__":
    args = get_args()
    play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.75)
'''
conda activate aloha

CUDA_VISIBLE_DEVICE=1 python legged_gym/legged_gym/scripts/play_and_record_with_mid360.py  --num_envs 1 --sim_device cuda:0 --rl_device cuda:0
'''
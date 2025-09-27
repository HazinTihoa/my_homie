#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python version of path_generator.m
A) Generate motion-primitive paths with layered yaw splines; export to PLY
B) Build sector-shaped voxel points; compute voxel->path_id correspondences
"""
# reference :https://github.com/HongbiaoZ/autonomous_exploration_development_environment
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial import cKDTree
# import math
import matplotlib.pyplot as plt
# ------------- 工具函数：写 ASCII PLY ------------------
def write_ply_points_with_fields(filename, pts, field_names):
    """
    pts: shape (N, K) float/int
    field_names: list[str], length K, e.g. ["x","y","z","path_id","group_id"]
    """
    N, K = pts.shape
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        for name in field_names:
            # 这里简单判断：整数写 int，浮点写 float
            # 你也可以更严格根据数据类型分别写
            if name.endswith("_id") or name.endswith("id"):
                f.write(f"property int {name}\n")
            elif name in ("group_id", "path_id"):
                f.write(f"property int {name}\n")
            else:
                f.write(f"property float {name}\n")
        f.write("end_header\n")
        np.savetxt(f, pts, fmt=" ".join(["%f"]*(K-1)+["%d"])) if \
            (pts[:, -1].dtype.kind in "iu") else np.savetxt(f, pts, fmt=" ".join(["%f"]*K))

# ------------- A. 生成轨迹库 ---------------------------
def generate_paths(
    dis=1.0,
    angle_deg=27.0,
    delta_angle_deg=None,  # 默认 angle/3
    scale=0.65,
    dr=0.01,
    plot=False,
):
    """
    生成分层偏航样条的轨迹库：
    - 第1层 shift1 in [-angle, +angle]，步进 deltaAngle
    - 第2层 shift2 in shift1 ± angle*scale，步进 deltaAngle*scale
    - 第3层 shift3 in shift2 ± angle*scale^2，步进 deltaAngle*scale^2
    每条轨迹的角度作为弧长 r 的函数，用样条拟合：
      theta(r) = spline(r_nodes, shift_nodes)
      x(r) = r * cos(theta), y(r) = r * sin(theta)
    返回：
      start_paths: (Ns, 4) => [x,y,z,group_id]
      all_paths  : (Np, 5) => [x,y,z,path_id,group_id]
      path_list  : (M, 5) => [end_x,end_y,end_z,path_id,group_id]
    """

    if delta_angle_deg is None:
        delta_angle_deg = angle_deg / 3.0

    # 容器
    start_paths = []  # 所有起始段点 [x,y,z,group_id]
    all_paths = []    # 所有路径点 [x,y,z,path_id,group_id]
    path_list = []    # 每条路径末端 [end_x,end_y,0,path_id,group_id]

    path_id = 0
    group_id = 0

    # 可选可视化
    if plot:

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('auto')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Generated Paths')

    # 第1层：起始段
    shift1_vals = np.arange(-angle_deg, angle_deg + 1e-9, delta_angle_deg)
    for shift1 in shift1_vals:
        # 起步样条：两个节点 (0, 0), (dis, shift1)
        r_start = np.arange(0.0, dis + 1e-9, dr)
        # 两点的样条在 scipy 中可用一次/线性插值代替；这里用简单线性插值足够
        shift_start = np.interp(r_start, [0.0, dis], [0.0, shift1])

        # 起始段 (极坐标 -> 笛卡尔)
        theta_start = np.deg2rad(shift_start)
        x_start = r_start * np.cos(theta_start)
        y_start = r_start * np.sin(theta_start)
        z_start = np.zeros_like(x_start)

        # 记录起始段
        gcol = np.full_like(x_start, group_id, dtype=np.int32)
        start_paths.append(np.stack([x_start, y_start, z_start, gcol], axis=1))

        # 第2层/第3层：继续叠加样条控制点
        shift2_vals = np.arange(shift1 - angle_deg*scale, shift1 + angle_deg*scale + 1e-9, delta_angle_deg*scale)
        for shift2 in shift2_vals:
            shift3_vals = np.arange(shift2 - angle_deg*(scale**2), shift2 + angle_deg*(scale**2) + 1e-9, delta_angle_deg*(scale**2))
            for shift3 in shift3_vals:
                # 节点构造：把起始段的采样 (r_start, shift_start) 当做“密集节点”，再追加 3 个新节点
                r_nodes = np.concatenate([r_start, [2*dis, 3*dis - 1e-3, 3*dis]])
                s_nodes = np.concatenate([shift_start, [shift2, shift3, shift3]])

                # 用三次样条“严格通过”节点（s=0）
                spline = UnivariateSpline(r_nodes, s_nodes, s=0, k=3)

                r_full = np.arange(0.0, r_nodes[-1] + 1e-9, dr)
                s_full = spline(r_full)  # (deg)

                theta = np.deg2rad(s_full)
                x = r_full * np.cos(theta)
                y = r_full * np.sin(theta)
                z = np.zeros_like(x)

                pid_col = np.full_like(x, path_id, dtype=np.int32)
                gid_col = np.full_like(x, group_id, dtype=np.int32)
                pts = np.stack([x, y, z, pid_col, gid_col], axis=1)
                all_paths.append(pts)

                # 末端记录
                path_list.append(np.array([x[-1], y[-1], 0.0, path_id, group_id], dtype=np.float32))
                if plot:
  
                    ax.plot3D(x, y, z, linewidth=0.8)

                path_id += 1

        group_id += 1

    if plot:

        ax.view_init(elev=25, azim=-60)
        plt.show()

    # 拼接
    start_paths = np.concatenate(start_paths, axis=0) if len(start_paths) else np.zeros((0,4))
    all_paths   = np.concatenate(all_paths, axis=0)   if len(all_paths)   else np.zeros((0,5))
    path_list   = np.stack(path_list, axis=0)         if len(path_list)   else np.zeros((0,5))

    # 类型修正（整数列）
    if all_paths.shape[1] == 5:
        all_paths[:, 3] = np.round(all_paths[:, 3]).astype(np.int32)
        all_paths[:, 4] = np.round(all_paths[:, 4]).astype(np.int32)
    if path_list.shape[1] == 5:
        path_list[:, 3] = np.round(path_list[:, 3]).astype(np.int32)
        path_list[:, 4] = np.round(path_list[:, 4]).astype(np.int32)
    if start_paths.shape[1] == 4:
        start_paths[:, 3] = np.round(start_paths[:, 3]).astype(np.int32)

    return start_paths, all_paths, path_list

# ------------- B. 扇形体素与对应关系 -------------------
def build_sector_voxels(offsetX=3.2, offsetY=4.5, voxelSize=0.02, searchRadius=0.45,
                        voxelNumX=161, voxelNumY=451):
    """
    构造与 MATLAB 脚本等价的“扇形/楔形”采样点：
      for indX=0..voxelNumX-1:
        x = offsetX - voxelSize*indX
        scaleY = x/offsetX + (searchRadius/offsetY) * (offsetX - x)/offsetX
        for indY=0..voxelNumY-1:
          y = scaleY * (offsetY - voxelSize*indY)
    返回 voxel_points: (N, 2)
    """
    pts = np.zeros((voxelNumX*voxelNumY, 2), dtype=np.float32)
    idx = 0
    for indX in range(voxelNumX):
        x = offsetX - voxelSize * indX
        scaleY = x/offsetX + (searchRadius/offsetY) * (offsetX - x)/offsetX
        for indY in range(voxelNumY):
            y = scaleY * (offsetY - voxelSize * indY)
            pts[idx, 0] = x
            pts[idx, 1] = y
            idx += 1
    return pts

def write_correspondences_txt(filename, voxel_points, path_all_xy, path_ids, searchRadius):
    """
    对每个 voxel 点，找 searchRadius 内的 path 采样点（用 KDTree），
    将 path_id 去重排序后写为：
      voxel_idx path_id_1 path_id_2 ... -1
    """
    tree = cKDTree(path_all_xy)  # (N_path_pts, 2)
    # 批量查询（返回 list of lists）
    idx_lists = tree.query_ball_point(voxel_points, r=searchRadius)

    with open(filename, "w") as f:
        for i, indices in enumerate(idx_lists):
            f.write(f"{i} ")
            if len(indices) > 0:
                pids = np.unique(path_ids[indices])  # 去重+排序
                for pid in pids:
                    f.write(f"{int(pid)} ")
            f.write("-1\n")

# ------------- main -------------------
if __name__ == "__main__":
    # A) 生成轨迹
    print("\nGenerating paths ...")
    start_paths, all_paths, path_list = generate_paths(
        dis=1.0, angle_deg=27.0, delta_angle_deg=None, scale=0.65, dr=0.01, plot=True   
    )
    print(f"start_paths: {start_paths.shape}, all_paths: {all_paths.shape}, path_list: {path_list.shape}")

    # 写 PLY（与 MATLAB 字段一致）
    write_ply_points_with_fields("startPaths.ply", start_paths, ["x","y","z","group_id"])
    write_ply_points_with_fields("paths.ply",      all_paths,   ["x","y","z","path_id","group_id"])
    write_ply_points_with_fields("pathList.ply",   path_list,   ["end_x","end_y","end_z","path_id","group_id"])
    print("PLY files written: startPaths.ply, paths.ply, pathList.ply")

    # B) 体素 + 对应关系
    print("\nPreparing sector voxels ...")
    voxel_points = build_sector_voxels(
        offsetX=3.2, offsetY=4.5, voxelSize=0.02, searchRadius=0.45, voxelNumX=161, voxelNumY=451
    )
    print(f"voxel_points: {voxel_points.shape} points")

    print("\nCollision / neighbor search ...")
    path_xy = all_paths[:, :2]
    path_ids = all_paths[:, 3].astype(np.int32)
    write_correspondences_txt("correspondences.txt", voxel_points, path_xy, path_ids, searchRadius=0.45)
    print("Correspondences written: correspondences.txt")

    print("\nDone.")

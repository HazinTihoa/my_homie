# -*- coding: utf-8 -*-
"""
PathLibNavigator
================
基于离线路径库（paths.ply）的并行选轨与 Pure-Pursuit 跟踪：
- 加载并重采样路径库到固定长度 N
- 候选 = 路径库 × 旋转 × 缩放（再叠加 base_yaw & 平移到世界）
- 使用 ESDF batched 查询打分：score = α*clearance + β*advance + γ*align
- 选择最优轨迹并给出 (vx, wz) 控制指令

注意：
- ESDF 查询通过回调 esdf_query_fn(points_world[B,M,3])->dist[B,M] 注入
- 若没有 ESDF，可先用 dummy_esdf_query 占位（恒正距）
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

# ============================================================
# Utils
# ============================================================

def _wrap_to_pi(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2 * math.pi) - math.pi

def _resample_polyline_xy(xy: np.ndarray, N: int) -> np.ndarray:
    """将二维 polyline 等距重采样到 N 个点。xy: [M,2]."""
    if xy.shape[0] == 0:
        return np.zeros((N, 2), dtype=np.float32)
    if xy.shape[0] == 1:
        return np.repeat(xy.astype(np.float32), N, axis=0)
    seg = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total < 1e-6:
        return np.repeat(xy[:1].astype(np.float32), N, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.linspace(0.0, total, N)
    out = np.zeros((N, 2), dtype=np.float32)
    j = 0

    for i, si in enumerate(s):
        while j + 1 < len(cum) and cum[j + 1] < si:
            j += 1
        
        # 边界保护：确保不会访问超出边界的索引
        if j + 1 >= len(cum):
            j = len(cum) - 2  # 确保 j+1 不超出边界
        
        t = 0.0 if cum[j + 1] == cum[j] else (si - cum[j]) / (cum[j + 1] - cum[j])
        
        # 边界保护：确保不会访问超出边界的xy索引
        j_safe = min(j, len(xy) - 1)
        j_next_safe = min(j + 1, len(xy) - 1)
        
        out[i] = xy[j_safe] * (1 - t) + xy[j_next_safe] * t
    return out

# ============================================================
# Minimal PLY ASCII Reader (x y z path_id group_id)
# ============================================================

def _load_ply_ascii_numeric(path: str) -> np.ndarray:
    """
    尝试作为 ASCII PLY 读取，返回 [N, C] 数值数组；
    如为 binary 或格式不符，请替换成你自己的 PLY 读取器。
    """
    with open(path, "r") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            header.append(line.strip())
            if line.startswith("format ") and "ascii" not in line:
                raise ValueError("PLY is not ASCII; please use your own loader.")
            if line.strip() == "end_header":
                break
        data = []
        for line in f:
            ls = line.strip()
            if not ls:
                continue
            toks = ls.split()
            try:
                row = [float(t) for t in toks]
            except ValueError:
                continue
            data.append(row)
    arr = np.array(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"PLY parsed to shape {arr.shape}, expected 2D")
    return arr

# ============================================================
# Path Library
# ============================================================

@dataclass
class PathLibrary:
    paths_xy: torch.Tensor   # [K, N, 2]
    end_dir: torch.Tensor    # [K] 末端朝向（弧度，局部帧，x轴为0）
    path_ids: torch.Tensor   # [K]
    group_ids: torch.Tensor  # [K]

    @staticmethod
    def from_ply(paths_ply: str, N: int = 60, device: torch.device = torch.device("cpu")) -> "PathLibrary":
        """
        从 paths.ply 组装路径库；要求列含 [x,y,(z), path_id, group_id]
        若 PLY 为 binary 或列名不同，请改成你已有的加载函数并返回以下四个字段。
        """
        arr = _load_ply_ascii_numeric(paths_ply)
        if arr.shape[1] < 5:
            raise ValueError("paths.ply needs >=5 columns: x y z path_id group_id")
        x, y = arr[:, 0], arr[:, 1]
        pid = arr[:, -2].astype(np.int64)
        gid = arr[:, -1].astype(np.int64)

        order = np.lexsort((np.arange(len(pid)), pid))
        x, y, pid, gid = x[order], y[order], pid[order], gid[order]
        uniq = np.unique(pid)

        paths_xy_list, end_dir_list, path_ids, group_ids = [], [], [], []
        for p in uniq:
            idx = np.where(pid == p)[0]
            xy = np.stack([x[idx], y[idx]], axis=-1)
            xy_rs = _resample_polyline_xy(xy, N)
            v = xy_rs[-1] - xy_rs[-2] if N >= 2 else np.array([1.0, 0.0], dtype=np.float32)
            ang = float(math.atan2(v[1], v[0]))
            paths_xy_list.append(xy_rs)
            end_dir_list.append(ang)
            path_ids.append(int(p))
            group_ids.append(int(gid[idx[0]]))

        paths_xy = torch.from_numpy(np.stack(paths_xy_list, 0)).to(device)
        end_dir = torch.tensor(end_dir_list, dtype=torch.float32, device=device)
        path_ids_t = torch.tensor(path_ids, dtype=torch.int64, device=device)
        group_ids_t = torch.tensor(group_ids, dtype=torch.int64, device=device)
        return PathLibrary(paths_xy=paths_xy, end_dir=end_dir, path_ids=path_ids_t, group_ids=group_ids_t)

# ============================================================
# Navigator
# ============================================================

@dataclass
class NavigatorCfg:
    resample_N: int = 60
    rot_deg: Tuple[int, ...] = tuple(range(-180, 180, 10))  # 候选旋转（度）
    scales: Tuple[float, ...] = (0.75, 1.0, 1.25)           # 候选缩放
    alpha: float = 1.0  # clearance 权重
    beta: float = 1.0   # advance   权重
    gamma: float = 0.5  # align     权重
    lookahead: float = 0.5
    v_max: float = 1.0
    w_gain: float = 3.0
    w_max: float = 1.0
    rs_chunk: int = 12  # R*S 分块
    k_chunk: int = 128  # K 分块

class PathLibNavigator:
    def __init__(self, paths_ply: str, device: torch.device, cfg: NavigatorCfg = NavigatorCfg()):
        self.device = device
        self.cfg = cfg
        self.lib = PathLibrary.from_ply(paths_ply, N=cfg.resample_N, device=device)
        # 预计算旋转
        rot_rad = torch.tensor([math.radians(d) for d in cfg.rot_deg], dtype=torch.float32, device=device)  # [R]
        self.rot_sin = torch.sin(rot_rad)  # [R]
        self.rot_cos = torch.cos(rot_rad)  # [R]
        self.rot_rad = rot_rad
        self.scales = torch.tensor(cfg.scales, dtype=torch.float32, device=device)  # [S]

    @torch.no_grad()
    def select_and_track(
        self,
        base_pos_xy: torch.Tensor,   # [B,2]
        base_yaw: torch.Tensor,      # [B]
        target_pos_xy: torch.Tensor, # [B,2]
        esdf_query_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        返回：vx, wz，以及用于记录的元数据（path_idx/id、rot_idx、scale_idx、score 等）
        esdf_query_fn: (B,M,3)->(B,M)；若 None，则 clearance 用常数。
        """
        import time
        
        # 性能测试 - 开始计时
        t_start = time.perf_counter()
        device = self.device
        B = base_pos_xy.shape[0]
        K, N, _ = self.lib.paths_xy.shape
        R = self.rot_cos.numel()
        S = self.scales.numel()

        # 目标方向
        vec_bt = (target_pos_xy - base_pos_xy)                   # [B,2]
        dist_bt = torch.linalg.norm(vec_bt, dim=-1).clamp(min=1e-6)
        dir_bt = vec_bt / dist_bt.unsqueeze(-1)                  # [B,2]
        goal_yaw = torch.atan2(dir_bt[..., 1], dir_bt[..., 0])   # [B]
        
        # 时间测试点 1 - 初始化完成
        t1 = time.perf_counter()

        # 初始化 best
        best_score = torch.full((B,), -1e9, device=device)
        best_meta = {
            'rot_idx': torch.zeros(B, dtype=torch.int64, device=device),
            'scale_idx': torch.zeros(B, dtype=torch.int64, device=device),
            'path_idx': torch.zeros(B, dtype=torch.int64, device=device),
            'clearance': torch.zeros(B, dtype=torch.float32, device=device),
            'advance': torch.zeros(B, dtype=torch.float32, device=device),
            'align': torch.zeros(B, dtype=torch.float32, device=device),
        }

        paths_xy = self.lib.paths_xy     # [K,N,2]
        end_dir = self.lib.end_dir       # [K]

        rs_total = R * S
        rs_per_chunk = self.cfg.rs_chunk
        k_per_chunk = self.cfg.k_chunk
        
        # 时间测试点 2 - 开始主循环
        t2 = time.perf_counter()
        esdf_total_time = 0.0
        path_eval_time = 0.0
        loop_count = 0

        for rs_beg in range(0, rs_total, rs_per_chunk):
            rs_end = min(rs_beg + rs_per_chunk, rs_total)
            idx = torch.arange(rs_beg, rs_end, device=device)
            ri = torch.div(idx, S, rounding_mode='floor')  # [rs]
            si = idx % S                                   # [rs]
            cos = self.rot_cos[ri].view(-1, 1, 1)          # [rs,1,1]
            sin = self.rot_sin[ri].view(-1, 1, 1)
            scl = self.scales[si].view(-1, 1, 1)

            px = paths_xy[..., 0].unsqueeze(0)  # [1,K,N]
            py = paths_xy[..., 1].unsqueeze(0)
            rx = (cos * px - sin * py) * scl    # [rs,K,N]
            ry = (sin * px + cos * py) * scl

            end_dir_rs = (end_dir.unsqueeze(0) + self.rot_rad[ri].unsqueeze(-1))  # [rs,K]
            end_dir_rs = _wrap_to_pi(end_dir_rs)

            for k_beg in range(0, K, k_per_chunk):
                k_end = min(k_beg + k_per_chunk, K)
                rx_k = rx[:, k_beg:k_end]   # [rs,kc,N]
                ry_k = ry[:, k_beg:k_end]
                ed_k = end_dir_rs[:, k_beg:k_end]  # [rs,kc]

                c_by = torch.cos(base_yaw).view(B, 1, 1, 1)
                s_by = torch.sin(base_yaw).view(B, 1, 1, 1)
                rx4 = rx_k.unsqueeze(0)  # [1,rs,kc,N]
                ry4 = ry_k.unsqueeze(0)
                wx = c_by * rx4 - s_by * ry4
                wy = s_by * rx4 + c_by * ry4
                wx = wx + base_pos_xy[:, None, None, None, 0]
                wy = wy + base_pos_xy[:, None, None, None, 1]
                # [B,rs,kc,N] world path

                end_yaw_world = _wrap_to_pi(ed_k.unsqueeze(0) + base_yaw.view(B, 1, 1))  # [B,rs,kc]

                # (1) clearance
                # 时间测试 - ESDF查询开始
                esdf_start = time.perf_counter()
                
                if esdf_query_fn is None:
                    clearance = torch.full((B, rx_k.shape[0], rx_k.shape[1]), 1e3, device=device)
                else:
                    rs, kc = rx_k.shape[0], rx_k.shape[1]
                    M_rs = rs * kc
                #     min_clear = torch.full((B, M_rs), 1e6, device=device)
                #     stepN = 32
                #     for n_beg in range(0, N, stepN):
                #         n_end = min(n_beg + stepN, N)
                #         n_chunk = n_end - n_beg
                #         xx = wx[..., n_beg:n_end]  # [B,rs,kc,n_chunk]
                #         yy = wy[..., n_beg:n_end]
                #         pts = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)  # [B,rs,kc,n_chunk,3]
                #         pts = pts.view(B, M_rs, n_chunk, 3)  # [B, M_rs, n_chunk, 3]
                        
                #         # 分批处理每个路径-尺度组合
                #         for m in range(M_rs):
                #             pts_m = pts[:, m, :, :]  # [B, n_chunk, 3]
                #             dist_m = esdf_query_fn(pts_m)  # [B, n_chunk]
                #             min_dist_m = torch.min(dist_m, dim=1)[0]  # [B]
                #             min_clear[:, m] = torch.minimum(min_clear[:, m], min_dist_m)
                    
                #     clearance = min_clear.view(B, rs, kc)  # [B,rs,kc]
                    min_clear = torch.full((B, M_rs), 1e6, device=device)
                    # 建议一次取完，或把 stepN 调大
                    stepN = N
                    for n_beg in range(0, N, stepN):
                        n_end = min(n_beg + stepN, N)
                        xx = wx[..., n_beg:n_end]  # [B,rs,kc,nc]
                        yy = wy[..., n_beg:n_end]
                        pts = torch.stack([xx, yy, torch.zeros_like(xx)], dim=-1)      # [B,rs,kc,nc,3]
                        pts = pts.view(B, M_rs * (n_end - n_beg), 3)                   # [B, M, 3]

                        # ✅ 一次性 ESDF 查询；返回 [B, M]
                        dist = esdf_query_fn(pts).view(B, M_rs, (n_end - n_beg))       # [B, M_rs, nc]

                        # 块内取最小，再和历史最小比较
                        min_clear = torch.minimum(min_clear, dist.min(dim=-1).values)  # [B, M_rs]

                    clearance = min_clear.view(B, rs, kc)
                
                # 时间测试 - ESDF查询结束
                esdf_end = time.perf_counter()
                esdf_total_time += (esdf_end - esdf_start)

                # (2) advance
                end_x = wx[..., -1]
                end_y = wy[..., -1]
                end_vec = torch.stack([
                    end_x - base_pos_xy[:, None, None, 0],
                    end_y - base_pos_xy[:, None, None, 1]
                ], dim=-1)  # [B,rs,kc,2]
                advance = (end_vec * dir_bt[:, None, None, :]).sum(-1)  # [B,rs,kc]

                # (3) align
                # 时间测试 - 路径评估开始
                path_eval_start = time.perf_counter()
                
                align = torch.cos(_wrap_to_pi(end_yaw_world - goal_yaw.view(B, 1, 1)))  # [B,rs,kc]

                score = self.cfg.alpha * clearance + self.cfg.beta * advance + self.cfg.gamma * align
                score_flat = score.view(B, -1)
                best_val, best_idx = torch.max(score_flat, dim=-1)  # [B]

                mask = best_val > best_score
                if mask.any():
                    best_score = torch.where(mask, best_val, best_score)
                    idx_rs = (best_idx // rx_k.shape[1]).to(torch.int64)
                    idx_kc = (best_idx %  rx_k.shape[1]).to(torch.int64)
                    ri_sel = ri[idx_rs]
                    si_sel = si[idx_rs]
                    k_sel  = torch.tensor(k_beg, device=device) + idx_kc

                    best_meta['rot_idx']   = torch.where(mask, ri_sel, best_meta['rot_idx'])
                    best_meta['scale_idx'] = torch.where(mask, si_sel, best_meta['scale_idx'])
                    best_meta['path_idx']  = torch.where(mask, k_sel,  best_meta['path_idx'])

                    gather_idx = idx_rs * rx_k.shape[1] + idx_kc
                    clear_flat = clearance.view(B, -1)
                    adv_flat   = advance.view(B, -1)
                    ali_flat   = align.view(B, -1)
                    best_meta['clearance'] = torch.where(mask, clear_flat.gather(1, gather_idx.view(-1,1)).squeeze(1), best_meta['clearance'])
                    best_meta['advance']   = torch.where(mask, adv_flat.gather(1, gather_idx.view(-1,1)).squeeze(1),   best_meta['advance'])
                    best_meta['align']     = torch.where(mask, ali_flat.gather(1, gather_idx.view(-1,1)).squeeze(1),   best_meta['align'])
                
                # 时间测试 - 路径评估结束
                path_eval_end = time.perf_counter()
                path_eval_time += (path_eval_end - path_eval_start)
                loop_count += 1

        # 时间测试点 3 - 主循环结束
        t3 = time.perf_counter()
        
        # -------- Pure-Pursuit 跟踪 --------
        ri_sel = best_meta['rot_idx']
        si_sel = best_meta['scale_idx']
        k_sel  = best_meta['path_idx']

        pxy = self.lib.paths_xy[k_sel]  # [B,N,2]
        rot_r = self.rot_rad[ri_sel]    # [B]
        c = torch.cos(rot_r).view(B, 1)
        s = torch.sin(rot_r).view(B, 1)
        scl = self.scales[si_sel].view(B, 1)
        px = pxy[..., 0]
        py = pxy[..., 1]
        rx = (c * px - s * py) * scl
        ry = (s * px + c * py) * scl

        c_by = torch.cos(base_yaw).view(B, 1)
        s_by = torch.sin(base_yaw).view(B, 1)
        wx = c_by * rx - s_by * ry + base_pos_xy[:, 0:1]
        wy = s_by * rx + c_by * ry + base_pos_xy[:, 1:2]

        dx = wx - base_pos_xy[:, 0:1]
        dy = wy - base_pos_xy[:, 1:2]
        dist = torch.sqrt(dx * dx + dy * dy)
        mask = dist >= self.cfg.lookahead
        idx_look = torch.where(mask.any(dim=1), mask.float().argmax(dim=1), dist.argmax(dim=1))
        gx = wx.gather(1, idx_look.view(-1, 1)).squeeze(1)
        gy = wy.gather(1, idx_look.view(-1, 1)).squeeze(1)

        vbx = torch.cos(base_yaw)
        vby = torch.sin(base_yaw)
        vlx = gx - base_pos_xy[:, 0]
        vly = gy - base_pos_xy[:, 1]
        yaw_err = _wrap_to_pi(torch.atan2(vly, vlx) - torch.atan2(vby, vbx))

        vx = (self.cfg.v_max * torch.clamp(torch.cos(yaw_err), min=0.0))
        wz = torch.clamp(self.cfg.w_gain * yaw_err, min=-self.cfg.w_max, max=self.cfg.w_max)
        
        # 时间测试点 4 - Pure-Pursuit完成
        t4 = time.perf_counter()
        
        # ========== 性能分析报告 ==========
        total_time = (t4 - t_start) * 1000  # 转换为毫秒
        init_time = (t1 - t_start) * 1000
        loop_setup_time = (t2 - t1) * 1000
        main_loop_time = (t3 - t2) * 1000
        pursuit_time = (t4 - t3) * 1000
        esdf_time = esdf_total_time * 1000
        path_eval_time_ms = path_eval_time * 1000
        
        # print(f"========== PathLibNavigator 性能分析 ==========")
        # print(f"总耗时: {total_time:.2f} ms")
        # print(f"  - 初始化: {init_time:.2f} ms ({init_time/total_time*100:.1f}%)")
        # print(f"  - 循环准备: {loop_setup_time:.2f} ms ({loop_setup_time/total_time*100:.1f}%)")
        # print(f"  - 主循环: {main_loop_time:.2f} ms ({main_loop_time/total_time*100:.1f}%)")
        # print(f"    - ESDF查询: {esdf_time:.2f} ms ({esdf_time/total_time*100:.1f}%)")
        # print(f"    - 路径评估: {path_eval_time_ms:.2f} ms ({path_eval_time_ms/total_time*100:.1f}%)")
        # print(f"    - 其他: {main_loop_time-esdf_time-path_eval_time_ms:.2f} ms")
        # print(f"  - Pure-Pursuit: {pursuit_time:.2f} ms ({pursuit_time/total_time*100:.1f}%)")
        # print(f"循环次数: {loop_count}, 平均每次循环: {main_loop_time/max(1,loop_count):.2f} ms")
        # print(f"=============================================")

        return {
            'vx': vx,
            'wz': wz,
            'selected_path_idx': k_sel,
            'selected_path_id': self.lib.path_ids[k_sel],
            'selected_group_id': self.lib.group_ids[k_sel],
            'rot_idx': ri_sel,
            'scale_idx': si_sel,
            'score': best_score,
            'clearance': best_meta['clearance'],
            'advance': best_meta['advance'],
            'align': best_meta['align'],
        }


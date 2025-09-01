# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
import threading
import time
import socket, struct, numpy as np
import time

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
from LidarSensor.example.isaacgym.utils.terrain.terrain import Terrain
from LidarSensor.example.isaacgym.utils.terrain.terrain_cfg import Terrain_cfg
import numpy as np
from pytorch3d.ops import sample_farthest_points
from isaacgym import gymutil
from legged_gym.utils.esdf_manager import ESDFManager

def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return roll_x.unsqueeze(1), pitch_y.unsqueeze(1), yaw_z.unsqueeze(1)
class PcBridgeClient:
    def __init__(self, host='127.0.0.1', port=5555, reconnect_interval=2.0):
        self.host = host
        self.port = port
        self.sock = None
        self.last_try = 0.0
        self.reconnect_interval = reconnect_interval

    def _connect(self):
        now = time.time()
        if self.sock is not None:
            return True
        if now - self.last_try < self.reconnect_interval:
            return False
        self.last_try = now
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((self.host, self.port))
            self.sock = s
            return True
        except Exception:
            self.sock = None
            return False
        
    def send_tf(self, parent_frame: str, child_frame: str,
                    xyz: np.ndarray, quat_xyzw: np.ndarray):
            """
            发送 TF: map->mid_360
            包格式:
            <uint32 total_len>
            <uint8  pkt_type=1>
            <uint8  parent_len><parent_bytes>
            <uint8  child_len><child_bytes>
            <7 * float32>  (x,y,z,qx,qy,qz,qw)
            """
            if self.sock is None and not self._connect():
                return
            parent_b = parent_frame.encode('utf-8')
            child_b  = child_frame.encode('utf-8')

            payload = struct.pack(
                "<B B{}s B{}s 7f".format(len(parent_b), len(child_b)),
                1,  # pkt_type = 1 表示 TF
                len(parent_b), parent_b,
                len(child_b),  child_b,
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]),
            )
            blob = struct.pack("<I", len(payload)) + payload
            try:
                self.sock.sendall(blob)
            except Exception:
                try:
                    if self.sock: self.sock.close()
                except Exception:
                    pass
                self.sock = None

    def send_points(self, points_xyz: np.ndarray, frame_id: str = "mid_360"):
        """
        pkt_type=0 的点云包:
        <uint32 total_len>
        <uint8  pkt_type=0>
        <uint32 N>
        <uint8  name_len><name_bytes>
        <N*3*float32>
        """
        if points_xyz is None or points_xyz.size == 0:
            return

        # dtype / 连续性
        pts = np.asarray(points_xyz, dtype=np.float32, order='C')
        N = int(pts.shape[0])

        name_bytes = frame_id.encode('utf-8')
        name_len = len(name_bytes)

        # payload: pkt_type(0) + header + raw points
        header = struct.pack("<B", 0) + struct.pack("<I", N) + struct.pack("<B", name_len) + name_bytes
        payload = header + pts.tobytes(order='C')

        # blob: total_len + payload
        blob = struct.pack("<I", len(payload)) + payload

        # 确保连接
        if self.sock is None and not self._connect():
            return

        try:
            self.sock.sendall(blob)
        except Exception:
            try:
                if self.sock:
                    self.sock.close()
            finally:
                self.sock = None


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        self.sensor_cfg = LidarConfig()
        self.sensor_cfg.sensor_type ="mid360" # mid360,horizon,HAP,mid70,mid40,tele,avia
        self.sensor_cfg.max_range = 15.0  # 增加扫描范围到15米
        self.sim_time = 0
        self.sensor_update_time = 0
        self.state_update_time = 0
        self.sensor_cfg.update_frequency = 50.0
        self.selected_env_idx = 0 
        wp.init()
        self.pc_bridge = PcBridgeClient(host='127.0.0.1', port=5555)
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.critic_history_length = self.cfg.env.num_critic_history
        self.actor_proprioceptive_obs_length = self.num_one_step_obs * self.actor_history_length
        self.critic_proprioceptive_obs_length = self.num_one_step_privileged_obs * self.critic_history_length
        self.actor_use_height = True if self.num_obs > self.actor_proprioceptive_obs_length else False
        self.num_lower_dof = self.cfg.env.num_actions
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.create_warp_env()
        self.create_warp_tensor()
        self.esdf_manager=ESDFManager(self.lidar_vertices,(self.terrain.cfg.border_size ,self.terrain.cfg.border_size))
        
        self.sensor = LidarSensor(self.warp_tensor_dict, None, self.sensor_cfg, 1, self.device)
        self.lidar_tensor, self.sensor_dist_tensor = self.sensor.update()

        # self._visualize_terrain_vertices()

    def visualize_esdf(self):
        """可视化ESDF，用颜色编码的球体表示距离障碍物的远近"""
        if not hasattr(self, 'esdf_manager') or self.esdf_manager is None:
            print("ESDF管理器未初始化")
            return
            
        if not hasattr(self, 'viewer') or self.viewer is None:
            print("无viewer可用，跳过ESDF可视化")
            return
            
        try:
            # 获取ESDF数据和格子中心坐标
            esdf_data = self.esdf_manager.esdf  # shape: (H, W)
            if esdf_data is None:
                print("ESDF数据未构建")
                return
                
            X, Y = self.esdf_manager.esdf_grid_centers()  # 格子中心的世界坐标
            
            # 下采样策略：限制最大显示点数，避免性能问题
            max_points = 4000  # 限制最大点数
            total_points = esdf_data.shape[0] * esdf_data.shape[1]

            downsample_factor = max(1, int(np.sqrt(total_points / max_points)))
            esdf_sampled = esdf_data[::downsample_factor, ::downsample_factor]
            X_sampled = X[::downsample_factor, ::downsample_factor]
            Y_sampled = Y[::downsample_factor, ::downsample_factor]
            
            # print(f"ESDF原始尺寸: {esdf_data.shape}, 下采样后: {esdf_sampled.shape}")
            # print(f"下采样因子: {downsample_factor}")
            
            # 扁平化数据以便处理
            esdf_flat = esdf_sampled.flatten()
            x_flat = X_sampled.flatten()
            y_flat = Y_sampled.flatten()
            
            # 过滤掉无效值和极端值
            valid_mask = (~np.isnan(esdf_flat)) & (~np.isinf(esdf_flat)) 
            esdf_valid = esdf_flat[valid_mask]
            x_valid = x_flat[valid_mask]
            y_valid = y_flat[valid_mask]
            
            # print(f"有效ESDF点数: {len(esdf_valid)}")
            
            if len(esdf_valid) == 0:
                print("没有有效的ESDF数据点")
                return

            # 定义高度（地面上方0.1米）
            z_height = 0.1
            
            # 为第一个环境绘制ESDF
            for i in range(len(esdf_valid)):
                distance = esdf_valid[i]
                x = x_valid[i]
                y = y_valid[i]
                
                # ESDF值处理：
                # 正值 = 距离障碍物的距离（自由空间）
                # 负值 = 在障碍物内部（用绝对值表示深度）
                if distance < 0:
                    # 在障碍物内部，用深红色表示
                    color = (0.8, 0.0, 0.0)  # 深红色
                else:
                    # 自由空间，根据距离用颜色梯度表示
                    color = self._distance_to_color(distance)
                
                # 创建球体几何体，大小根据距离调整
             
                sphere_geom = gymutil.WireframeSphereGeometry(
                    0.1, 6, 6, None, color=color
                )
                
                # 创建球体位置
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z_height), r=None)
                
                # 绘制球体
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
            
            # print(f"ESDF可视化完成，显示了 {len(esdf_valid)} 个点")
            # print("深红色=障碍物内部, 红色=危险(近距离), 黄色=中等距离, 绿色=安全(2m+)")
            
            # 绘制ESDF地图的四个顶点边界，用蓝色表示
            self._draw_esdf_corners(z_height)
            
        except Exception as e:
            print(f"ESDF可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _distance_to_color(self, distance):
        """将距离值转换为颜色（红色=危险，绿色=安全）"""
        # 距离范围：0-2m
        # 0m: 红色 (1, 0, 0)
        # 1m: 黄色 (1, 1, 0) 
        # 2m+: 绿色 (0, 1, 0)
        
        distance = max(0.0, distance)  # 确保距离非负
        
        if distance <= 1.0:
            # 0-1m: 红色到黄色过渡
            ratio = distance / 1.0
            return (1.0, ratio, 0.0)  # 红色(1,0,0) -> 黄色(1,1,0)
        elif distance <= 2.0:
            # 1-2m: 黄色到绿色过渡
            ratio = (distance - 1.0) / 1.0
            return (1.0 - ratio, 1.0, 0.0)  # 黄色(1,1,0) -> 绿色(0,1,0)
        else:
            # 2m+: 纯绿色（安全）
            return (0.0, 1.0, 0.0)
    
    def _draw_esdf_corners(self, z_height):
        """绘制ESDF地图的四个边界顶点，用蓝色表示"""
        try:
            # 获取ESDF地图的边界信息
            origin = self.esdf_manager.origin  # [min_x, min_y]
            map_width = self.esdf_manager.map_width
            map_height = self.esdf_manager.map_height
            resolution = self.esdf_manager.map_resolution
            
            # 计算地图边界的世界坐标
            min_x, min_y = origin[0], origin[1]
            max_x = min_x + map_width * resolution
            max_y = min_y + map_height * resolution
            
            # 四个边界顶点的坐标
            corners = [
                (min_x, min_y, z_height + 0.2),  # 左下角
                (max_x, min_y, z_height + 0.2),  # 右下角
                (max_x, max_y, z_height + 0.2),  # 右上角
                (min_x, max_y, z_height + 0.2),  # 左上角
            ]
            
            # 创建蓝色球体几何体用于标记顶点
            corner_sphere_geom = gymutil.WireframeSphereGeometry(
                0.15, 8, 8, None, color=(0.0, 0.0, 1.0)  # 蓝色
            )
            
            # 绘制四个顶点
            for i, (x, y, z) in enumerate(corners):
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(corner_sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
            
            print(f"ESDF边界顶点已绘制：({min_x:.2f},{min_y:.2f}) 到 ({max_x:.2f},{max_y:.2f})")
            print(f"地图尺寸：{map_width}×{map_height} 格子，分辨率：{resolution}m")
            
        except Exception as e:
            print(f"绘制ESDF边界顶点失败: {e}")

    def create_warp_tensor(self):
        self.warp_tensor_dict={}
        self.lidar_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.sensor_cfg.num_sensors, #1
                    self.sensor_cfg.vertical_line_num, #128
                    self.sensor_cfg.horizontal_line_num, #512
                    3, #3
                ),
                device=self.device,
                requires_grad=False,
            )        
        self.sensor_dist_tensor = torch.zeros(
                (
                    self.num_envs,  #4
                    self.sensor_cfg.num_sensors, #1
                    self.sensor_cfg.vertical_line_num, #128
                    self.sensor_cfg.horizontal_line_num, #512
                ),
                device=self.device,
                requires_grad=False,
            ) 
        #self.mesh_ids = self.mesh_ids_array = wp.array(self.warp_mesh_id_list, dtype=wp.uint64)
        self.sensor_pos_tensor = torch.zeros_like(self.root_states[:, 0:3])
        self.sensor_quat_tensor = torch.zeros_like(self.root_states[:, 3:7])
        
        
        self.sensor_translation = torch.tensor([0., 0.0, 0.436], device=self.device).repeat((self.num_envs, 1))
        rpy_offset = torch.tensor([3.14, 0, 0], device=self.device)

        self.sensor_offset_quat = quat_from_euler_xyz(rpy_offset[0], rpy_offset[1], rpy_offset[2]).repeat((self.num_envs, 1))
        # self.sensor_pos_tensor = self.root_states[:, 0:3]
        # self.sensor_quat_tensor = self.root_states[:, 3:7]
        
        self.warp_tensor_dict["sensor_dist_tensor"] = self.sensor_dist_tensor
        self.warp_tensor_dict['device'] = self.device
        self.warp_tensor_dict['num_envs'] = self.num_envs
        self.warp_tensor_dict['num_sensors'] = self.sensor_cfg.num_sensors
        self.warp_tensor_dict['sensor_pos_tensor'] = self.sensor_pos_tensor
        self.warp_tensor_dict['sensor_quat_tensor'] = self.sensor_quat_tensor
        self.warp_tensor_dict['mesh_ids'] = self.mesh_ids

    def create_warp_env(self):
            

            print("input lidar_triangles",self.lidar_triangles)
            print("input lidar_vertices",self.lidar_vertices)
            terrain_mesh = trimesh.Trimesh(vertices=self.lidar_vertices, faces=self.lidar_triangles)
            #save terrain mesh
            transform = np.zeros((3,))
            transform[0] = -self.terrain_cfg.border_size 
            transform[1] = -self.terrain_cfg.border_size
            transform[2] = 0.0
            translation = trimesh.transformations.translation_matrix(transform)
            terrain_mesh.apply_transform(translation)

            # current_script_dir = os.path.dirname(os.path.abspath(__file__))
            
            
            # two_levels_up = os.path.dirname(os.path.dirname(current_script_dir))
            
            
            # obstacle_mesh_path = os.path.join(two_levels_up, "resources", "robots","g1_29", "robot_combined.stl")
            
            # obstacle_mesh = trimesh.load(obstacle_mesh_path)


            #     #obstacle_mesh = trimesh.load(self.terrain_cfg.obstacle_config.obstacle_root_path+"/human/meshes/Male.OBJ")
            # transaltion = np.zeros((3,))
            # transaltion[0]=self.root_states[0,0]
            # transaltion[1]=self.root_states[0,1]
            # transaltion[2]=self.root_states[0,2]
            # # quat = self.root_states[0,3:7].numpy()
            # # rotation = trimesh.transformations.quaternion_matrix(quat)
            # translation = trimesh.transformations.translation_matrix(transaltion)
                
            # obstacle_mesh.apply_transform(translation)

            # combine_mesh = trimesh.util.concatenate([terrain_mesh, obstacle_mesh])
            
            #save combined mesh
            #combine_mesh.export("robot_terrain_combined.stl")
            vertices = terrain_mesh.vertices
            triangles = terrain_mesh.faces
            vertex_tensor = torch.tensor( 
                    vertices,
                    device=self.device,
                    requires_grad=False,
                    dtype=torch.float32,
                )
            
            #if none type in vertex_tensor
            if vertex_tensor.any() is None:
                print("vertex_tensor is None")
            vertex_vec3_array = wp.from_torch(vertex_tensor,dtype=wp.vec3)        
            faces_wp_int32_array = wp.from_numpy(triangles.flatten(), dtype=wp.int32,device=self.device)
                    
            self.wp_meshes =  wp.Mesh(points=vertex_vec3_array,indices=faces_wp_int32_array)
            
            self.mesh_ids = self.mesh_ids_array = wp.array([self.wp_meshes.id], dtype=wp.uint64)
    


    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
  
        clip_actions = self.cfg.normalization.clip_actions
        if (self.common_step_counter % self.cfg.domain_rand.upper_interval == 0):
            # (NOTE) implementation of upper-body curriculum
            self.random_upper_ratio = min(self.action_curriculum_ratio, 1.0)
            uu = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device)
            self.random_upper_ratio = -1.0 / (20 * (1-self.random_upper_ratio*0.99))*torch.log(1 - uu + uu * np.exp(-20 * (1-self.random_upper_ratio*0.99)))
            
            self.random_joint_ratio = self.random_upper_ratio * torch.rand(self.num_envs, self.num_actions - self.num_lower_dof).to(self.device)
            rand_pos = torch.rand(self.num_envs, self.num_actions - self.num_lower_dof, device=self.device) - 0.5
            self.random_upper_actions = ((self.action_min[:, self.num_lower_dof:] * (rand_pos >= 0)) + (self.action_max[:, self.num_lower_dof:] * (rand_pos < 0) ))* self.random_joint_ratio
            self.delta_upper_actions = (self.random_upper_actions - self.current_upper_actions) / (self.cfg.domain_rand.upper_interval)
            
        self.current_upper_actions += self.delta_upper_actions
        self.current_upper_actions[:] = 1.
        
        # print("current_upper_actions:",self.current_upper_actions.shape)
        # print("actions:",actions.shape)
        # actions = torch.cat((actions, self.current_upper_actions), dim=-1)
        # print("actions:",actions.shape)

        
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device) 
        self.origin_actions[:] = self.actions[:]
        self.delayed_actions = self.actions.clone().view(1, self.num_envs, self.num_actions).repeat(self.cfg.control.decimation, 1, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
                
        # Randomize Joint Injections
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        
        
        for _ in range(self.cfg.control.decimation):
            
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            # upper-body with position control; lower-body with force control;
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)



        termination_ids, termination_priveleged_obs = self.post_physics_step()
 
        """按 sensor_cfg.update_frequency 做节流更新；可选下采样/可视化"""
        self.sensor_update_time += self.dt
        self.base_pose[:] = self.root_states[:, :7]  
        self.base_quat[:] = self.root_states[:, 3:7]
        self.sensor_quat_tensor[:] = quat_mul(self.base_quat, self.sensor_offset_quat)
        self.sensor_pos_tensor[:]  = self.base_pose[:, :3] + quat_apply(self.base_quat, self.sensor_translation)
        self.lidar_tensor, self.sensor_dist_tensor = self.sensor.update()

        pts = self.lidar_tensor.view(self.num_envs, -1, 3)
        # print("base", self.base_pose[0, :3])
        down, _ = sample_farthest_points(pts, K=min(1000, pts.shape[1]))

        self.downsampled_cloud = down.view(self.num_envs, 1, down.shape[1], 3)
        env_id = getattr(self, "selected_env_idx", 0)  # 选择要发的那个 env
 
        pts_np = self.downsampled_cloud[env_id, 0].detach().contiguous().to('cpu', dtype=torch.float32).numpy()  # (K,3)
        # if getattr(self, "_pc_send_step", 0) % 3 == 0:
        self.pc_bridge.send_points(pts_np, frame_id="mid_360")  # frame_id 和 RViz Fixed Frame 对齐
        # self._pc_send_step = getattr(self, "_pc_send_step", 0) + 1


        pos = self.sensor_pos_tensor[env_id].detach().to('cpu').numpy()      # (3,)
        quat = self.sensor_quat_tensor[env_id].detach().to('cpu').numpy()    # (4,) xyzw
        self.pc_bridge.send_tf(parent_frame="map", child_frame="mid_360",
                       xyz=pos, quat_xyzw=quat)

        if self.sensor_update_time + 1e-9 > 1/self.sensor_cfg.update_frequency:
            self.gym.clear_lines(self.viewer)
            if self.downsampled_cloud is not None:
                print("stepping")
                # self._draw_lidar_vis()
                self.visualize_esdf()
            # self._visualize_terrain_vertices()

            self.sensor_update_time = 0.0
        self.downsampled_cloud= None
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)


        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs
   
    def _draw_lidar_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines

        
        #self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 0, 0))

        if self.sensor_cfg.pointcloud_in_world_frame:
            self.global_pixels =  self.downsampled_cloud
            for i in range(self.selected_env_idx,self.selected_env_idx+1):
                for j in range(int(self.global_pixels.shape[2])):
                    for k in range(self.global_pixels.shape[3]):
                        x = self.global_pixels[i, 0,j,k,0]#+self.root_states[:1, 0]
                        y = self.global_pixels[i, 0,j,k,1]
                        z = self.global_pixels[i, 0,j,k,2]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
        else:
            self.local_pixels_downsampled = self.downsampled_cloud.reshape(-1, 3)
            self.sensor_axis= self.sensor_pos_tensor[:,:]       
            pixels = self.local_pixels_downsampled.view(self.num_envs,-1,3)
            pixels_num = pixels.shape[1]
            sensor_axis_shaped = self.sensor_axis.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 3)
            sensor_quat = self.sensor_quat_tensor.unsqueeze(1).repeat(1, pixels_num, 1).view(self.num_envs, -1, 4)
            self.global_pixels = sensor_axis_shaped + quat_apply(sensor_quat, pixels)
            
            #def draw_line(p1, p2, color, gym, viewer, env):
            B=self.num_envs
            self.global_pixels.view(self.num_envs,-1, 3)
        
            for i in range(0,B):
                for j in range(0,self.global_pixels.shape[1]):
                        x = self.global_pixels[i, j,0]
                        y = self.global_pixels[i, j,1]
                        z = self.global_pixels[i, j,2]
                        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        feet_height, feet_height_var = self._get_feet_heights()
        self.feet_max_height = torch.maximum(self.feet_max_height, feet_height)
        
        # compute joint power
        joint_power = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((self.joint_powers[:, 1:], joint_power), dim=1)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        # self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt
        self.feet_max_height *= ~self.contact_filt

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 10., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.gravity_termination_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        # update action curriculum for specific dofs
        if self.cfg.env.action_curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_action_curriculum(env_ids)
            
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # resample commands
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.random_upper_actions[env_ids] = 0. 
        self.current_upper_actions[env_ids] = 0.
        self.delta_upper_actions[env_ids] = 0.
        reset_roll, reset_pitch, reset_yaw = euler_from_quaternion(self.base_quat[env_ids])
        self.roll[env_ids] = reset_roll
        self.pitch[env_ids] = reset_pitch
        self.yaw[env_ids] = reset_yaw
        self.reset_buf[env_ids] = 1
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            # self.extras["episode"]["height_curriculum_ratio"] = self.height_curriculum_ratio
        if self.cfg.env.action_curriculum:
            self.extras["episode"]["action_curriculum_ratio"] = self.action_curriculum_ratio
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if torch.isnan(rew).any():
                import ipdb; ipdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def compute_observations(self):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions[:, :12],
                                    ),dim=-1)
        current_actor_obs = torch.clone(current_obs)
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof)]           
        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_proprioceptive_obs_length], current_actor_obs[:, :self.num_one_step_obs]), dim=-1)
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        self.privileged_obs_buf = torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        imu_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.rigid_body_states[:, self.imu_index,10:13])
        imu_projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.imu_index,3:7], self.gravity_vec)
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.commands[:, 4].unsqueeze(1),
                                    imu_ang_vel  * self.obs_scales.ang_vel,
                                    imu_projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions[:, :12],
                                    ),dim=-1)

        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(10 + 2 * self.num_actions + self.num_lower_dof)]
        current_critic_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel), dim=-1)
        return torch.cat((self.privileged_obs_buf[:, self.num_one_step_privileged_obs:self.critic_proprioceptive_obs_length], current_critic_obs), dim=-1)[env_ids]
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        self._create_ground()
        
        self._create_envs()

    def _create_ground(self):
        """Create a ground plane."""
        self.terrain_cfg = Terrain_cfg()
        self.terrain = Terrain(self.terrain_cfg, self.num_envs)
        self._create_trimesh()
        self._create_random_obstacle()
        # self._add_static_box_trimesh(center=(4.0, 5.0, 1.0), size=(1.6, 1.6, 10.0), yaw_rad=np.pi/3)
        # self._add_static_box_trimesh(center=(15.0, 15.0, 1.0), size=(4, 4, 10.0), yaw_rad=np.pi/4)

    def _create_random_obstacles(self):
        """Create random static obstacles
            根据地形面积，障碍物密度，随机生成障碍物位置和大小和方向，给定一些参数用来控制障碍物生成
        """
        import random
        
        # 障碍物生成参数
        obstacle_density = 0.2  # 障碍物密度：每平方米的障碍物数量
        min_obstacle_distance = 5.0  # 障碍物之间的最小距离
        robot_safe_radius = 2.0  # 机器人起始位置的安全半径
        min_required_distance = 2
        # 地形参数计算 - 30x30地形的实际尺寸
        # 配置: terrain_length=10, terrain_width=10, num_rows=2, num_cols=2
        # 实际地形 = (terrain_length * num_cols) x (terrain_width * num_rows) = 20x20
        # 加上边界 = (20 + 2*border_size) x (20 + 2*border_size) = 30x30
        
        single_terrain_length = self.terrain_cfg.terrain_length  # 10m
        single_terrain_width = self.terrain_cfg.terrain_width    # 10m
        num_rows = self.terrain_cfg.num_rows  # 2
        num_cols = self.terrain_cfg.num_cols  # 2
        border_size = self.terrain_cfg.border_size  # 5m
        
        # 实际地形尺寸（不包括边界）
        terrain_total_length = single_terrain_length * num_cols  # 20m
        terrain_total_width = single_terrain_width * num_rows    # 20m
        
        # 障碍物生成区域（在地形坐标系中，_add_static_box_trimesh会自动处理border_size平移）
        min_x = 1.0  # 距离地形边缘1m
        max_x = terrain_total_length - 1.0  # 19m
        min_y = 1.0  # 距离地形边缘1m  
        max_y = terrain_total_width - 1.0   # 19m
        
        # 计算需要生成的障碍物数量
        effective_area = (max_x - min_x) * (max_y - min_y)
        num_obstacles = int(effective_area * obstacle_density)
        
        print(f"=== 随机障碍物生成 ===")
        print(f"地形配置: {single_terrain_length}×{single_terrain_width}m × {num_rows}×{num_cols} + border({border_size}m)")
        print(f"实际地形: {terrain_total_length}×{terrain_total_width}m, 生成区域: ({min_x}-{max_x}, {min_y}-{max_y})")
        print(f"有效面积: {effective_area:.1f}m², 障碍物密度: {obstacle_density}/m², 计划数量: {num_obstacles}")
        
        # 障碍物大小范围
        size_ranges = {
            'small': (0.3, 0.6),    # 小障碍物
            'medium': (0.6, 1.2),   # 中等障碍物
            'large': (1.5, 2.0),    # 大障碍物
        }
        
        # 障碍物高度范围
        height_range = (0.5, 1.5)

        # 障碍物类型权重（小:中:大 = 2:7:1）
        obstacle_types = ['small'] * 3 + ['medium'] * 7 + ['large'] * 0
        
        # 记录已生成的障碍物位置，用于避免重叠
        existing_positions = []
        
        successful_obstacles = 0
        max_attempts = num_obstacles * 5  # 最大尝试次数，避免无限循环
        
        for _ in range(max_attempts):
            if successful_obstacles >= num_obstacles:
                break
                
            # 随机选择障碍物类型和尺寸
            obstacle_type = random.choice(obstacle_types)
            size_min, size_max = size_ranges[obstacle_type]
            
            # 随机生成障碍物参数
            width = random.uniform(size_min, size_max)
            length = random.uniform(size_min, size_max)
            height = random.uniform(*height_range)
            
            # 随机生成位置
            x = random.uniform(min_x + width/2, max_x - width/2)
            y = random.uniform(min_y + length/2, max_y - length/2)
            z = height / 2.0  # 障碍物中心高度
            
            # 检查与机器人起始位置的距离（假设机器人在原点附近）
            robot_distance = np.sqrt(x**2 + y**2)
            if robot_distance < robot_safe_radius:
                continue
            
            # 检查与现有障碍物的距离
            too_close = False
            for ex_x, ex_y, ex_w, ex_l in existing_positions:
                distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
                # 计算两个障碍物中心之间的最小安全距离
                current_radius = max(width, length) / 2
                existing_radius = max(ex_w, ex_l) / 2
                
                if distance < min_required_distance:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # 随机旋转角度
            yaw_rad = random.uniform(0, 2 * np.pi)
            
            # 生成障碍物
            self._add_static_box_trimesh(
                center=(x, y, z), 
                size=(width, length, height), 
                yaw_rad=yaw_rad
            )
            
            # 记录位置
            existing_positions.append((x, y, width, length))
            successful_obstacles += 1
            
            print(f"障碍物 {successful_obstacles}: {obstacle_type} 尺寸={width:.1f}×{length:.1f}×{height:.1f}m, 位置=({x:.1f},{y:.1f}), 角度={np.degrees(yaw_rad):.0f}°")
        
        print(f"成功生成 {successful_obstacles} 个随机障碍物 (计划{num_obstacles}个)")

    def _create_random_obstacle(self):
        """生成更均匀分布的随机障碍物，避开 border"""
        import random
        import numpy as np

        # 参数
        obstacle_density = 0.1       # 每平方米障碍物数量（建议降低一些）
        robot_safe_radius = 2.0      # 机器人安全区
        min_required_distance = 2.0  # 最小障碍间距（米）

        # 地形总尺寸（不含 border）
        terrain_total_length = self.terrain_cfg.terrain_length * self.terrain_cfg.num_cols
        terrain_total_width  = self.terrain_cfg.terrain_width * self.terrain_cfg.num_rows
        border_size = self.terrain_cfg.border_size

        # 有效生成区域（避开 border）
        min_x = border_size
        max_x = border_size + terrain_total_length
        min_y = border_size
        max_y = border_size + terrain_total_width

        # 有效面积
        effective_area = (max_x - min_x) * (max_y - min_y)
        num_obstacles = int(effective_area * obstacle_density)

        print(f"=== 随机障碍物生成 ===")
        print(f"实际可用区域: X=[{min_x},{max_x}], Y=[{min_y},{max_y}]")
        print(f"有效面积: {effective_area:.1f} m², 计划障碍物数: {num_obstacles}")

        # 障碍物尺寸范围
        size_ranges = {
            'small': (0.3, 0.6),
            'medium': (0.6, 1.2),
            'large': (1.5, 2.0),
        }
        height_range = (0.5, 1.5)
        obstacle_types = ['small'] * 2 + ['medium'] * 7 + ['large'] * 1

        # === 网格均匀采样 ===
        grid_rows = int(np.sqrt(num_obstacles))
        grid_cols = grid_rows
        cell_w = (max_x - min_x) / grid_cols
        cell_h = (max_y - min_y) / grid_rows

        successful_obstacles = 0
        existing_positions = []

        for r in range(grid_rows):
            for c in range(grid_cols):
                if successful_obstacles >= num_obstacles:
                    break

                # cell 中心 + 抖动
                cx = min_x + (c + 0.5) * cell_w + random.uniform(-0.4, 0.4) * cell_w
                cy = min_y + (r + 0.5) * cell_h + random.uniform(-0.4, 0.4) * cell_h

                # 避开机器人起始位置
                if np.hypot(cx, cy) < robot_safe_radius:
                    continue

                # 随机选类型
                obstacle_type = random.choice(obstacle_types)
                size_min, size_max = size_ranges[obstacle_type]
                width = random.uniform(size_min, size_max)
                length = random.uniform(size_min, size_max)
                height = random.uniform(*height_range)
                z = height / 2.0

                # 距离检查
                too_close = False
                for ex_x, ex_y, ex_w, ex_l in existing_positions:
                    if np.hypot(cx - ex_x, cy - ex_y) < min_required_distance:
                        too_close = True
                        break
                if too_close:
                    continue

                # 随机旋转
                yaw_rad = random.uniform(0, 2*np.pi)

                # 生成障碍物
                self._add_static_box_trimesh(
                    center=(cx, cy, z),
                    size=(width, length, height),
                    yaw_rad=yaw_rad
                )

                existing_positions.append((cx, cy, width, length))
                successful_obstacles += 1
                print(f"障碍物 {successful_obstacles}: {obstacle_type}, "
                    f"尺寸={width:.2f}×{length:.2f}×{height:.2f}, "
                    f"位置=({cx:.1f},{cy:.1f}), 角度={np.degrees(yaw_rad):.0f}°")

        print(f"成功生成 {successful_obstacles}/{num_obstacles} 个障碍物")


    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.terrain_cfg.static_friction
        tm_params.dynamic_friction = self.terrain_cfg.dynamic_friction
        tm_params.restitution = self.terrain_cfg.restitution

        vertices = self.terrain.vertices.astype(np.float32).flatten(order='C')
        triangles = self.terrain.triangles.astype(np.uint32).flatten(order='C')

        self.gym.add_triangle_mesh(self.sim, vertices, triangles, tm_params)

        # 保存给 LiDAR 用
        self.lidar_vertices = self.terrain.vertices
        self.lidar_triangles = self.terrain.triangles

        # # 假设 vertices: (Nv,3) float, triangles: (Nt,3) int
        # vertices = self.lidar_vertices
        # triangles = self.lidar_triangles

        # Nv = vertices.shape[0]
        # Nt = triangles.shape[0]
        # print("=== 地形网格密度和分辨率分析 ===")
        # print(f"顶点数量: {Nv}")
        # print(f"面片数量: {Nt}")

        # if Nv == 0 or Nt == 0:
        #     print("网格为空，跳过分析。")
        #     print("=== 分析完成 ===\n")
        # else:
        #     # 包围盒估算（注意这只是 XY 外接矩形面积）
        #     mins = vertices.min(axis=0)
        #     maxs = vertices.max(axis=0)
        #     size = maxs - mins
        #     area_xy = float(size[0] * size[1])

        #     print(f"地形边界: X=[{mins[0]:.2f}, {maxs[0]:.2f}], "
        #         f"Y=[{mins[1]:.2f}, {maxs[1]:.2f}], Z=[{mins[2]:.2f}, {maxs[2]:.2f}]")
        #     print(f"地形尺寸: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} 米")
        #     print(f"地形XY外接矩形面积(估算): {area_xy:.2f} 平方米")

        #     if area_xy > 0:
        #         vertex_density = Nv / area_xy
        #         print(f"顶点密度(基于包围盒): {vertex_density:.2f} 顶点/平方米")
        #     else:
        #         print("警告: XY面积为0，无法计算顶点密度。")

        #     # ---- 面片面积（矢量化）----
        #     # 选取采样索引：全量或随机子集
        #     max_sample_tri = min(20000, Nt)  # 可调整上限
        #     if Nt > max_sample_tri:
        #         idx = np.random.default_rng().choice(Nt, size=max_sample_tri, replace=False)
        #     else:
        #         idx = np.arange(Nt)

        #     T = triangles[idx]  # (Ns,3)
        #     v0 = vertices[T[:,0]]
        #     v1 = vertices[T[:,1]]
        #     v2 = vertices[T[:,2]]
        #     # 面积 = 0.5 * |(v1-v0) x (v2-v0)|
        #     areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        #     # 过滤退化
        #     areas = areas[np.isfinite(areas) & (areas > 1e-12)]

        #     if areas.size > 0:
        #         print(f"面片面积统计 (采样{areas.size}个面片):")
        #         print(f"  平均: {areas.mean():.6f}  中位数: {np.median(areas):.6f} "
        #             f"  [5%,95%]: {np.percentile(areas,5):.6f}, {np.percentile(areas,95):.6f}")

        #         # 如果一定要从面积推边长（近似等边）
        #         est_edge_from_area = np.sqrt(4.0 * areas.mean() / np.sqrt(3.0))
        #         print(f"  (等边近似) 面片等效边长: {est_edge_from_area:.4f} m")
        #     else:
        #         print("面片面积统计数据不足（可能存在大量退化三角形）。")

        #     # ---- 边长统计（矢量化）----
        #     max_sample_tri_edges = min(10000, Nt)
        #     if Nt > max_sample_tri_edges:
        #         idx_e = np.random.default_rng().choice(Nt, size=max_sample_tri_edges, replace=False)
        #     else:
        #         idx_e = np.arange(Nt)

        #     T_e = triangles[idx_e]
        #     v0 = vertices[T_e[:,0]]
        #     v1 = vertices[T_e[:,1]]
        #     v2 = vertices[T_e[:,2]]
        #     e01 = np.linalg.norm(v1 - v0, axis=1)
        #     e12 = np.linalg.norm(v2 - v1, axis=1)
        #     e20 = np.linalg.norm(v0 - v2, axis=1)

        #     edges = np.concatenate([e01, e12, e20])
        #     edges = edges[np.isfinite(edges) & (edges > 1e-12)]

        #     if edges.size > 0:
        #         print(f"边长统计 (采样{edges.size}条边):")
        #         print(f"  平均: {edges.mean():.4f}  中位数: {np.median(edges):.4f} "
        #             f"  众数近似(5%区间内频次最大bin中心): {np.quantile(edges, 0.5):.4f}")
        #         print(f"  最小: {edges.min():.4f}  最大: {edges.max():.4f}")

        #         # 建议把“分辨率”报告为边长分布的"中位数/众数"
        #         est_resolution = float(np.median(edges))
        #         print(f"  建议报告的地形‘分辨率’(边长中位数): {est_resolution:.4f} m")
        #     else:
        #         print("边长统计数据不足。")

        #     print("=== 分析完成 ===\n")
        
      

    def _visualize_terrain_vertices(self):
        """在viewer初始化完成后可视化地形顶点"""
        print("=== 可视化地形顶点 ===")
        
        # 检查是否有viewer可用（在GUI模式下才有viewer）
        if hasattr(self, 'viewer') and self.viewer is not None:
            try:
                # 创建用于可视化的小球几何体 - 用红色表示地形顶点
                sphere_geom = gymutil.WireframeSphereGeometry(0.05, 32,32, None, color=(1, 0, 0))
                
                # 由于顶点太多，需要采样显示，避免可视化过于密集
                vertices = self.lidar_vertices
                total_vertices = len(vertices)
                
                # 采样策略：使用farthest point sampling，最多显示1000个顶点
                max_display_vertices = 7000
                border_size = self.terrain.cfg.border_size
                
                if total_vertices > max_display_vertices:
                    # 使用 farthest point sampling 替代均匀采样
                    vertices_tensor = torch.tensor(vertices, dtype=torch.float32, device=self.device).unsqueeze(0)
                    sampled_vertices, _ = sample_farthest_points(vertices_tensor, K=max_display_vertices)
                    sampled_vertices = sampled_vertices.squeeze(0).cpu().numpy()
                    print(f"使用farthest point sampling显示 {max_display_vertices} 个顶点 (从总共 {total_vertices} 个中)")
                else:
                    sampled_vertices = vertices
                    print(f"显示全部 {total_vertices} 个顶点")
                
                # 为第一个环境绘制采样的地形顶点
                for vertex in sampled_vertices:
                    x = float(vertex[0]) - border_size
                    y = float(vertex[1]) - border_size
                    z = float(vertex[2])
                    # 创建球体位置
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    
                    # 绘制球体
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose)
                
                print("地形顶点可视化完成 (红色小球)")
                
            except Exception as e:
                print(f"可视化失败: {e}")
        else:
            print("无viewer可用，跳过可视化 (运行在headless模式)")
        print()

    def _add_static_box_trimesh(self, center, size, yaw_rad=0.0, friction=1.0, restitution=0.0):
        # center: (cx, cy, cz), size: (sx, sy, sz)
        cx, cy, cz = center
        sx, sy, sz = size
        # 8个顶点
        vx = sx * 0.5; vy = sy * 0.5; vz = sz * 0.5
        # 立方体8点
        V = np.array([
            [cx-vx, cy-vy, cz-vz],
            [cx+vx, cy-vy, cz-vz],
            [cx+vx, cy+vy, cz-vz],
            [cx-vx, cy+vy, cz-vz],
            [cx-vx, cy-vy, cz+vz],
            [cx+vx, cy-vy, cz+vz],
            [cx+vx, cy+vy, cz+vz],
            [cx-vx, cy+vy, cz+vz],
        ], dtype=np.float32)

            # 2) 绕Z轴旋转（围绕center）
        if yaw_rad != 0.0:
            c, s = np.cos(yaw_rad, dtype=np.float32), np.sin(yaw_rad, dtype=np.float32)
            Rz = np.array([[c, -s, 0.0],
                        [s,  c, 0.0],
                        [0.0,0.0,1.0]], dtype=np.float32)
            V_rel = V - np.array([[cx, cy, cz]], dtype=np.float32)  # 相对中心
            V_rot = (V_rel @ Rz.T) + np.array([[cx, cy, cz]], dtype=np.float32)
            V = V_rot.astype(np.float32)

        # 12个三角面（每面2三角）
        T = np.array([
            [0,1,2], [0,2,3],   # bottom
            [4,5,6], [4,6,7],   # top
            [0,1,5], [0,5,4],   # -y
            [1,2,6], [1,6,5],   # +x
            [2,3,7], [2,7,6],   # +y
            [3,0,4], [3,4,7],   # -x
        ], dtype=np.uint32)

        # 提交给模拟器（每个障碍单独一份tm_params即可）
        tm = gymapi.TriangleMeshParams()
        tm.nb_vertices   = V.shape[0]
        tm.nb_triangles  = T.shape[0]
        tm.transform.p.x = -self.terrain.cfg.border_size
        tm.transform.p.y = -self.terrain.cfg.border_size
        tm.transform.p.z = 0.0
        tm.static_friction = friction
        tm.dynamic_friction = friction
        tm.restitution = restitution
        self.gym.add_triangle_mesh(self.sim, V.flatten(order='C'), T.flatten(order='C'), tm)


        base_idx = self.lidar_vertices.shape[0]
        # 不对box进行坐标变换，因为create_warp_env中会统一处理
        V_world = V.copy()

        res = getattr(self,"map_resolution",0.05)
        edge_pts = self._densify_edges(V_world, step=0.5*res)  # 采样间隔取 0.5*分辨率

        self.lidar_vertices  = np.vstack([self.lidar_vertices, V_world, edge_pts])
        self.lidar_triangles = np.vstack([self.lidar_triangles, T + base_idx])


    def _densify_edges(self, V, step):
        """对立方体的 12 条边插值"""
        edges = [
            (0,1),(1,2),(2,3),(3,0),   # bottom
            (4,5),(5,6),(6,7),(7,4),   # top
            (0,4),(1,5),(2,6),(3,7),   # vertical
        ]
        pts = []
        for i0,i1 in edges:
            p0, p1 = V[i0], V[i1]
            L = np.linalg.norm(p1[:2] - p0[:2])   # 只看 XY 平面长度
            n = max(1, int(np.ceil(L/step)))
            ts = np.linspace(0, 1, n+1)
            pts.append((1-ts)[:,None]*p0 + ts[:,None]*p1)
        return np.vstack(pts).astype(np.float32)

    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, torso_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)
            
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id): #给每个机器人都随机初始化刚体摩擦系数
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):#可在这个函数对关节属性进行修改
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[self.torso_body_index].mass = self.default_rigid_body_mass[self.torso_body_index] + self.payload[env_id, 0]
            props[self.left_hand_index].mass = self.default_rigid_body_mass[self.left_hand_index] + self.hand_payload[env_id, 0]
            props[self.right_hand_index].mass = self.default_rigid_body_mass[self.right_hand_index] + self.hand_payload[env_id, 1]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
        if self.cfg.domain_rand.randomize_body_displacement:
            props[self.torso_body_index].com = self.default_body_com + gymapi.Vec3(self.body_displacement[env_id, 0], self.body_displacement[env_id, 1], self.body_displacement[env_id, 2])

        
        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
                
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        set_x = torch.rand(len(env_ids), 1).to(self.device)
        is_height = set_x < 1/3
        is_vel = set_x > 1/2
        self.commands[env_ids, 0] = (torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        self.commands[env_ids, 1] = (torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1) 
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = (torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        else:
            self.commands[env_ids, 2] = (torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device) * is_vel).squeeze(1)
            self.commands[env_ids, 4] = (torch_rand_float(self.command_ranges["height"][0], self.command_ranges["height"][1], (len(env_ids), 1), device=self.device) * is_height).squeeze(1) + self.cfg.rewards.base_height_target # height
        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        self.joint_pos_target = self.default_dof_pos + actions_scaled
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
            torques = torques + self.actuation_offset + self.joint_injection
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        elif control_type=="M":
            torques = self.p_gains * self.Kp_factors * (
                    self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            
            torques = torques + self.actuation_offset + self.joint_injection
            torques = torch.clip(torques, -self.torque_limits, self.torque_limits)
            return torch.cat((torques[..., :self.num_lower_dof], self.joint_pos_target[..., self.num_lower_dof:]), dim=-1)
        
        else:
            raise NameError(f"Unknown controller type: {control_type}")

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        dof_upper = self.dof_pos_limits[:, 1].view(1, -1)
        dof_lower = self.dof_pos_limits[:, 0].view(1, -1)
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            init_dos_pos = self.default_dof_pos * torch_rand_float(self.cfg.domain_rand.initial_joint_pos_scale[0], self.cfg.domain_rand.initial_joint_pos_scale[1], (len(env_ids), self.num_dof), device=self.device)
            init_dos_pos += torch_rand_float(self.cfg.domain_rand.initial_joint_pos_offset[0], self.cfg.domain_rand.initial_joint_pos_offset[1], (len(env_ids), self.num_dof), device=self.device)
            self.dof_pos[env_ids] = torch.clip(init_dos_pos, dof_lower, dof_upper)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch.ones((len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2:3] += torch_rand_float(0.0, 0.1, (len(env_ids), 1), device=self.device) # z position within 0.1m of the ground
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self): #随即给机器人一个推力，训练抗干扰能力
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 75% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]) and (torch.mean(self.episode_sums["tracking_y_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_y_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)

        
    def update_action_curriculum(self, env_ids):
        """ Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if (torch.mean(self.episode_sums["tracking_x_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_x_vel"]):
            self.action_curriculum_ratio += 0.05
            self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(10 + 2*self.num_actions + self.num_lower_dof, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:4] = 0. # commands
        noise_vec[4:7] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[7:10] = noise_scales.gravity * noise_level
        noise_vec[10:(10 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(10 + self.num_actions):(10 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(10 + 2 * self.num_actions):(10 + 2 * self.num_actions + self.num_lower_dof)] = 0. # previous actions
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.origin_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        
        self.actor_jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)

        self.whole_jac = gymtorch.wrap_tensor(self.actor_jacobian) 

        self.base_pose = self.root_states[:, 0:7]
        self.base_quat = self.root_states[:, 3:7]

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / self.cfg.control.action_scale
        self.action_curriculum_ratio = self.cfg.domain_rand.init_upper_ratio
        self.target_heights = torch.ones((self.num_envs), device=self.device) * self.cfg.rewards.base_height_target
        print(f"Action min: {self.action_min}")
        print(f"Action max: {self.action_max}")
        
        self.random_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.current_upper_actions = torch.zeros((self.num_envs, self.num_actions - self.num_lower_dof), device=self.device)
        self.delta_upper_actions = torch.zeros((self.num_envs, 1), device=self.device)
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_actions), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs ,2), device=self.device)

        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        self.body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(self.body_names)
        self.num_dof = len(self.dof_names)
        feet_names = [s for s in self.body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in self.body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in self.body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in self.body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)
        self.cfg.init_state.pos[2] += 0.02
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.hand_payload = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
            self.hand_payload = torch_rand_float(self.cfg.domain_rand.hand_payload_mass_range[0], self.cfg.domain_rand.hand_payload_mass_range[1], (self.num_envs, 2), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.randomize_body_displacement:
            self.body_displacement = torch_rand_float(self.cfg.domain_rand.body_displacement_range[0], self.cfg.domain_rand.body_displacement_range[1], (self.num_envs, 3), device=self.device)
        
        self.torso_body_index = self.body_names.index("torso_link")
        self.left_hand_index = self.body_names.index("left_hand_palm_link")
        self.right_hand_index = self.body_names.index("right_hand_palm_link")    
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
        
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            dof_props["driveMode"][12:].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][12:] = [300., 200., 200., 200., 100.,  20.,  20.,  20., 200., 200., 200., 100.,  20.,  20.,  20.]
            dof_props["damping"][12:] = [5.0000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000,
                                            0.5000, 4.0000, 4.0000, 4.0000, 1.0000, 0.5000, 0.5000, 0.5000]
        
        
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                self.default_body_com = copy.deepcopy(body_props[self.torso_body_index].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])
            
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_foot_names[i])
        
        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_foot_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
      
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.asset.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.asset.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
        
        self.left_hip_joint_indices = torch.zeros(len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
        
        self.hip_pitch_joint_indices = torch.zeros(len(self.cfg.asset.hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.hip_pitch_joints)):
            self.hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.hip_pitch_joints[i])
    
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])
            
        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])
            
        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.upper_body_link)
        self.imu_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.asset.imu_link)

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
        self.cfg.domain_rand.upper_interval = np.ceil(self.cfg.domain_rand.upper_interval_s / self.dt)

    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices, :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices, :3].clone()
        if self.cfg.terrain.mesh_type == 'plane':
            left_foot_height = torch.mean(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            left_foot_height_var = torch.var(left_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height = torch.mean(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            right_foot_height_var = torch.var(right_foot_pos[:, :, 2], dim = -1, keepdim=True)
            return torch.cat((left_foot_height, right_foot_height), dim=-1), torch.cat((left_foot_height_var, right_foot_height_var), dim=-1)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            left_points = left_foot_pos[env_ids].clone()
            right_points = right_foot_pos[env_ids].clone()
        else:
            left_points = left_foot_pos.clone()
            right_points = right_foot_pos.clone()

        left_points += self.terrain.cfg.border_size
        right_points += self.terrain.cfg.border_size
        left_points = (left_points/self.terrain.cfg.horizontal_scale).long()
        right_points = (right_points/self.terrain.cfg.horizontal_scale).long()
        left_px = left_points[:, :, 0].view(-1)
        right_px = right_points[:, :, 0].view(-1)
        left_py = left_points[:, :, 1].view(-1)
        right_py = right_points[:, :, 1].view(-1)
        left_px = torch.clip(left_px, 0, self.height_samples.shape[0]-2)
        right_px = torch.clip(right_px, 0, self.height_samples.shape[0]-2)
        left_py = torch.clip(left_py, 0, self.height_samples.shape[1]-2)
        right_py = torch.clip(right_py, 0, self.height_samples.shape[1]-2)

        left_heights1 = self.height_samples[left_px, left_py]
        left_heights2 = self.height_samples[left_px+1, left_py]
        left_heights3 = self.height_samples[left_px, left_py+1]
        left_heights = torch.min(left_heights1, left_heights2)
        left_heights = torch.min(left_heights, left_heights3)
        left_heights = left_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        left_foot_heights =  left_foot_pos[:, :, 2] - left_heights

        right_heights1 = self.height_samples[right_px, right_py]
        right_heights2 = self.height_samples[right_px+1, right_py]
        right_heights3 = self.height_samples[right_px, right_py+1]
        right_heights = torch.min(right_heights1, right_heights2)
        right_heights = torch.min(right_heights, right_heights3)
        right_heights = right_heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        right_foot_heights =  right_foot_pos[:, :, 2] - right_heights

        feet_heights = torch.cat((torch.mean(left_foot_heights, dim=-1, keepdim=True), torch.mean(right_foot_heights, dim=-1, keepdim=True)), dim=-1)
        feet_heights_var = torch.cat((torch.var(left_foot_heights, dim=-1, keepdim=True), torch.var(right_foot_heights, dim=-1, keepdim=True)), dim=-1)

        return torch.clip(feet_heights, min=0.), feet_heights_var

    #------------ reward functions----------------
    def _reward_tracking_x_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_y_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, 1:2] - self.base_lin_vel[:, 1:2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2]) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_tracking_base_height(self):
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        height_error = torch.abs(base_height - self.commands[:, 4] + self.cfg.asset.ankle_sole_distance)
        return torch.exp(-height_error * 4)
    
    def _reward_deviation_hip_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_joint_indices], dim=-1) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_deviation_ankle_joint(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos)[:, self.ankle_joint_indices], dim=-1) *  (self.commands[:, 4] >= 0.735)
    
    def _reward_deviation_knee_joint(self):
        height_error = (self.root_states[:, 2] - self.commands[:, 4])
        knee_action_min = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_min[:, self.knee_joint_indices]
        knee_action_max = self.default_dof_pos[:, self.knee_joint_indices] + self.cfg.control.action_scale * self.action_max[:, self.knee_joint_indices]
        joint_deviation = (self.dof_pos[:, self.knee_joint_indices] - knee_action_min) / (knee_action_max - knee_action_min) # always positive
        return torch.sum(torch.abs((joint_deviation-0.5) * height_error.unsqueeze(-1)), dim=-1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0])[:, :self.num_actions].clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1])[:, :self.num_actions].clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * self.first_contacts, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > 0.1 # no reward for zero command
        return rew_airTime
    
    def _reward_feet_clearance(self):
        cur_feetvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        feetvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            feetvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_feetvel_translated[:, i, :])
        feet_height, feet_height_var = self._get_feet_heights()
        height_error = torch.square(feet_height - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        feet_leteral_vel = torch.sqrt(torch.sum(torch.square(feetvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * feet_leteral_vel, dim=1) * (self.commands[:, 4]>=0.71)
    
    def _reward_feet_distance_lateral(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        foot_leteral_dis = torch.abs(footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1])
        return torch.clamp(foot_leteral_dis - self.cfg.rewards.least_feet_distance_lateral, max=0) + torch.clamp(-foot_leteral_dis + self.cfg.rewards.most_feet_distance_lateral, max=0) * (self.commands[:, 4] >= 0.735)
    
    def _reward_knee_distance_lateral(self):
        cur_knee_pos_translated = self.rigid_body_states[:, self.knee_indices, :3].clone() - self.root_states[:, 0:3].unsqueeze(1)
        knee_pos_in_body_frame = torch.zeros(self.num_envs, len(self.knee_indices), 3, device=self.device)
        for i in range(len(self.knee_indices)):
            knee_pos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_knee_pos_translated[:, i, :])
        knee_lateral_dis = torch.abs(knee_pos_in_body_frame[:, 0, 1] - knee_pos_in_body_frame[:, 2, 1]) + torch.abs(knee_pos_in_body_frame[:, 1, 1] - knee_pos_in_body_frame[:, 3, 1])
        return torch.clamp(knee_lateral_dis - self.cfg.rewards.least_knee_distance_lateral * 2, max=0) + torch.clamp(-knee_lateral_dis + self.cfg.rewards.most_knee_distance_lateral * 2, max=0) * (self.commands[:, 4] >= 0.735)
    
    def _reward_feet_ground_parallel(self):
        feet_heights, feet_heights_var = self._get_feet_heights()
        continue_contact = (self.feet_air_time >= 3* self.dt) * self.contact_filt
        return torch.sum(feet_heights_var * continue_contact, dim=1)
    
    def _reward_feet_parallel(self):
        left_foot_pos = self.rigid_body_states[:, self.left_foot_indices[0:3], :3].clone()
        right_foot_pos = self.rigid_body_states[:, self.right_foot_indices[0:3], :3].clone()
        feet_distances = torch.norm(left_foot_pos - right_foot_pos, dim=2)
        feet_distances_var = torch.var(feet_distances, dim=1)
        return feet_distances_var * (self.commands[:, 4] >= 0.735)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:, 0:2]), dim=-1) + 0.2 * torch.square(self.commands[:, 2]), min=0.1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 3 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square((self.torques / self.p_gains.unsqueeze(0))[:, :self.num_lower_dof]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel[:, :self.num_lower_dof]), dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit)[:, :self.num_lower_dof].clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit)[:, :self.num_lower_dof].clip(min=0.), dim=1)
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.5
        single_contact = torch.sum(1.*contacts, dim=1)==1
        rew_no_fly = 1.0 * single_contact
        rew_no_fly = torch.max(rew_no_fly, 1. * (torch.norm(self.commands[:, :3], dim=1) < 0.1)) # full reward for zero command
        return rew_no_fly
    
    def _reward_joint_tracking_error(self):
        return torch.sum(torch.square(self.joint_pos_target[:, :self.num_lower_dof] - self.dof_pos[:, :self.num_lower_dof]), dim=-1)
    
    def _reward_feet_slip(self): 
        # Penalize feet slipping
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.sum(torch.norm(self.feet_vel[:,:,:2], dim=2) * contact, dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_contact_momentum(self):
        # encourage soft contacts
        feet_contact_momentum_z = torch.clip(self.feet_vel[:, :, 2], max=0) * torch.clip(self.contact_forces[:, self.feet_indices, 2] - 50, min=0)
        return torch.sum(feet_contact_momentum_z, dim=1)
    
    def _reward_action_vanish(self):
        upper_error = torch.clip(self.origin_actions[:, :self.num_lower_dof] - self.action_max[:, :self.num_lower_dof], min=0)
        lower_error = torch.clip(self.action_min[:, :self.num_lower_dof] - self.origin_actions[:, :self.num_lower_dof], min=0)
        return torch.sum(upper_error + lower_error, dim=-1)
    
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        contacts = torch.sum(self.contact_forces[:, self.feet_indices, 2] < 0.1, dim=-1)
        error_sim = (contacts) * (self.commands[:, 4] >= 0.735)
        return error_sim * (torch.norm(self.commands[:, :3], dim=1) < 0.1)
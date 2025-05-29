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

import math
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import keyboard 
import onnxruntime as ort
import pygame
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymapi
import numpy as np
import torch


def load_policy():
    body = torch.jit.load("", map_location="cuda:0")
    def policy(obs):
        action = body.forward(obs)
        return action
    return policy

def load_onnx_policy():
    model = ort.InferenceSession("")
    def run_inference(input_tensor):
        ort_inputs = {model.get_inputs()[0].name: input_tensor.cpu().numpy()}
        ort_outs = model.run(None, ort_inputs)
        return torch.tensor(ort_outs[0], device="cuda:0")
    return run_inference

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
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
    env_cfg.terrain.mesh_type = 'plane'
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
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file
    
    # policy = load_onnx_policy() # Use this to load from exported onnx file
    
    #pygame
    pygame.init()
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.set_mode((500,300))  # 小窗口即可，仅为捕获键盘
    pygame.display.set_caption("HomieL 控制窗口")
    clock = pygame.time.Clock()
    
    
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
        
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    # env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    cpu_inds = torch.arange(env.num_envs, dtype=torch.int32)
    env.reset_idx(cpu_inds)
    vx = vy = yaw = 0.0

    #创建目标点
    arrive_tollerance  = 0.3 #pos tollerance
    heading_thresh = 0.1 #yaw tollerance
    max_speed=1
    viewer = env.viewer
    env.gym.clear_lines(viewer)
    np.random.seed(123)
            
    coef = np.random.rand(2)*2-1
    a = np.round(coef*5, decimals=1) 
    target_pos = np.array([a[0], a[1]], dtype=np.float32)
    for _ in range(30*int(env.max_episode_length)):

        
        tx, ty, tz = float(target_pos[0]), float(target_pos[1]), 1.0

# build 横竖两段线的点阵，shape = (2*2, 3)
        pts = np.array([
            [tx - 0.1, ty,     tz],
            [tx + 0.1, ty,     tz],  # 横线两个端点
            [tx,      ty - 0.1, tz],
            [tx,      ty + 0.1, tz],  # 竖线两个端点
        ], dtype=np.float32)

        # 对应每个点的颜色
        cols = np.array([
            [1.0, 0.0, 0.0],  # 红
            [1.0, 0.0, 0.0],  
            [1.0, 0.0, 0.0],  
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)

        # 一共2段线
        count = 2

        # 对每个环境都画一次
        for env_handle in env.envs:
            env.gym.add_lines(viewer, env_handle, count, pts, cols)
            
        
        # —— 1. 处理 pygame 事件（否则无法更新键盘状态）True
        screen.fill((30, 30, 30))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_w:
                    vx += 0.5
                elif ev.key == pygame.K_s:
                    vx += -0.5
                elif ev.key == pygame.K_a:
                    vy += 0.5
                elif ev.key == pygame.K_d:
                    vy += -0.5
                elif ev.key == pygame.K_q:
                    yaw += 0.5
                elif ev.key == pygame.K_e:
                    yaw += -0.5
                elif ev.key == pygame.K_UP:
                    height += 0.05
                elif ev.key == pygame.K_DOWN:
                    height -= 0.05
            if vx>1:  vx =1
            if vx<-1: vx =-1
            if vy>1:  vy =1
            if vy<-1: vy =-1
            if height>1:  height =1
            if height<-1: height =-1      
            if yaw>3:  yaw =3
            if yaw<-3: yaw =-3
            
        current_pos = env.root_states[0, :2].cpu().numpy()  # 只取 x,y
        qx, qy, qz, qw  = env.root_states[0, 3:7].cpu().numpy()   # 方向四元数
        #cal_yaw
        siny_cosp = 2*(qw*qz + qx*qy)
        cosy_cosp = 1 - 2*(qy*qy + qz*qz)
        yaw_currrent = math.atan2(siny_cosp, cosy_cosp)
        
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        yaw_target = math.atan2(dy, dx)
        
        dtheta = yaw_target - yaw_currrent
        dtheta = (dtheta + math.pi) % (2*math.pi) - math.pi
        error = target_pos - current_pos        # 2-dim 向量
        dist  = np.linalg.norm(error)
        
        if abs(dtheta) > heading_thresh :
            vx_cmd = 0.0
            vy_cmd = 0.0
            yaw_cmd = max(-1.0, min(1.0, dtheta*5))
        else:
        #cal_vel
            if dist >arrive_tollerance:
                vx_cmd=  dist/(dist+1) * max_speed
                vy_cmd=0
            else:
                vx_cmd,vy_cmd = np.zeros(2, dtype=np.float32)  # 到了就停
        
        reset = False
        
        if dist <0.3 and dtheta <1.5:
            vx_cmd=0
            vy_cmd=0
            yaw_cmd=0
            print("arrive at target")
            reset = True
     
            
            
            
        vx = vx_cmd
        vy = vy_cmd
        yaw = yaw_cmd

        # —— 3. 更新命令
        env.commands[:,0] = torch.tensor(vx)
        env.commands[:,1] = torch.tensor(vy)
        env.commands[:,2] = torch.tensor(yaw)
        env.commands[:,4] = height

        # —— 4. 推策略、步进仿真
        num_lower_dof = env.num_lower_dof
        actions = policy(obs.detach())
        actions[:, num_lower_dof:] = 0
        obs, reward, _, done, *_ = env.step(actions.detach())
        
        if done[0] or reset:
            env.gym.clear_lines(viewer)
            coef = np.random.rand(2)*2-1
            a = np.round(coef*5, decimals=1) 
            target_pos = np.array([a[0], a[1]], dtype=np.float32)
        
        actual_vx = env.base_lin_vel[0,0].item()
        actual_vy = env.base_lin_vel[0,1].item()
        current_height = env.root_states[0,2].item()
        current_yaw = env.root_states[0,3].item()
        target_vx = vx
        target_vy = vy
        target_height = height
        target_yaw = yaw

        # env.root_states 是一个 (num_envs, 13) 的张量，
        # 前 3 个元素是 [x, y, z]
        current_pos_with_height = env.root_states[0, :3].cpu().numpy()    # 位置
        # current__quat  = env.root_states[0, 3:7].cpu().numpy()   # 方向四元数
        current_lin_v  = env.root_states[0, 7:10].cpu().numpy()  # 线速度
        current_ang_v  = env.root_states[0, 10:13].cpu().numpy() # 角速度
        


        # print("current_pos",current_pos)

        # print(obs.detach())
        # print("actions: ",actions)
        # print(actions.shape)
        # import pdb; pdb.set_trace()

        # —— 6. 渲染文本
        lines = [
            f"real vx: {actual_vx:.2f}  tar vx: {target_vx:.2f}",
            f"real vy: {actual_vy:.2f}  tar vy: {target_vy:.2f}",
            f"real hei: {current_height:.2f}  tar hei: {target_height:.2f}",
            f"real yaw: {current_yaw:.2f}  tar yaw: {target_yaw:.2f}",
            "↑↓:up/down  WASD:move  QE:orien"
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (200,200,200))
            screen.blit(surf, (5, 5 + i*20))
        # （可选）移动摄像机
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        # —— 5. 限制帧率
        pygame.display.flip()
        clock.tick(60)  # 60 FPS


if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args, x_vel=0., y_vel=0., yaw_vel=0., height=0.75)
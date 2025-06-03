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
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
import pinocchio as pin
from ik import solve_right_arm_ik
from isaacgym import gymapi


def draw_target_cross(env, viewer, target_pos):
    tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    pts = np.array([
        [tx - 0.1, ty,     tz],
        [tx + 0.1, ty,     tz],  # 横线两个端点
        [tx,      ty - 0.1, tz],
        [tx,      ty + 0.1, tz],  # 竖线两个端点
    ], dtype=np.float32)

    cols = np.array([
        [1.0, 0.0, 0.0],  # 红
        [1.0, 0.0, 0.0],  
        [1.0, 0.0, 0.0],  
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)

    count = 2
    for env_handle in env.envs:
        env.gym.add_lines(viewer, env_handle, count, pts, cols)


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
    
    waist_yaw_joint=torch.zeros(1)
    left_arm_joint=torch.zeros(7)
    right_arm_joint=torch.zeros(7)
    
    ##check shoulder pitch link
    actor_handle = env.actor_handles[0]
    rb_names = env.gym.get_actor_rigid_body_names(env.envs[0], actor_handle)
    rb_shoulder_index = rb_names.index("right_shoulder_pitch_link")
    
    #pygame
    pygame.init()
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.set_mode((1000,1000))  # 小窗口即可，仅为捕获键盘
    pygame.display.set_caption("HomieL 控制窗口")
    clock = pygame.time.Clock()
    
        
    
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
    
    q_init_7dof = np.zeros(7)
    for _ in range(30*int(env.max_episode_length)):

        
        gym = env.gym
        env_handle = env.envs[0]        # 这里只取第 0 个并行环境
        actor = actor_handle            # 你之前已拿到的 actor_handle
        rb_states = gym.get_actor_rigid_body_states(env_handle, actor, gymapi.STATE_ALL)

        shoulder_state = rb_states[rb_shoulder_index]

        pos_w = shoulder_state[0][0]  
        rot_w = shoulder_state[0][1]   
        p = np.array([pos_w[0],pos_w[1], pos_w[2]], dtype=np.float64)  # 位置向量
        qx, qy, qz, qw = rot_w
        quat = pin.Quaternion(float(qw), float(qx), float(qy), float(qz))

        R = quat.toRotationMatrix()  # 或者 quat.matrix()
        base_se3 = pin.SE3(R, p)  # 创建 SE3 对象
        # print(f"base_se3: {base_se3}")

        dof_states = gym.get_actor_dof_states(env_handle,
                                            actor_handle,
                                            gymapi.STATE_POS   # 只读位置也行，或者 STATE_ALL 也可以
                                            )
        all_pos = dof_states['pos']  
        right_arm_pos = all_pos[20:27] 
        # print("right_arm_pos:", np.round(right_arm_pos, 3))
        # q_init_7dof = right_arm_pos.cpu().numpy()  # 转为 numpy 数组


        target_pos=np.array([0.1, -0.25])
        pos_hei = 1.3
        target_pos1=np.array([target_pos[0], target_pos[1], pos_hei])

        draw_target_cross(env, viewer, target_pos1)

 
                
        
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
        
        reset = False
        
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
        
        if dist <0.3 and dtheta <1.5:
            vx_cmd=0
            vy_cmd=0
            yaw_cmd=0
            # print("arrive at target")
            reset = True
   
        vx = vx_cmd
        vy = vy_cmd
        yaw = yaw_cmd
        
   
        #======arm joint get========
        # target_pos1=np.array([1.55, -1.0, 1.3])

 
        convergence, q_sol, err_norm, *_ = solve_right_arm_ik(target_pos1, q_init_7dof, base_SE3=base_se3)

        if convergence:
            print(f"IK 收敛，err_norm={err_norm:.6f}")
            right_arm_joint= torch.tensor(q_sol, dtype=torch.float32)*4
        # print(f"IK 求解结果: {np.round(q_sol,3)}")  
        # q_init_7dof = q_sol.copy() 
        
        # env.commands[:,0] = torch.tensor(vx, dtype=torch.float32)
        # env.commands[:,1] = torch.tensor(vy, dtype=torch.float32)
        # env.commands[:,2] = torch.tensor(yaw, dtype=torch.float32)
        # env.commands[:,4] = torch.tensor(height, dtype=torch.float32)  # height
        env.commands[:,0] = torch.tensor(0.0)
        env.commands[:,1] = torch.tensor(0.0)
        env.commands[:,2] = torch.tensor(0.0)
        env.commands[:,4] = 0.75
        
        # —— 4. 推策略、步进仿真
        num_lower_dof = env.num_lower_dof
        actions = policy(obs.detach())

        left_arm_joint = left_arm_joint.view(1, -1)
        right_arm_joint = right_arm_joint.view(1, -1)
        waist_yaw_joint = waist_yaw_joint.view(1, -1)
        # right_arm_joint[:,:] =torch.tensor([-3.0,2.,.5,1.53,0.,0.0,0.])*4
        right_arm_joint[:,:] = torch.tensor([-0.18, -0.3, -0.3, -0.13, 0.00, -0.03, 0.0])*4
        actions = torch.cat([actions,waist_yaw_joint,left_arm_joint,right_arm_joint],dim=1)
        
        obs, reward, _, done, *_ = env.step(actions.detach()) #zai reset中也会调用        
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

    
        # import pdb; pdb.set_trace()

        # 渲染文本
        lines = [
            f"real vx: {actual_vx:.2f}  tar vx: {target_vx:.2f}",
            f"real vy: {actual_vy:.2f}  tar vy: {target_vy:.2f}",
            f"real hei: {current_height:.2f}  tar hei: {target_height:.2f}",
            f"real yaw: {current_yaw:.2f}  tar yaw: {target_yaw:.2f}",
            f"q_solve         : {np.array2string(q_sol*4, precision=2)}",
            f"right_arm_joint: {np.array2string(right_arm_joint.cpu().numpy(), precision=2)}",
            f"right_arm_pos: {np.array2string(right_arm_pos, precision=2)}",
            
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (200,200,200))
            screen.blit(surf, (5, 5 + i*20))

        # —— 5. 限制帧率
        pygame.display.flip()
        clock.tick(60)  # 60 FPS


if __name__ == '__main__':
    args = get_args()
    play(args, x_vel=0., y_vel=0., yaw_vel=0., height=0.75)

import math
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import onnxruntime as ort
import pygame
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch
# import pinocchio as pin
# from ik import solve_right_arm_ik
from isaacgym import gymapi
from isaacgym import gymapi
from isaacgym import gymtorch
import torch.nn.functional as F

def draw_target_cross(env, viewer, target_pos):
    tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    pts = np.array([
        [tx - 0.1, ty,     tz],
        [tx + 0.1, ty,     tz],  
        [tx,      ty - 0.1, tz],
        [tx,      ty + 0.1, tz],  
    ], dtype=np.float32)
    cols = np.array([
        [1.0, 0.0, 0.0],  
        [1.0, 0.0, 0.0],  
        [1.0, 0.0, 0.0],  
        [1.0, 0.0, 0.0],
    ], dtype=np.float32)
    count = 2
    for env_handle in env.envs:
        env.gym.add_lines(viewer, env_handle, count, pts, cols)

# IK func
def solve_right_arm_ik_jacobian(env, env_handle, actor_handle, 
                                 wrist_body_index, arm_joint_indices, 
                                 target_pos, q_init):

    device = env.device  # e.g. "cuda:0"
    rb_states_np = env.gym.get_actor_rigid_body_states(env_handle,actor_handle,gymapi.STATE_ALL)
    pos_w = rb_states_np[wrist_body_index][0][0]  # (px, py, pz)  
    cur_wrist_pos = np.array([pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    pos_err = target_pos - cur_wrist_pos  # numpy (3,)
    err_norm = np.linalg.norm(pos_err)
    actor_jacobian = env.gym.acquire_jacobian_tensor(env.sim, env.cfg.asset.name)  # 拿到当前全身雅可比，并提取末端对机械臂关节的 6×7 子雅可比
    env.gym.refresh_jacobian_tensors(env.sim)
    whole_jac = gymtorch.wrap_tensor(actor_jacobian) 
    j_eef = whole_jac[0, wrist_body_index, :6, arm_joint_indices]  # [6,7]，Tensorprin
    dpos = torch.zeros(6, device=device)    #构建 dpose = [pos_err; zeros(3)]，并转成 Tensor (6,1)
    dpos[0:3] = torch.from_numpy(pos_err).to(device).float() 
    dpose = dpos.unsqueeze(-1)  # [6,1]     # orientation 误差置 0（不关心姿态）

    #  Δq = J^T (J J^T + λ² I)^(-1) dpose
    J = j_eef  # [6,7]
    JJT = J @ J.T  # [6,6]
    lambda_sq = (0.05 ** 2)
    reg = torch.eye(6, device=device) * lambda_sq  # [6,6]
    inv_term = torch.inverse(JJT + reg)  # [6,6]
    dq = (J.T @ inv_term @ dpose).squeeze(-1)  # [7]
    q_new = q_init + dq.cpu().numpy()*1  # numpy (7,)
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
    
    #init upper body joint
    waist_yaw_joint=torch.zeros(1)
    left_arm_joint=torch.zeros(7)
    right_arm_joint=torch.zeros(7)
    
    ##check shoulder pitch link
    actor_handle = env.actor_handles[0]
    rb_names = env.gym.get_actor_rigid_body_names(env.envs[0], actor_handle)#link names
    rb_shoulder_index = rb_names.index("right_shoulder_pitch_link")
    rb_wrist_index = rb_names.index("right_wrist_yaw_link") 
  
    #pygame initialize
    pygame.init()
    font = pygame.font.SysFont(None, 24)
    screen = pygame.display.set_mode((300,300))  # 小窗口即可，仅为捕获键盘
    pygame.display.set_caption("HomieL 控制窗口")
    clock = pygame.time.Clock()
    
    # env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    cpu_inds = torch.arange(env.num_envs, dtype=torch.int32)
    env.reset_idx(cpu_inds)
    vx = vy = yaw = 0.0
    
    arrive_tollerance  = 0.3 #pos tollerance
    heading_thresh = 0.1 #yaw tollerance
    max_speed=1 # max linear speed
    arm_end_target_err = 0.1  
    
    viewer = env.viewer
    env.gym.clear_lines(viewer)

    np.random.seed(123)
    coef_height = np.random.rand(1)+0.3
    tar_height= coef_height[0]  # z

    coef = np.random.rand(2)*2-1
    a = np.round(coef*1.5, decimals=1) 
    target_pos = np.array([a[0], a[1]], dtype=np.float32)
    q_init_7dof = np.zeros(7)
    arm_joint_indices = torch.arange(26, 33, dtype=torch.long, device=env.device)

    i=0
    for _ in range(30*int(env.max_episode_length)):
        gym = env.gym
        env_handle = env.envs[0]        # 这里只取第 0 个并行环境
        actor = actor_handle            # 你之前已拿到的 actor_handle
        rb_states = gym.get_actor_rigid_body_states(env_handle, actor, gymapi.STATE_ALL)
        shoulder_state = rb_states[rb_shoulder_index]
        pos_w = shoulder_state[0][0]  
        rot_w = shoulder_state[0][1]   

        dof_states = gym.get_actor_dof_states(env_handle,actor_handle,gymapi.STATE_POS)
        all_pos = dof_states['pos']  
        right_arm_pos = all_pos[20:27] 
        q_init_7dof = right_arm_pos

        tar_height= coef_height[0]
        target_pos1=np.array([target_pos[0], target_pos[1], tar_height])
        draw_target_cross(env, viewer, target_pos1) #draw target cross

        # Calculate and display FPS
        fps = clock.get_fps()

        screen.fill((30, 30, 30))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_w:
                    vx = min(vx + 0.5, 1)
                elif ev.key == pygame.K_s:
                    vx = max(vx - 0.5, -1)
                elif ev.key == pygame.K_a:
                    vy = min(vy + 0.5, 1)
                elif ev.key == pygame.K_d:
                    vy = max(vy - 0.5, -1)
                elif ev.key == pygame.K_q:
                    yaw = min(yaw + 0.5, 3)
                elif ev.key == pygame.K_e:
                    yaw = max(yaw - 0.5, -3)
                elif ev.key == pygame.K_UP:
                    height = min(height + 0.1, 1)
                elif ev.key == pygame.K_DOWN:
                    height = max(height - 0.1, -1)
            
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
            yaw_cmd = max(-1.0, min(1.0, dtheta*2))
            if dist >arrive_tollerance:
                vx_cmd=  dist/(dist+1) * max_speed
                vy_cmd=0
            else:
                vx_cmd,vy_cmd = np.zeros(2, dtype=np.float32)  # 到了就停
        else:
            vx_cmd=  dist/(dist+1) * max_speed
            vy_cmd=0
        
        if dist <0.3 :
            vx_cmd=0
            vy_cmd=0
            if dtheta <0.2:
                height = max(0.4, min(0.9, target_pos1[2]))
                yaw_cmd=0
                q_new, err_norm = solve_right_arm_ik_jacobian( env, env_handle, actor_handle, rb_wrist_index, arm_joint_indices, target_pos1, q_init_7dof) #  arm joint get
                right_arm_joint = torch.tensor(q_new, dtype=torch.float32)*4 
                if err_norm < 0.1:  
                    reset = True     
                    print("arrive at target") 
            else:
                yaw_cmd = max(-1.0, min(1.0, dtheta*2))
                
        vx= vx_cmd
        vy = vy_cmd
        yaw = yaw_cmd
        arm_end_target_err = err_norm if 'err_norm' in locals() else 0.0
          
        env.commands[:,0] = torch.tensor(vx, dtype=torch.float32)
        env.commands[:,1] = torch.tensor(vy, dtype=torch.float32)
        env.commands[:,2] = torch.tensor(yaw, dtype=torch.float32)
        env.commands[:,4] = torch.tensor(height, dtype=torch.float32)  # height

        num_lower_dof = env.num_lower_dof
        actions = policy(obs.detach())

        left_arm_joint = left_arm_joint.view(1, -1)
        right_arm_joint = right_arm_joint.view(1, -1)
        waist_yaw_joint = waist_yaw_joint.view(1, -1)
        actions = torch.cat([actions,waist_yaw_joint,left_arm_joint,right_arm_joint],dim=1)
        
        obs, reward, _, done, *_ = env.step(actions.detach()) #reset中也会调用        
        if done[0] or reset:
            env.gym.clear_lines(viewer)
            coef = np.random.rand(2)*2-1
            a = np.round(coef*5, decimals=1) 
            target_pos = np.array([a[0], a[1]], dtype=np.float32)
            right_arm_joint= torch.zeros(7, dtype=torch.float32)  # 重置右臂关节角度
            height = 0.75 # reset height

            coef_height = np.random.rand(1)+0.3
            tar_height= coef_height[0]  # z

            coef = np.random.rand(2)*2-1
            a = np.round(coef*1.5, decimals=1) 
            target_pos = np.array([a[0], a[1]], dtype=np.float32)
        
        actual_vx = env.base_lin_vel[0,0].item()
        actual_vy = env.base_lin_vel[0,1].item()
        current_height = env.root_states[0,2].item()
        current_yaw = env.root_states[0,3].item()

        target_vx = vx
        target_vy = vy
        target_height = height
        target_yaw = yaw
        i=i+1
        if i % 60 == 0:
            print("fps: ", round(fps,1))
        # import pdb; pdb.set_trace()

        # 渲染文本
        lines = [
            f"real vx: {actual_vx:.2f}  tar vx: {target_vx:.2f}",
            f"real vy: {actual_vy:.2f}  tar vy: {target_vy:.2f}",
            f"real hei: {current_height:.2f}  tar hei: {tar_height:.2f}",
            f"real yaw: {current_yaw:.2f}  tar yaw: {target_yaw:.2f}",
            # f"q_new         : {np.array2string(q_new, precision=2)}",
            f"right_arm_joint: {np.array2string(right_arm_joint.cpu().numpy(), precision=2)}",
            f"right_arm_pos: {np.array2string(right_arm_pos, precision=2)}",
            f"arm_end_target_err: {arm_end_target_err:.2f}",
            f"fps: {fps:.2f}",
            
        ]
        for i, text in enumerate(lines):
            surf = font.render(text, True, (255,255,255))
            screen.blit(surf, (5, 5 + i*20))
        pygame.display.flip()
        clock.tick(60)  


if __name__ == '__main__':
    args = get_args()
    play(args, x_vel=0., y_vel=0., yaw_vel=0., height=0.75)
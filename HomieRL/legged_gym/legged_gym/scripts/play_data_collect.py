
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
from isaacgym import gymapi
from isaacgym import gymtorch
import torch.nn.functional as F

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

# *** added start: 定义基于雅可比的右臂 IK 函数
def solve_right_arm_ik_jacobian(env, env_handle, actor_handle, 
                                 wrist_body_index, arm_joint_indices, 
                                 target_pos, q_init):

    device = env.device  # e.g. "cuda:0"

    # 1. 拿到当前所有刚体状态

    rb_states_np = env.gym.get_actor_rigid_body_states(env_handle,
                                                       actor_handle,
                                                       gymapi.STATE_ALL)

    # 取第 wrist_body_index 个刚体的 (x,y,z)
    pos_w = rb_states_np[wrist_body_index][0][0]  # (px, py, pz)  # 这已经是 Python 列表或 NumPy 数组

    cur_wrist_pos = np.array([pos_w[0], pos_w[1], pos_w[2]], dtype=np.float64)
    # 2. 计算位置误差
    pos_err = target_pos - cur_wrist_pos  # numpy (3,)
    err_norm = np.linalg.norm(pos_err)

    # 3. 拿到当前全身雅可比，并提取末端对机械臂关节的 6×7 子雅可比
    # actor_jacobian = env.gym.acquire_jacobian_tensor(env.sim,env.cfg.asset.name)
    actor_jacobian = env.gym.acquire_jacobian_tensor(env.sim, env.cfg.asset.name)
    env.gym.refresh_jacobian_tensors(env.sim)
    whole_jac = gymtorch.wrap_tensor(actor_jacobian) 
    # whole_jac = gymtorch.wrap_tensor(actor_jacobian) 
    # print(f"whole_jac: {whole_jac}")  # [num_envs, num_bodies, 6, num_dofs]
    # print(f"whole_jac.shape: {whole_jac.shape}")  # [num_envs, num_bodies, 6, num_dofs]
    # [num_envs, num_bodies, 6, num_dofs]
    j_eef = whole_jac[0, wrist_body_index, :6, arm_joint_indices]  # [6,7]，Tensorprint
    # print(f"j_eef: {j_eef}")

    # 4. 构建 dpose = [pos_err; zeros(3)]，并转成 Tensor (6,1)
    dpos = torch.zeros(6, device=device)
    dpos[0:3] = torch.from_numpy(pos_err).to(device).float()
    # orientation 误差置 0（不关心姿态）
    dpose = dpos.unsqueeze(-1)  # [6,1]

    # 5. 计算阻尼最小二乘解： Δq = J^T (J J^T + λ² I)^(-1) dpose
    J = j_eef  # [6,7]
    JJT = J @ J.T  # [6,6]
    lambda_sq = (0.05 ** 2)
    reg = torch.eye(6, device=device) * lambda_sq  # [6,6]
    inv_term = torch.inverse(JJT + reg)  # [6,6]
    dq = (J.T @ inv_term @ dpose).squeeze(-1)  # [7]

    # 6. 更新关节角度
    q_new = q_init + dq.cpu().numpy()*0.2  # numpy (7,)

    return q_new, err_norm
# *** added end

def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 2
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
    idx=0

    for n in rb_names:
        print(f"{idx}: {n}")
        idx += 1
    rb_shoulder_index = rb_names.index("right_shoulder_pitch_link")
    # print("rb_shoulder_index:", rb_shoulder_index)
    
    rb_wrist_index = rb_names.index("right_wrist_yaw_link") 
    # rb_wrist_index = rb_names.index("right_hand_palm_link") 
    # print("rb_wrist_index:", rb_wrist_index)
    
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
    coef_height = np.random.rand(1)

    a = np.round(coef*1.5, decimals=1) 
    
    target_pos = np.array([a[0], a[1]], dtype=np.float32)
    tar_height= coef_height[0]  # 固定高度
    q_init_7dof = np.zeros(7)
   
    
    arm_joint_indices = torch.arange(26, 33, dtype=torch.long, device=env.device)
    
    print("arm_joint_indices:", arm_joint_indices)
    
    arm_end_target_err = 0.0  
    b=0.3
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
        q_init_7dof = right_arm_pos

        tar_height= coef_height[0]+ b  # 固定高度

        target_pos1=np.array([target_pos[0], target_pos[1], tar_height])

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
                    b += 0.7
                elif ev.key == pygame.K_DOWN:
                    b -= 0.7
             
            if vx>1:  vx =1
            if vx<-1: vx =-1
            if vy>1:  vy =1
            if vy<-1: vy =-1
            if height>1:  height =1
            if height<-1: height =-1      
            if yaw>3:  yaw =3
            if yaw<-3: yaw =-3
            if b>1:  b =1
            if b<0.3: b =0.3
            
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
            # vx_cmd = 0.0
            # vy_cmd = 0.0
            yaw_cmd = max(-1.0, min(1.0, dtheta*2))
        # else:
            #cal_vel
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
                # ======new arm joint get========
                q_new, err_norm = solve_right_arm_ik_jacobian(
                    env, env_handle, actor_handle,
                    rb_wrist_index, arm_joint_indices,
                    target_pos1, q_init_7dof
                )
                right_arm_joint = torch.tensor(q_new, dtype=torch.float32)*4 # *** added
                if err_norm < 0.1:  # 如果误差小于阈值，认为到达目标
                    reset = True     
            else:
                yaw_cmd = max(-1.0, min(1.0, dtheta*2))
                
            # print("arrive at target")
        vx= vx_cmd
        # vx = max(-0.7, min(0.7, vx_cmd))
        vy = vy_cmd
        yaw = yaw_cmd
        arm_end_target_err = err_norm if 'err_norm' in locals() else 0.0
        
        #======arm joint get========
        # target_pos1=np.array([1.55, -1.0, 1.3])

 
        # convergence, q_sol, err_norm, *_ = solve_right_arm_ik(target_pos1, q_init_7dof, base_SE3=base_se3)

        # if convergence:
        #     print(f"IK 收敛，err_norm={err_norm:.6f}")
        #     right_arm_joint= torch.tensor(q_sol, dtype=torch.float32)*4
        # print(f"IK 求解结果: {np.round(q_sol,3)}")  
        # q_init_7dof = q_sol.copy() 
         #======arm joint get end========
         
         
        #======new arm joint get========
         
        # q_new, err_norm = solve_right_arm_ik_jacobian(
        #     env, env_handle, actor_handle,
        #     rb_wrist_index, arm_joint_indices,
        #     target_pos1, q_init_7dof
        # )
        # right_arm_joint = torch.tensor(q_new, dtype=torch.float32)*4 # *** added
         
         
         
         
         
         
         
         
         
         
        env.commands[:,0] = torch.tensor(vx, dtype=torch.float32)
        env.commands[:,1] = torch.tensor(vy, dtype=torch.float32)
        env.commands[:,2] = torch.tensor(yaw, dtype=torch.float32)
        env.commands[:,4] = torch.tensor(height, dtype=torch.float32)  # height
        # env.commands[:,0] = torch.tensor(0.0)
        # env.commands[:,1] = torch.tensor(0.0)
        # env.commands[:,2] = torch.tensor(0.0)
        # env.commands[:,4] = 0.75
        
        # —— 4. 推策略、步进仿真
        num_lower_dof = env.num_lower_dof
        actions = policy(obs.detach())

        left_arm_joint = left_arm_joint.view(1, -1)
        right_arm_joint = right_arm_joint.view(1, -1)
        waist_yaw_joint = waist_yaw_joint.view(1, -1)
        # right_arm_joint[:,:] =torch.tensor([-3.0,2.,.5,1.53,0.,0.0,0.])*4
        # right_arm_joint[:,:] = torch.tensor([-0.18, -0.3, -0.3, -0.13, 0.00, -0.03, 0.0])*4
        actions = torch.cat([actions,waist_yaw_joint,left_arm_joint,right_arm_joint],dim=1)
        
        obs, reward, _, done, *_ = env.step(actions.detach()) #zai reset中也会调用        
        if done[0] or reset:
            env.gym.clear_lines(viewer)
            coef = np.random.rand(2)*2-1
            a = np.round(coef*5, decimals=1) 
            target_pos = np.array([a[0], a[1]], dtype=np.float32)
            right_arm_joint= torch.zeros(7, dtype=torch.float32)  # 重置右臂关节角度
            height = 0.75
        # print("env.cfg.asset.name:", env.cfg.asset.name)

        # print(dir(env))
        # print(f"whole_jac: {whole_jac}")  # [num_envs, num_bodies, 6, num_dofs]
        # print(f"whole_jac.shape: {whole_jac.shape}")  # [num_envs, num_bodies, 6, num_dofs]
        
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
            f"real hei: {current_height:.2f}  tar hei: {tar_height:.2f}",
            f"real yaw: {current_yaw:.2f}  tar yaw: {target_yaw:.2f}",
            # f"q_new         : {np.array2string(q_new, precision=2)}",
            f"right_arm_joint: {np.array2string(right_arm_joint.cpu().numpy(), precision=2)}",
            f"right_arm_pos: {np.array2string(right_arm_pos, precision=2)}",
            f"arm_end_target_err: {arm_end_target_err:.2f}",
            
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
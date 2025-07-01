from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
import torch


def play(args, x_vel=0.0, y_vel=0.0, yaw_vel=0.0, height=0.74):

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
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
    env.action_curriculum_ratio = 1.0
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device) # Use this to load from trained pt file
    print("policy:",policy)
    # 
    B=env.num_envs
    waist_yaw_joint = torch.zeros(env.num_envs, 1, device=env.device)  # (B,1)
    left_arm_joint = torch.zeros(env.num_envs, 7, device=env.device)  # (B,7)
    right_arm_joint = torch.zeros(env.num_envs, 7, device=env.device)  # (B,7)

    env.reset_idx(torch.arange(env.num_envs).to("cuda:0"))
    for _ in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        print("actions: ", actions)
        print("actions:",actions.shape)
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        env.commands[:, 4] = height

        left_arm_joint = left_arm_joint.view(B, -1)
        right_arm_joint = right_arm_joint.view(B, -1)
        waist_yaw_joint = waist_yaw_joint.view(B, -1)
        actions = torch.cat([actions, waist_yaw_joint, left_arm_joint, right_arm_joint], dim=1)
        obs, _, _, _, _, _, _ = env.step(actions.detach())


if __name__ == '__main__':
    args = get_args()
    play(args, x_vel=0., y_vel=0., yaw_vel=0., height=0.54)
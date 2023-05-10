import argparse
import numpy as np
import gym

import os
# from mpi4py import MPI
from envs import register_envs
from envs.multi_world_wrapper import NoisyAction
from rl_modules.actionablemodel_agent import ActionableModel
from rl_modules.ddpg_agent import DDPG
from rl_modules.gofar_agent import GoFAR
from rl_modules.gcsl_agent import GCSL

import random
import torch
# import wandb

from gym.wrappers import ClipAction

import roboverse as rv

from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import InsertImageEnv
from rlkit.envs.contextual.goal_conditioned import PresampledPathDistribution
from rlkit.experimental.kuanfang.envs.contextual_env import ContextualEnv
from rlkit.experimental.kuanfang.envs.reward_fns import GoalReachingRewardFn
from rlkit.experimental.kuanfang.envs.drawer_pnp_push_commands import drawer_pnp_push_commands

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env', type=str, default='FetchReach', help='the environment name')
    parser.add_argument('--n-epochs', type=int, default=100, help='the number of epochs to train the agent')
    parser.add_argument('--n-cycles', type=int, default=20, help='the times to collect samples per epoch')
    parser.add_argument('--n-batches', type=int, default=20, help='the times to update the network')
    parser.add_argument('--save-interval', type=int, default=5, help='the interval that save the trajectory')
    parser.add_argument('--num-workers', type=int, default=1, help='the number of cpus to collect samples')
    parser.add_argument('--replay-strategy', type=str, default='future', help='the HER strategy')
    parser.add_argument('--clip-return', type=float, default=50, help='if clip the returns')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')
    parser.add_argument('--random-eps', type=float, default=0.3, help='random eps')
    parser.add_argument('--buffer-size', type=int, default=int(2e6), help='the size of the buffer')
    parser.add_argument('--replay-k', type=int, default=4, help='ratio to be replace')
    parser.add_argument('--clip-obs', type=float, default=200, help='the clip ratio')
    parser.add_argument('--batch-size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=1, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.001, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.001, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.95, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')
    parser.add_argument('--clip-range', type=float, default=5, help='the clip range')
    parser.add_argument('--demo-length', type=int, default=20, help='the demo length')
    parser.add_argument('--cuda', default=True, type=boolean, help='if use gpu do the acceleration')
    parser.add_argument('--device', default=0, type=int, help='gpu device number')
    parser.add_argument('--num-rollouts-per-mpi', type=int, default=1, help='the rollouts per mpi')

    # hyperparameters that need to be changed
    parser.add_argument('--eval', default=True, type=boolean)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--method', default='gofar', type=str)
    parser.add_argument('--f', default='chi', type=str)
    parser.add_argument('--online', default=False, type=boolean)

    parser.add_argument('--noise', default=False, type=boolean, help='add noise to action')
    parser.add_argument('--noise-eps', type=float, default=1.0, help='noise eps')

    parser.add_argument('--relabel', default=True, type=boolean)
    parser.add_argument('--relabel_percent', default=0.5, type=float)

    parser.add_argument('--reward_type', default='binary', type=str)
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--disc_iter', type=int, default=20)
    parser.add_argument('--disc_lambda', type=float, default=0.01)

    parser.add_argument('--expert_percent', type=float, default=0.1, help='the expert coefficient')
    parser.add_argument('--random_percent', type=float, default=0.9, help='the random coefficient')

    # sawyer env6
    parser.add_argument('--presampled_goal_dir', type=str,
                        default='~/offline_c_learning/dataset/env6_td_pnp_push/goals_early_stop')
    parser.add_argument('--goal_timeoutk', type=int, default=-1)
    parser.add_argument('--eval_seed', type=int, default=31)
    parser.add_argument('--env_obs_img_dim', type=int, default=196)
    parser.add_argument('--obs_img_dim', type=int, default=48)

    args = parser.parse_args()

    return args

def get_env_params(env):
    obs = env.reset()
    # close the environment
    if isinstance(env, ContextualEnv):
        params = {'obs': obs['image_observation'].shape[0],
                  'goal': obs['image_desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  'action_space': env.action_space}
        params['max_timesteps'] = env._max_episode_steps
    else:
        params = {'obs': obs['observation'].shape[0],
                  'goal': obs['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  'action_space': env.action_space}
        params['max_timesteps'] = env._max_episode_steps
    return params

def get_full_envname(name):
    dic = {
        'FetchReach':'FetchReach-v1',
        'FetchPush': 'FetchPush-v1',
        'FetchSlide': 'FetchSlide-v1',
        'FetchPick': 'FetchPickAndPlace-v1',
        'HandReach':'HandReach-v0',
        'DClawTurn': 'DClawTurn-v0',
    }
    if name in dic.keys():
        return dic[name]
    else:
        return name

def get_method_params(args):
    if args.online:
        args.n_batches = 40
        args.n_cycles = 50

    if args.method == 'ddpg' or args.method == 'td3bc':
        args.lr_actor = 0.001
        args.lr_critic = 0.001
    elif args.method == 'goaldice' or 'gcsl' in args.method or 'gcbc' in args.method:
        args.lr_actor = 5e-4
        args.lr_critic = 5e-4

    if 'gcsl' in args.method or 'AM' in args.method:
        args.relabel_percent = 1.0
    if 'gcbc' in args.method:
        args.relabel = False
    if 'gofar' in args.method:
        args.relabel = False
        args.reward_type = 'disc'

    if args.env == 'DClawTurn' or args.env == 'FetchReach':
        args.expert_percent = 0.


def make_sawyer_env6(args):
    presampled_goal_dir = os.path.expanduser(args.presampled_goal_dir)
    presampled_goals_path = os.path.join(
        presampled_goal_dir,
        'td_pnp_push_scripted_goals_timeoutk{}_seed{}.pkl'.format(
            args.goal_timeoutk, args.eval_seed))

    raw_env = rv.make(
        "SawyerRigAffordances-v6",
        gui=False,
        expl=True,
        env_obs_img_dim=args.env_obs_img_dim,
        obs_img_dim=args.obs_img_dim,
        test_env=True,
        test_env_command=drawer_pnp_push_commands[args.eval_seed],
        downsample=True,
    )

    state_env = ClipAction(raw_env)
    renderer = EnvRenderer(
        create_image_format='HWC',
        output_image_format='CWH',
        flatten_image=True,
        width=args.obs_img_dim,
        height=args.obs_img_dim)
    env = InsertImageEnv(
        state_env,
        renderer=renderer)

    diagnostics = env.get_contextual_diagnostics
    context_distribution = PresampledPathDistribution(
        presampled_goals_path,
        None,
        initialize_encodings=False)
    reward_fn = GoalReachingRewardFn(
        state_env.env,
        use_pretrained_reward_classifier_path=False,
        obs_type='state',
        reward_type='highlevel',
        epsilon=3.0,
        terminate_episode=0
    )

    env = ContextualEnv(
        env,
        context_distribution=context_distribution,
        reward_fn=reward_fn,
        observation_key='latent_observation',
        contextual_diagnostics_fns=[diagnostics] if not isinstance(
            diagnostics, list) else diagnostics,
    )
    env._max_episode_steps = 74

    return env


def launch(args):
    get_method_params(args)
    args.use_disc = True if args.reward_type =='disc' else False

    args.env_id = get_full_envname(args.env)
    # load environment
    register_envs()

    if args.env == "SawyerEnv6":
        env = make_sawyer_env6(args)
    else:
        env = gym.make(args.env_id)
    env_id = args.env_id

    # stochastic environment setting
    if args.noise:
        env = NoisyAction(env, noise_eps=args.noise_eps)
        env._max_episode_steps = 50

    if args.relabel == False:
        args.relabel_percent = 0.
    relabel_tag = f'relabel{args.relabel_percent}'

    reward_tag = args.reward_type
    if args.reward_type == 'disc':
        reward_tag = f'{args.disc_iter}disc{args.disc_lambda}'
    elif args.reward_type == 'binary':
        reward_tag = f'{args.reward_type}{args.threshold}'
    run_name = f'{args.env_id}-{args.expert_percent}-{args.random_percent}-{args.method}-{reward_tag}-{relabel_tag}-{args.seed}'

    if args.noise:
        run_name = f'{args.env_id}-noise{args.noise_eps}-{args.expert_percent}-{args.random_percent}-{args.method}-{reward_tag}-{relabel_tag}-{args.seed}'

    args.run_name = run_name

    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     wandb.init(project='gofar', name=run_name,
    #     group=args.env, config=args)

    # set random seeds for reproduce
    # env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # get the environment parameters
    env_params = get_env_params(env)

    # create agent
    if args.method == 'ddpg':
        trainer = DDPG(args, env, env_params)
    elif args.method == 'gofar':
        trainer = GoFAR(args, env, env_params)
    elif 'gcsl' in args.method or 'gcbc' in args.method:
        trainer = GCSL(args, env, env_params)
    elif 'action' in args.method or 'AM' in args.method:
        trainer = ActionableModel(args, env, env_params)
    else:
        raise NotImplementedError
    print(run_name)

    # do offline goal-conditioned rl 
    trainer.learn(evaluate_agent=args.eval)

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    # get the params
    launch(args)

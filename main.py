from util import Memory, PriorityMemory
from mujoco_models import Actor, Critic
import mujoco_py
import gym
from tensorboardX import SummaryWriter
from itertools import count
from trainer import SILCRTrainer
import torch
import argparse
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    env_dict = {0:'Swimmer-v2', 1:'Hopper-v2', 2:'HalfCheetah-v2',
                3:'Walker2d-v2', 4:'Humanoid-v2'}

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_id', type=int, default=4, help='env id from 0 to 4')
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=int(2e6))
    parser.add_argument('--priority_size', type=int, default=50000)

    args = parser.parse_args()

    if args.env_id not in env_dict.keys():
        print('no such env id')
        exit()

    env = gym.make(env_dict[args.env_id])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(state_dim, action_dim).to(device)
    critic_1 = Critic(state_dim, action_dim).to(device)
    critic_2 = Critic(state_dim, action_dim).to(device)

    critic_1_target = Critic(state_dim, action_dim).to(device)
    critic_2_target = Critic(state_dim, action_dim).to(device)

    online_replay = Memory(int(1e6))
    expert_replay = PriorityMemory(args.priority_size)
    writer = SummaryWriter('logs_sil_' + env_dict[args.env_id] + '_run_' + str(args.run_id))
    max_steps = args.max_steps

    learner = SILCRTrainer(
        actor, 
        critic_1, 
        critic_2, 
        critic_1_target, 
        critic_2_target, 
        online_replay, 
        expert_replay, 
        -float(action_dim),
        writer,
        actor_learn_rate=3e-4,
        critic_learn_rate=3e-4,
        alpha_learn_rate=3e-4
        )

    batch_size = 128

    learn_steps = 0

    cur_evaluation_step = 0

    evaluation_time_interval = int(max_steps / 1000)

    evaluate = False

    rewards_log = []

    for epoch in count():
        episode_reward = 0.
        state = env.reset()
        states = []
        next_states = []
        actions = []
        dones = []
        for time_steps in range(1000):
            action, log_prob, _ = actor.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            online_replay.add((state, next_state, action, 0., done))
            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            dones.append(done)
            state = next_state
            learn_steps = learner.learn(batch_size)
            if learn_steps >= cur_evaluation_step * evaluation_time_interval:
                cur_evaluation_step += 1
                evaluate = True
            if done:
                break
        for i in range(len(states)):
            expert_replay.add((states[i], next_states[i], actions[i], 1., dones[i]), episode_reward)
        writer.add_scalar('episode reward', episode_reward, learn_steps)
        if evaluate:
            evaluate_episode_reward = 0
            state = env.reset()
            for time_steps in range(1000):
                _, log_prob, action = actor.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                evaluate_episode_reward += reward
                state = next_state
                if done:
                    break
            writer.add_scalar('evaluate episode reward', evaluate_episode_reward, 
                                (cur_evaluation_step-1) * evaluation_time_interval)
            rewards_log.append((evaluate_episode_reward, (cur_evaluation_step-1) * evaluation_time_interval))
            print('>>>>>>> evaluate episode reward : {}<<<<<<<'.format(evaluate_episode_reward))
            evaluate = False
        if epoch % 1 == 0:
            print('Epoch:{}, episode reward is {}, lowest reward in expert is {}'.format(epoch, episode_reward, expert_replay.get_lowest_rewards()))
	
        if epoch % 100 == 0:
            learner.save_model('models_' + env_dict[args.env_id] + '_run_' + str(args.run_id))
            print('model saved!')
        if learn_steps >= max_steps:
            break
    print('-----------------training over-------------------------')
    learner.save_model('models_' + env_dict[args.env_id] + '_run_' + str(args.run_id))
    rewards_log = np.asarray(rewards_log)
    np.save('rewards_log_' + env_dict[args.env_id] + '_run_' + str(args.run_id), rewards_log)


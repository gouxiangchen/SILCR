import torch.nn.functional as F
import torch
import numpy as np
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


class SILCRTrainer:
    def __init__(self, actor, critic_1, critic_2, critic_1_target, critic_2_target, online_replay_memory, 
                 expert_replay_memory, target_entropy, logger_writer, begin_train=10240, tau=0.05, actor_learn_rate=3e-5,
                 critic_learn_rate=3e-5, alpha_learn_rate=3e-4, gamma=0.99, reload=False, reload_path=''):
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.critic_1_target = critic_1_target
        self.critic_2_target = critic_2_target

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_learn_rate)
        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(), lr=critic_learn_rate)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(), lr=critic_learn_rate)


        if reload:
            self.actor.load_state_dict(torch.load(reload_path + '/actor.para'))
            self.critic_1.load_state_dict(torch.load(reload_path + '/critic_1.para'))
            self.critic_2.load_state_dict(torch.load(reload_path + '/critic_2.para'))
            self.critic_1_target.load_state_dict(torch.load(reload_path + '/critic_1_target.para'))
            self.critic_2_target.load_state_dict(torch.load(reload_path + '/critic_2_target.para'))

            log_alpha_np = np.load(reload_path + '/log_alpha.npy')
            self.log_alpha = torch.FloatTensor(log_alpha_np).to(device)

        else:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        self.log_alpha.requires_grad_()
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_learn_rate)
        
        self.memory_replay = online_replay_memory
        self.expert_replay = expert_replay_memory

        self.writer = logger_writer
        self.begin_train = begin_train

        self.steps = 0

        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

    def save_model(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self.actor.state_dict(), path + '/actor.para')
        torch.save(self.critic_1.state_dict(), path + '/critic_1.para')
        torch.save(self.critic_2.state_dict(), path + '/critic_2.para')
        torch.save(self.critic_1_target.state_dict(), path + '/critic_1_target.para')
        torch.save(self.critic_2_target.state_dict(), path + '/critic_2_target.para')
        np.save(path + '/log_alpha.npy', self.log_alpha.detach().cpu().numpy())

    def learn(self, batch_size):
        if self.memory_replay.size() > self.begin_train:
            self.steps += 1

            alpha = self.log_alpha.exp().detach()

            experiences = self.memory_replay.sample(batch_size // 2, False)
            online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = zip(
                *experiences)

            online_batch_state = torch.FloatTensor(online_batch_state).to(device)
            online_batch_next_state = torch.FloatTensor(online_batch_next_state).to(device)
            online_batch_action = torch.FloatTensor(online_batch_action).to(device)
            online_batch_reward = torch.FloatTensor(online_batch_reward).unsqueeze(1).to(device)
            online_batch_done = torch.FloatTensor(online_batch_done).unsqueeze(1).to(device)

            experiences = self.expert_replay.sample(batch_size // 2, False)
            expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = zip(
                *experiences)

            expert_batch_state = torch.FloatTensor(expert_batch_state).to(device)
            expert_batch_next_state = torch.FloatTensor(expert_batch_next_state).to(device)
            expert_batch_action = torch.FloatTensor(expert_batch_action).to(device)
            expert_batch_reward = torch.FloatTensor(expert_batch_reward).unsqueeze(1).to(device)
            expert_batch_done = torch.FloatTensor(expert_batch_done).unsqueeze(1).to(device)

            if len(expert_batch_action.shape) == 1:
                expert_batch_action = expert_batch_action.unsqueeze(1)
                online_batch_action = online_batch_action.unsqueeze(1)

            batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
            batch_next_state = torch.cat([online_batch_next_state, expert_batch_next_state], dim=0)
            batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
            batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
            batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.actor(batch_next_state)
                qf1_next_target = self.critic_1_target(batch_next_state, next_state_action)
                qf2_next_target = self.critic_2_target(batch_next_state, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = batch_reward + self.gamma * (1 - batch_done) * (min_qf_next_target)

            critic_1_loss = F.mse_loss(self.critic_1(batch_state, batch_action), next_q_value)
            self.critic_1_optim.zero_grad()
            critic_1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), 40.)
            self.critic_1_optim.step()

            critic_2_loss = F.mse_loss(self.critic_2(batch_state, batch_action), next_q_value)
            self.critic_2_optim.zero_grad()
            critic_2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), 40.)
            self.critic_2_optim.step()

            pi, log_pi, _ = self.actor(batch_state)
            qf1_pi = self.critic_1(batch_state, pi)
            qf2_pi = self.critic_2(batch_state, pi)

            actor_loss = ((alpha * log_pi) - torch.min(qf1_pi, qf2_pi)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40.)
            self.actor_optim.step()

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.writer.add_scalar('alpha loss', alpha_loss.item(), self.steps)
            self.writer.add_scalar('alpha', self.log_alpha.exp().item(), self.steps)

            self.writer.add_scalar('critic 1 loss', critic_1_loss.item(), self.steps)
            self.writer.add_scalar('critic 2 loss', critic_2_loss.item(), self.steps)
            self.writer.add_scalar('actor loss', actor_loss.item(), self.steps)

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
        return self.steps

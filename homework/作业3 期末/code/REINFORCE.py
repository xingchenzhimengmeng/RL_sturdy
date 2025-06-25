import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ale_py
import re
import matplotlib.pyplot as plt
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


class PolicyNetCNN(nn.Module):
    def __init__(self, action_dim):
        super(PolicyNetCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),  # 输入为 (4, 84, 84)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.policy_net = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255.0  # 归一化像素值
        x = self.fc(self.conv(x))
        return F.softmax(self.policy_net(x), dim=1)

class ValueNetCNN_Base(nn.Module):
    def __init__(self, action_dim):
        super(ValueNetCNN_Base, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),  # 输入为 (4, 84, 84)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.value_net = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0  # 归一化像素值
        x = self.fc(self.conv(x))
        return self.value_net(x).squeeze(-1)

class REINFORCE:
    def __init__(self, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNetCNN(action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.value_net = ValueNetCNN_Base(action_dim).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.current_step = 0

    def take_action(self, state):  # 根据动作概率分布随机采样
        self.current_step+=1
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']
        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.from_numpy(state_list[i]).float().unsqueeze(0).to(self.device)
            action = torch.from_numpy(action_list[i]).view(-1, 1).to(self.device)
            probs = self.policy_net(state)
            log_prob = torch.log(probs.gather(1, action))
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # 计算熵
            v = self.value_net(state)
            G = self.gamma * G + reward
            critic_loss = F.mse_loss(v, torch.tensor([G], device=self.device))
            critic_loss.backward()
            loss = -log_prob * (G - v.detach()) - 0.01 * entropy # 每一步的损失函数
            loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 梯度下降
        self.value_optimizer.step()  # 梯度下降

class Trainer():
    def __init__(self, env_name='ALE/Breakout-v5', max_steps = 20000000, learning_rate = 1e-4, gamma = 0.99):
        self.avg_rewardlist = []
        self.timesteps = []
        self.learning_rate = learning_rate
        self.num_episodes = 100000
        self.gamma = gamma
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
        print(self.device)
        self.env_name = env_name
        self.max_steps = max_steps
    def train(self):
        gym.register_envs(ale_py)
        env = gym.make(self.env_name, obs_type="grayscale")
        env = AtariPreprocessing(env, frame_skip=1)
        env = FrameStackObservation(env, 4)
        torch.manual_seed(0)
        action_dim = env.action_space.n
        agent = REINFORCE(action_dim, self.learning_rate, self.gamma,
                          self.device)
        return_list = []
        for i_episode in range(int(self.num_episodes)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state, _ = env.reset()
            done = False
            t=0
            while not done:
                t+=1
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(np.array([action]))
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            self.timesteps.append(agent.current_step)
            avg_reward0 = episode_return/t
            self.avg_rewardlist.append(avg_reward0)
            if i_episode % 10 == 0:
                print(f"Episode: {i_episode}, Avg reward: {avg_reward0:8.3f}, " +
                      f"episode_reward: {episode_return:8.2f}, episode_length:{t}")
            agent.update(transition_dict)
            if agent.current_step >= self.max_steps:
                break

    # 绘制单幕平均奖励曲线
    def plot_avg_reward(self):
        # 创建新的图形
        plt.figure()
        # 计算迄今为止最好的平均回报
        best_avg = []
        max_so_far = float('-inf')
        for reward in self.avg_rewardlist:
            if reward > max_so_far:
                max_so_far = reward
            best_avg.append(max_so_far)
        plt.plot(self.timesteps, self.avg_rewardlist, label='Average Reward per Epoch', color='steelblue',
                 alpha=0.8)
        plt.plot(self.timesteps, best_avg, label='Best Average Reward so far', color='crimson', linewidth=2,
                 linestyle='--')
        plt.xlabel("Timesteps")
        plt.ylabel("Average reward")
        title = (f'Reward curve of REINFORCE on game:{self.env_name}')
        title = re.sub(r'[\\/:*?"<>|\n]', '_', title)
        plt.legend(loc='lower right', fontsize=12, fancybox=True, shadow=True)
        plt.title(title)
        plt.savefig(title + '.png')
        # 关闭图形，释放内存
        plt.close()
if __name__ == '__main__':
    obj = Trainer()
    obj.train()
    obj.plot_avg_reward()
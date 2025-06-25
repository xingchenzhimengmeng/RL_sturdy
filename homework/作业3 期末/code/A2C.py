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

class ValueNet(torch.nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
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
        return self.value_net(x)

class A2C:
    def __init__(self, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNetCNN(action_dim).to(device)
        self.value_net = ValueNet().to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.current_step = 0
        self.update_time=0

    def take_action(self, state):  # 根据动作概率分布随机采样
        self.current_step+=1
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        self.update_time += 1
        states  = torch.from_numpy(np.array(transition_dict['states'])).float().to(self.device)
        actions  = torch.from_numpy(np.array(transition_dict['actions'])).view(-1, 1).to(self.device)
        rewards = torch.from_numpy(np.array(transition_dict['rewards'])).float().view(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.array(transition_dict['next_states'])).float().to(self.device)
        dones = torch.from_numpy(np.array(transition_dict['dones'])).float().view(-1, 1).to(self.device)
        # 时序差分目标
        td_target = rewards + self.gamma * self.value_net(next_states) * (1 - dones)
        probs = self.policy_net(states)
        td_delta = td_target - self.value_net(states)  # 时序差分误差
        log_probs = torch.log(probs.gather(1, actions))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # 计算熵
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        # if self.update_time % MOD < n:
        actor_loss = torch.mean(-log_probs * td_delta.detach() - 0.01 * entropy)
        actor_loss.backward()  # 计算策略网络的梯度
        self.policy_optimizer.step()  # 更新策略网络的参数
        # else:
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.value_net(states), td_target.detach()))
        critic_loss.backward()  # 计算价值网络的梯度
        self.value_optimizer.step()  # 更新价值网络的参数

class Trainer():
    def __init__(self, env_name='ALE/Breakout-v5', max_steps = 5000000, learning_rate = 1e-4, gamma = 0.99):
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
        env = gym.make(self.env_name, obs_type="grayscale")
        env = AtariPreprocessing(env, frame_skip=1)
        env = FrameStackObservation(env, 4)
        torch.manual_seed(0)
        action_dim = env.action_space.n
        agent = A2C(action_dim, self.learning_rate, self.gamma,
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
            max_episode_len = 512  # 最多只收集512步
            while not done and t < max_episode_len:
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
            transition_dict['dones'][-1] = True
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
        title = (f'Reward curve of A2C on game:{self.env_name}')
        title = re.sub(r'[\\/:*?"<>|\n]', '_', title)
        plt.legend(loc='lower right', fontsize=12, fancybox=True, shadow=True)
        plt.title(title)
        plt.savefig(title + '.png')
        # 关闭图形，释放内存
        plt.close()
if __name__ == '__main__':
    gym.register_envs(ale_py)
    # l = [1, 5, 5, 1, 5, 5]
    # for i in range(0, len(l), 2):
    #     n = l[i]
    #     m=l[i+1]
    #     MOD = n+m
    obj = Trainer(env_name='ALE/Breakout-v5')
    obj.train()
    obj.plot_avg_reward()
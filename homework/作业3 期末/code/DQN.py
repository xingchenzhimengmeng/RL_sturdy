import numpy as np
import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
from numpy import dtype
from torch import optim
import torch.nn.functional as F
from collections import deque, namedtuple
from gymnasium.wrappers import FrameStackObservation  , AtariPreprocessing
import matplotlib.pyplot as plt
import re
from itertools import count
import random, pickle, os.path, math, glob
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward'))
## 超参数
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 1000000
EPS_RANDOM_COUNT = 50000  # 前50000步纯随机用于探索
RENDER = False
lr = 1e-4
INITIAL_MEMORY = 10000
MEMORY_SIZE = 10 * INITIAL_MEMORY

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity  # 移动指针，经验池满了之后从最开始的位置开始将最近的经验存进经验池

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 从经验池中随机采样

    def __len__(self):
        return len(self.memory)
class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # 将卷积层的输出展平
        x = F.relu(self.fc4(x))  # .view(x.size(0), -1)
        out = self.head(x)
        return out

class DQN_agent():
    def __init__(self, in_channels=4, action_space=[], learning_rate=1e-4, memory_size=10000):
        self.in_channels = in_channels
        self.action_space = action_space
        self.action_dim = self.action_space.n
        self.memory_buffer = ReplayMemory(memory_size)
        self.stepdone = 0
        self.DQN = DQN(self.in_channels, self.action_dim).to(device)
        self.target_DQN = DQN(self.in_channels, self.action_dim).to(device)
        self.target_DQN.load_state_dict(self.DQN.state_dict())
        self.optimizer = optim.RMSprop(self.DQN.parameters(), lr=learning_rate, eps=0.001, alpha=0.95)
    def select_action(self, state):
        self.stepdone += 1
        state = state.to(device)
        epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.stepdone / EPS_DECAY)
        # epsilon-greedy策略选择动作
        if self.stepdone < EPS_RANDOM_COUNT or random.random() < epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], device=device, dtype=torch.long)
        else:
            action = self.DQN(state).detach().max(1)[1].view(1, 1)  # 选择Q值最大的动作并view
        return action
    def learn(self):
        # 经验池小于BATCH_SIZE则直接返回
        if self.memory_buffer.__len__() < BATCH_SIZE:
            return
        transitions = self.memory_buffer.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # 判断是不是在最后一个状态，最后一个状态的next设置为None
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action, dim=0).to(device)
        reward_batch = torch.cat(batch.reward, dim=0).to(device)
        # 计算当前状态的Q值
        state_action_values = self.DQN(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_actions = self.DQN(non_final_next_states).max(1)[1]  # 当前网络选动作
        next_state_values[non_final_mask] = self.target_DQN(non_final_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.DQN.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

class Trainer():
    def __init__(self, env, agent, n_episode, game_name, target_update, max_steps=1000000):
        self.target_update = target_update
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        self.rewardlist = []
        self.avg_rewardlist = []
        self.timesteps = []
        self.game_name = game_name
        self.max_steps = max_steps
    # 获取当前状态，将env返回的状态通过transpose调换轴后作为状态
    def get_state(self, obs):
        state = np.array(obs)
        state = torch.from_numpy(state).to(torch.uint8)
        return state.unsqueeze(0)
    # 训练智能体
    def train(self):
        for episode in range(self.n_episode):
            state, info = self.env.reset()
            state = self.get_state(state)
            episode_reward = 0.0
            avg_reward = 0.0
            t = 0
            done = False
            while not done:
                t+=1
                action = self.agent.select_action(state).to('cpu')
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                if not done:
                    next_state = self.get_state(next_state)
                else:
                    next_state = None
                self.agent.memory_buffer.push(state, action, next_state, torch.tensor([reward]))  # 里面的数据都是Tensor
                state = next_state
                # 经验池满50000
                if self.agent.stepdone >= INITIAL_MEMORY:
                    self.agent.learn()
                    if self.agent.stepdone % self.target_update == 0:
                        # print('======== target DQN updated =========')
                        self.agent.target_DQN.load_state_dict(self.agent.DQN.state_dict())
                if done:
                    avg_reward = episode_reward/(t+1)
                    break

            if episode%10==0:
                print(f"Episode: {episode}, Avg reward: {avg_reward:8.3f}, "+
                      f"episode_reward: {episode_reward:8.2f}, episode_length:{t}")
            self.timesteps.append(self.agent.stepdone)
            self.rewardlist.append(episode_reward)
            self.avg_rewardlist.append(avg_reward)
            self.env.close()
            if self.agent.stepdone >= self.max_steps:
                break
        return
    # 绘制单幕总奖励曲线
    def plot_total_reward(self):
        # 创建新的图形
        plt.figure()
        plt.plot(self.timesteps, self.rewardlist)
        plt.xlabel("Timesteps")
        plt.ylabel("Total reward per episode")
        plt.title('Total reward curve of DQN')
        title = (f'Total reward curve of DQN on game:{self.game_name}'+"_buffer_size"+ str(len(self.agent.memory_buffer))+'_target_update'+
                 str(self.target_update))
        title = re.sub(r'[\\/:*?"<>|\n]', '_', title)
        plt.legend(loc='lower right', fontsize=12, fancybox=True, shadow=True)
        plt.title(title)
        plt.savefig(title + '.png')
        # 关闭图形，释放内存
        plt.close()

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
        plt.plot(self.timesteps, self.avg_rewardlist, label='Average Reward per Epoch', color='steelblue', alpha=0.8)
        plt.plot(self.timesteps, best_avg, label='Best Average Reward so far', color='crimson', linewidth=2, linestyle='--')
        plt.xlabel("Timesteps")
        plt.ylabel("Average reward")
        title = (f'Reward curve of DQN on game:{self.game_name}'+"_buffer_size"+ str(len(self.agent.memory_buffer))+'_target_update'+
                 str(self.target_update))
        title = re.sub(r'[\\/:*?"<>|\n]', '_', title)
        plt.legend(loc='lower right', fontsize=12, fancybox=True, shadow=True)
        plt.title(title)
        plt.savefig(title + '.png')
        # 关闭图形，释放内存
        plt.close()
game_list = ['ALE/Breakout-v5', 'ALE/Pong-v5']
'''
Pong->1000000步
Breakout->5000000步
'''
def get_result(game_name, target_update, memory_size):
    # memory_size = 100000
    gym.register_envs(ale_py)
    env = gym.make(id=game_name, obs_type="grayscale")
    env = AtariPreprocessing(env, frame_skip=1)
    env = FrameStackObservation(env, 4)
    action_space = env.action_space
    agent = DQN_agent(in_channels=4, action_space=action_space, learning_rate=lr, memory_size=memory_size)
    trainer = Trainer(env, agent, n_episode, game_name=game_name, target_update=target_update, max_steps=max_steps)
    trainer.train()
    trainer.plot_avg_reward()
    trainer.plot_total_reward()
TARGET_UPDATE = 10000  # steps
n_episode = 100000
max_steps = 5000000
defalut_memory_size = 100000
game_name = 'ALE/Pong-v5'
# for game_name in game_list:
# get_result(game_name, TARGET_UPDATE, defalut_memory_size)
# get_result(game_name, 1000, defalut_memory_size)
get_result(game_name, 5000, 50000)
get_result(game_name, 5000, defalut_memory_size*2)
get_result(game_name, 5000, defalut_memory_size)
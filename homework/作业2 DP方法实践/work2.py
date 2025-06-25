import time
import torch
from google.protobuf.internal.wire_format import INT64_MIN
from sympy import false
import copy
import matplotlib.pyplot as plt

def q_value(i, j, k, grid):
    if i == 0 and j == 3 or i == 3 and j == 0:
        return 0
    if k == 0:
        j += 1
        if j == 4:
            j -= 1
    elif k == 1:
        i += 1
        if i == 4:
            i -= 1
    elif k == 2:
        j -= 1
        if j < 0:
            j += 1
    else:
        i -= 1
        if i < 0:
            i += 1
    return grid[i][j] - 1

# 策略评估
def policy_evaluation(grid, pi):
    small_number = 0.00001
    max_change = 1
    iteration_number = 0
    start = time.time()
    while max_change > small_number:
        max_change = 0
        iteration_number += 1
        grid2 = grid.clone()
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 3 or i == 3 and j == 0:
                    continue
                v = grid[i][j]
                t = 0
                for k, p in pi[i][j]:
                    t += p * q_value(i, j, k, grid)
                grid2[i][j] = t
                max_change = max(max_change, abs(t - v))
        grid = grid2
    # print("v_pi:", grid)
    # print("policy_evaluation收敛时间", time.time() - start)
    # print("迭代次数", iteration_number)
    return grid

# 策略优化
def policy_improvement(grid, pi):
    new_pi = copy.deepcopy(pi)
    policy_stable = True
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 3 or i == 3 and j == 0:
                continue
            a = pi[i][j]
            q_values = torch.tensor([q_value(i, j, k, grid) for k in range(4)])
            max_value = torch.max(q_values)
            max_indices = torch.nonzero(q_values == max_value).flatten().tolist()  # 所有最大值的位置
            # 生成新策略
            new_action = []
            total_prob = 1.0 / len(max_indices)  # 平均分配概率
            for k in range(4):
                # 如果该动作是最大值之一，则赋予概率，否则为0
                prob = total_prob if k in max_indices else 0.0
                new_action.append([k, prob])
            if a!=new_action:
                new_pi[i][j] = new_action
                policy_stable = False
    return new_pi, policy_stable

def policy_iteration(grid, pi):
    start = time.time()
    policy_stable = false
    iteration_number=0
    while not policy_stable:
        iteration_number+=1
        grid = policy_evaluation(grid, pi)
        pi, policy_stable = policy_improvement(grid, pi)
    print("v_pi:", grid)
    print("policy_iteration收敛时间", time.time() - start)
    print("迭代次数", iteration_number)
    visualize_policy(pi)

def visualize_policy(pi):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    fig, ax = plt.subplots(figsize=(6, 6))
    # 绘制网格线
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.grid(which='both', color='black', linestyle='-', linewidth=1)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.invert_yaxis()  # 保持与价值函数相同的坐标系方向
    ax.set_aspect('equal')
    # 箭头参数配置
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF']  # 红/绿/蓝/紫对应0-3
    labels = ['→', '↓', '←', '↑']
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 3 or i == 3 and j == 0:
                continue  # 跳过终止状态
            # 计算格子中心坐标
            x_center = j + 0.5
            y_center = i + 0.5
            # 绘制每个动作的概率箭头
            for action in pi[i][j]:
                k, p = action
                if p < 0.01: continue  # 忽略微小概率
                # 计算箭头向量
                dx, dy = 0, 0
                if k == 0:
                    dx = p * 0.4  # 右
                elif k == 1:
                    dy = p * 0.4  # 下
                elif k == 2:
                    dx = -p * 0.4  # 左
                else:
                    dy = -p * 0.4  # 上
                # 绘制箭头主体
                ax.arrow(x_center, y_center,
                         dx, dy,
                         head_width=0.1,
                         head_length=0.1,
                         fc=colors[k],
                         ec=colors[k],
                         alpha=0.6,
                         width=0.02)
                # 添加方向标签
                if dx != 0 or dy != 0:
                    ax.text(x_center + dx * 1.1,
                            y_center + dy * 1.1,
                            labels[k],
                            color=colors[k],
                            ha='center',
                            va='center',
                            fontsize=8)
    plt.title("策略可视化")
    plt.show()
if __name__ == '__main__':
    grid = torch.zeros([4, 4])
    pi = [[[[k, 0.25] for k in range(4)] for j in range(4)] for i in range(4)]
    policy_iteration(grid, pi)
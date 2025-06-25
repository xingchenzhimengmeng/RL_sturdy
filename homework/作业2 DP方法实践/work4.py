import time
import torch
from google.protobuf.internal.wire_format import INT64_MIN
from sympy import false
import copy
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False

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
    # start = time.time()
    # change_list = []
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
        # change_list.append(max_change)
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
    change_list = []
    max_change=0
    while not policy_stable:
        iteration_number+=1
        grid2 = policy_evaluation(grid, pi)
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 3 or i == 3 and j == 0:
                    continue
                max_change = max(max_change, abs(grid2[i][j] - grid[i][j]))
        change_list.append(max_change)
        grid = grid2
        pi, policy_stable = policy_improvement(grid, pi)
    # 绘制曲线
    print(iteration_number)
    plt.plot(range(1, len(change_list)+1), change_list,
             label='分布曲线',
             color='blue',  # 线条颜色
             linestyle='-',  # 实线样式
             linewidth=2,  # 线宽
             )  # 每5个点显示一个标记
    # 添加图形元素
    plt.title('policy_iteration收敛误差随着迭代次数的分布曲线')  # 标题
    plt.xlabel('迭代次数')  # X轴标签
    plt.ylabel('收敛误差')  # Y轴标签
    plt.grid(True, linestyle=':', alpha=0.6)  # 显示网格线（虚线，半透明）
    plt.legend(loc='upper right')  # 显示图例在右上方
    plt.tight_layout()  # 自动调整子图参数
    plt.show()
    # print(change_lists)
    # print("v_pi:", grid)
    # print("policy_iteration收敛时间", time.time() - start)
    # print("迭代次数", iteration_number)

def value_iteration(grid):
    small_number = 0.00001
    max_change = 1
    iteration_number = 0
    start = time.time()
    change_list = []
    while max_change > small_number:
        max_change = 0
        iteration_number += 1
        grid2 = grid.clone()
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 3 or i == 3 and j == 0:
                    continue
                v = grid[i][j]
                grid2[i][j] = max([q_value(i, j, k, grid) for k in range(4)])
                max_change = max(max_change, abs(grid2[i][j] - v))
        grid = grid2
        change_list.append(max_change)
    # 绘制曲线
    plt.plot(range(1, len(change_list)+1), change_list,
             label='分布曲线',
             color='blue',  # 线条颜色
             linestyle='-',  # 实线样式
             linewidth=2,  # 线宽
             )  # 每5个点显示一个标记
    # 添加图形元素
    plt.title('Value Iteration收敛误差随着迭代次数的分布曲线')  # 标题
    plt.xlabel('迭代次数')  # X轴标签
    plt.ylabel('收敛误差')  # Y轴标签
    plt.grid(True, linestyle=':', alpha=0.6)  # 显示网格线（虚线，半透明）
    plt.legend(loc='upper right')  # 显示图例在右上方
    plt.tight_layout()  # 自动调整子图参数
    plt.show()
    # print("v_pi:", grid)
    # print("value_iteration收敛时间", time.time() - start)
    # print("迭代次数", iteration_number)
    pi = [[[[k, 0.25] for k in range(4)] for j in range(4)] for i in range(4)]
    for i in range(4):
        for j in range(4):
            if i == 0 and j == 3 or i == 3 and j == 0:
                continue
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
            pi[i][j] = new_action


if __name__ == '__main__':
    start_grid = torch.zeros([4, 4])
    pi = [[[[k, 0.25] for k in range(4)] for j in range(4)] for i in range(4)]
    policy_iteration(start_grid, pi)
    value_iteration(start_grid)


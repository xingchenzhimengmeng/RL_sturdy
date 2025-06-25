import time
import torch
import copy

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
    print("v_pi:", grid)
    print("policy_evaluation收敛时间", time.time() - start)
    print("迭代次数", iteration_number)
    return grid

if __name__ == '__main__':
    # 状态价值V_pi
    grid = torch.zeros([4, 4])
    # 策略pi
    pi = [[[[k, 0.25] for k in range(4)] for j in range(4)] for i in range(4)]
    grid = policy_evaluation(grid, pi)
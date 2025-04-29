import numpy as np
import scipy.stats as stats  # 新增导入

def sdp():
    T = 3
    state = 4
    # 原需求定义替换为：
    mu = 1.5      # 正态分布均值
    sigma = 0.3   # 正态分布标准差
    # 生成离散化需求点（3点高斯积分）
    points, weights = stats.hermgauss(3)
    scaled_points = np.maximum(mu + sigma * np.sqrt(2) * points, 0)  # 非负处理
    d_list = np.round(scaled_points, 2)  # 保留两位小数
    probs = weights / sum(weights)
    
    # 删除原有d定义，改用离散化需求点
    # d = [1, 2]  # 注释或删除这行
    
    # 修改成本函数为基于离散需求点的期望计算
    # 在mu定义后添加：
    global mu
    hold_cost = lambda x, z: x + z - mu  # 等价于期望形式
    salv_cost2 = 0
    # 保持其他函数不变
    prod_cost1 = 0
    prod_cost2 = lambda z: 3 + 2 * z


    xx = np.zeros((T, state), dtype=int)  # 记录最优决策
    M = 100
    CC = np.full((T, state), M)  # 记录累计最优成本

    # 逆向阶段遍历
    for t in reversed(range(1, T+1)):  # t从3到1
        for temp_x in range(state):    # 状态遍历,这里到底要遍历多大的状态空间需要思考，例题中相当于是0，1，2，3四个状态，因为最高订购至水平为4，而每一期的需求至少为1，所以初始库存水平不会超过3。
            x = temp_x 
            if t == 1:  # 初始阶段特殊处理
                x = 0   # 固定初始库存为0
            
            minc = M
            # 生成可迭代的生产量范围（包含端点）
            for z in range(2 - x, 4 - x + 1):
                if t == 3:
                    current_salv = salv_cost2(x, z)
                    CC_next = 0
                else:
                    current_salv = salv_cost1
                    # 修改为多需求点期望计算
                    CC_next = 0
                    for d, p in zip(d_list, probs):
                        next_x = x + z - d
                        next_x_idx = max(min(int(next_x), state-1), 0)  # 状态边界保护
                        CC_next += p * CC[t][next_x_idx]
                
                current_prod = prod_cost1 if z == 0 else prod_cost2(z)
                temp_c = current_prod + hold_cost(x, z) - current_salv + CC_next
                
                if temp_c < minc:
                    minc = temp_c
                    CC[t-1][x] = minc  # 调整阶段索引
                    xx[t-1][x] = z
                
            if t == 1:
                break
    
    eOptimalCost = np.min(CC[0])
    index = np.argmin(CC[0])
    print(f'optimal expected total cost={eOptimalCost:.4f}')
    print(f'optimal production amount in period 1 = {xx[0][index]}')

if __name__ == "__main__":
    sdp()

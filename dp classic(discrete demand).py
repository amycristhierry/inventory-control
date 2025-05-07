import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from numba import njit, prange  # 新增：导入numba相关模块

# 记录代码开始运行的时间，用于后续计算代码运行耗时
start_time = time.time()

# 定义模型所需的参数
T = 10  # 总期数
mu = 10  # 需求服从正态分布的均值
sigma = 2  # 需求服从正态分布的标准差
h = 1  # 单位库存持有成本
p = 5  # 单位缺货成本
c = 2  # 单位订购成本

# 确定状态空间的边界，状态空间表示所有可能的库存水平
k = 10  # 安全系数，用于确定状态空间的范围
state_space_lower = -k * (mu + 3 * sigma)  # 状态空间的下限
state_space_upper = k * (mu + 3 * sigma)  # 状态空间的上限
# 生成状态空间数组，包含从下限到上限的所有整数
state_space = np.arange(int(state_space_lower), int(state_space_upper) + 1)

# 确定行动空间的边界，行动空间表示所有可能的订购量
action_space_lower = 0  # 行动空间的下限，订购量不能为负数
# 增大行动空间的上限，确保能覆盖可能的订购需求
action_space_upper = 1.5 * k * (mu + 3 * sigma) 
# 生成行动空间数组，包含从下限到上限的所有整数
action_space = np.arange(action_space_lower, int(action_space_upper) + 1)

# 打印状态空间和行动空间，方便调试和确认
print(f"状态空间: {state_space}")
print(f"行动空间: {action_space}")

# 初始化值函数和策略函数
# 值函数 V 记录每个阶段和状态下的最大期望价值
V = np.zeros((T + 1, len(state_space)))
# 策略函数 policy 记录每个阶段和状态下的最优行动（订购量）
policy = np.zeros((T, len(state_space)), dtype=int)

# 定义离散需求的范围和概率质量函数
demand_min = round(mu - 3 * sigma, 1)  # 需求下限，精确到小数点后1位
demand_max = round(mu + 3 * sigma, 1)  # 需求上限，精确到小数点后1位
discrete_demand = np.arange(demand_min, demand_max + 0.1, 0.1).round(1)  # 以0.1为步长生成离散需求
demand_pmf = norm.pdf(discrete_demand, loc=mu, scale=sigma)
demand_pmf /= demand_pmf.sum()  # 归一化概率

# ------------------------ 加速核心函数 ------------------------
@njit
def reward(x, z, d, h, p, c):
    """计算即时奖励（numba加速版）"""
    next_x = x + z - d
    holding_cost = h * max(next_x, 0)
    shortage_cost = p * max(-next_x, 0)
    ordering_cost = c * z
    return -(holding_cost + shortage_cost + ordering_cost)

@njit
def expected_reward(x, z, t, discrete_demand, demand_pmf, state_space, V, h, p, c):
    """通过求和计算期望奖励和下一阶段价值（numba加速版）"""
    total = 0.0
    for i in range(len(discrete_demand)):
        d = discrete_demand[i]
        # 计算当前需求下的奖励
        r = reward(x, z, d, h, p, c)
        # 计算下一阶段库存并查找状态索引
        next_x = x + z - d
        next_i = np.abs(state_space - next_x).argmin()
        # 累加加权值（奖励 + 下一阶段价值）
        total += (r + V[t + 1, next_i]) * demand_pmf[i]
    return total

@njit(parallel=True)  # 启用并行计算
def compute_policy(T, state_space, action_space, discrete_demand, demand_pmf, h, p, c, V, policy):
    """动态规划主逻辑（numba加速版，并行处理状态循环）"""
    for t in range(T - 1, -1, -1):  # 时间循环保持顺序（逆向递推）
        # 并行处理每个状态x（利用多核加速）
        for i in prange(len(state_space)):
            x = state_space[i]
            max_value = -np.inf
            best_action = 0
            # 遍历所有可能的订购量z
            for z in action_space:
                total_value = expected_reward(x, z, t, discrete_demand, demand_pmf, state_space, V, h, p, c)
                if total_value > max_value:
                    max_value = total_value
                    best_action = z
            # 记录当前阶段的最大价值和最优策略
            V[t, i] = max_value
            policy[t, i] = best_action

# ------------------------ 主流程调用加速函数 ------------------------
if __name__ == '__main__':
    start_time = time.time()
    
    # 调用加速后的动态规划计算函数
    compute_policy(T, state_space, action_space, discrete_demand, demand_pmf, h, p, c, V, policy)
    
    def get_optimal_order(n, x):
        """获取第 n 期初始库存为 x 时的最优订购量"""
        # 找到最接近的状态索引
        i = np.abs(state_space - x).argmin()
        # 返回对应阶段和状态下的最优订购量
        return policy[n - 1, i]
    
    # 打印每个周期的最优策略图
    def plot_optimal_policies():
        # 设置中文字体，确保图表标题和标签能正常显示中文
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体，若系统没有该字体，可尝试其他字体如 'Microsoft YaHei'
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False  
    
        for t in range(T):
            # 计算每个初始库存水平对应的最优订购至水平
            order_up_to_levels = state_space + policy[t]
            # 创建一个新的图表
            plt.figure(figsize=(10, 6))
            # 绘制初始库存水平与最优订购至水平的关系曲线
            plt.plot(state_space, order_up_to_levels, marker='o')
            # 设置图表标题
            plt.title(f'第 {t + 1} 期 最优订购至水平')
            # 设置 x 轴标签
            plt.xlabel('初始库存水平')
            # 设置 y 轴标签
            plt.ylabel('最优订购至水平')
            # 显示网格线
            plt.grid(True)
            # 显示图表
            plt.show()
    
    # 调用函数打印图表
    plot_optimal_policies()
    
    # 示例使用
    #n = 5  # 第 5 期
    #x = 10  # 初始库存水平为 10
    #optimal_z = get_optimal_order(n, x)
    #print(f"在第 {n} 期，初始库存水平为 {x} 时，最优订购量决策 z 为 {optimal_z}")
    
    # 记录代码结束运行的时间
    end_time = time.time()
    # 计算并打印代码运行耗时
    elapsed_time = end_time - start_time
    print(f"代码运行耗时: {elapsed_time:.2f} 秒")
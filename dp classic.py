import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from scipy.integrate import quad

# 记录代码开始运行的时间，用于后续计算代码运行耗时
start_time = time.time()

# 定义模型所需的参数
T = 3  # 总期数，代表整个规划的时间周期数量
mu = 10  # 需求服从正态分布的均值，即平均需求
sigma = 2  # 需求服从正态分布的标准差，反映需求的波动程度
h = 1  # 单位库存持有成本，即每单位库存每期需要支付的存储费用
p = 5  # 单位缺货成本，即每短缺一单位产品需要支付的额外费用
c = 2  # 单位订购成本，即每订购一单位产品需要支付的费用

# 确定状态空间的边界，状态空间表示所有可能的库存水平
k = 2  # 安全系数，用于确定状态空间的范围
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

# 定义需求的概率密度函数
def demand_pdf(d):
    return norm.pdf(d, loc=mu, scale=sigma)

def reward(x, z, d):
    """计算即时奖励"""
    # 计算下一阶段的库存水平
    next_x = x + z - d
    # 计算库存持有成本，仅当库存为正时产生
    holding_cost = h * max(next_x, 0)
    # 计算缺货成本，仅当库存为负时产生
    shortage_cost = p * max(-next_x, 0)
    # 计算订购成本
    ordering_cost = c * z
    # 即时奖励为负的总成本
    return -(holding_cost + shortage_cost + ordering_cost)

def expected_reward(x, z, t):
    """通过积分计算期望奖励和下一阶段的值函数"""
    def integrand(d):
        # 计算即时奖励
        r = reward(x, z, d)
        # 计算下一阶段的库存水平
        next_x = x + z - d
        # 找到最接近的状态索引
        next_i = np.abs(state_space - next_x).argmin()
        # 加上下一阶段的值函数
        return (r + V[t + 1, next_i]) * demand_pdf(d)
    result, _ = quad(integrand, 0, np.inf)
    return result

# 动态规划逆向递推，从倒数第二期（T - 1）开始，到第 0 期结束
for t in range(T - 1, -1, -1):
    # 遍历所有可能的库存状态
    for i, x in enumerate(state_space):
        # 初始化最大价值为负无穷，用于后续比较
        max_value = -np.inf
        # 初始化最优行动（订购量）为 0
        best_action = 0
        # 遍历行动空间中的每一个可能的订购量 z
        for z in action_space:
            # 计算期望奖励和下一阶段的值函数
            total_value = expected_reward(x, z, t)
            # 直接比较 total_value 与 max_value
            if total_value > max_value:
                # 更新最大价值
                max_value = total_value
                # 更新最优行动（订购量）
                best_action = z
        # 记录当前阶段和状态下的最大期望价值
        V[t, i] = max_value
        # 记录当前阶段和状态下的最优行动（订购量）
        policy[t, i] = best_action

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

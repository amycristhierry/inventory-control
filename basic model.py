import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
start = time.perf_counter()
from numba import jit

@jit(nopython=True)  # 使用Numba JIT编译以提高性能
def bm(T, B, K, D_t, dprob, delta_t, delta_prob, cu,p_t,s, max_D):
    v = np.zeros((T + 2, B + (T+1)*max_D + 1))  # 创建一个二维数组用于存储最优值
    a_qs = np.zeros((T + 2, B + (T+1)*max_D + 1))  # 创建一个二维数组用于存储最优解 a


    # 逆向遍历时间阶段，从 T+1 到 1
    for t in range(T+1, 0, -1):
        # 遍历当前阶段所有可能的库存状态
        for x in range(B + (t - 1) * max_D + 1):
            # 将当前循环索引 x 转换为实际的库存水平 ori_x
            ori_x = x - (t - 1) * max_D
            # 初始化最大价值为负无穷，用于后续比较
            max_value = -np.inf
            # 初始化最优订购量为 0
            best_qs = 0
            # 计算当前最大可订购量，取 B 和 B - ori_x 中的较小值
            Q = min(B, B - ori_x)
            # 遍历所有可能的订购量
            for qs in range(Q + 1):
                # 初始化数组，用于存储不同 delta 值下的计算结果
                grid = np.zeros((len(delta_t)))
                # 初始化数组，用于存储不同需求值下的收益
                H = np.zeros((len(D_t)))
                # 初始化数组，用于存储不同 delta 值下的价值
                vgrid = np.zeros((len(delta_t)))
                # 初始化二维数组，用于存储不同 delta 和需求组合下的价值
                value1 = np.zeros((len(delta_t), len(D_t)))
                # 初始化数组，用于存储不同 delta 值下的期望价值
                value2 = np.zeros((len(delta_t)))
                # 遍历所有可能的 delta 值
                for i, delta in enumerate(delta_t):
                    # 计算考虑 delta 和 K 后的有效订购量
                    grid[i] = max(qs - delta * K, 0)
                    # 计算不同 delta 值下的订购成本
                    vgrid[i] = - cu * grid[i]
                    # 遍历所有可能的需求值
                    for d, D in enumerate(D_t):
                        # 将需求值转换为整数
                        D = int(D)
                        # 计算下一阶段的库存水平
                        new_x = t * max_D + ori_x - D + qs
                        # 计算不同需求值下的收益
                        H[d] = p_t * D - s * max(D - ori_x, 0)
                        # 从存储最优值的数组中获取下一阶段的价值
                        value1[i][d] = v[t + 1][new_x]
                    # 计算不同 delta 值下的期望价值
                    value2[i] = np.sum(value1[i] * dprob)
                # 计算总的期望价值
                value = np.sum(value2 * delta_prob)
                # 计算不同 delta 值下订购成本的期望
                Egrid = np.sum(vgrid * delta_prob)
                # 计算不同需求值下收益的期望
                EH = np.sum(H * dprob)
                # 计算当前订购量下的总价值
                new_value = Egrid + EH + value

                # 如果当前总价值大于等于最大价值，则更新最大价值和最优订购量
                if new_value >= max_value:
                    max_value = new_value
                    best_qs = qs
            # 记录当前阶段和库存状态下的最大价值，保留 100 位小数
            v[t][x] = round(max_value, 100)
            # 记录当前阶段和库存状态下的最优订购量
            a_qs[t][x] = best_qs

    # 返回存储最优值和最优订购量的二维数组
    return v, a_qs




def discretize_normal(mu, sigma, lower_bound, upper_bound, n):
    # 计算每个区间的宽度
    bin_width = (upper_bound-lower_bound)/n
    print(sigma)

    # 创建离散化的取值范围
    x = np.arange(lower_bound, upper_bound+bin_width, bin_width)

    # 创建离散化后的概率密度函数
    pdf = norm.pdf(x, loc=mu, scale=sigma)

    # 计算每个区间的概率密度
    probabilities = pdf * bin_width


    # 标准化概率，确保其总和等于1
    probabilities /= np.sum(probabilities)

    return x, probabilities




def plot_fig_qs_x(qs):
    x_values = np.arange(-(t_i-1)*max_D, B + 1)
    qs_values = np.zeros_like(x_values, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))


    for i, actual_x in enumerate(x_values):
        x_index = actual_x + (t_i-1)*max_D  # 将实际 x 转换为索引
        qs_values[i] = qs[x_index]
    ax.plot(x_values, qs_values, label=f'qs')  # 给每条曲线设置一个标签
    ax.set_xlabel('x')
    ax.set_ylabel('qs')
    ax.set_title("qs versus x")
    ax.legend()
    plt.show()


def plot_fig_qs_plus_x(qs):
    x_values = np.arange(-(t_i-1)*max_D, B + 1)
    qs_plus_x_values = np.zeros_like(x_values, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 10))


    for i, actual_x in enumerate(x_values):
        x_index = actual_x + (t_i-1)*max_D  # 将实际 x 转换为索引
        qs_plus_x_values[i] = qs[x_index] + actual_x
    ax.plot(x_values, qs_plus_x_values)  # 给每条曲线设置一个标签
    ax.set_xlabel('x')
    ax.set_ylabel('qs_plus_x')
    ax.set_title("qs_plus_x")
    # ax.legend()
    plt.show()

def plot_fig_v_x(v):
    x_values = np.arange(-(t_i-1)*max_D, B + 1)
    v_values = np.zeros_like(x_values, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 10))


    for i, actual_x in enumerate(x_values):
        x_index = actual_x + (t_i-1)*max_D  # 将实际 x 转换为索引
        v_values[i] = v[x_index]
    ax.plot(x_values, v_values, label=f'v')  # 给每条曲线设置一个标签
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title("v versus x")
    ax.legend()
    plt.show()




if __name__ == '__main__':
    T = 12
    K = 0
    B= 21
    cu = 40
    p_t = 30
    s = 10  # 跟s有关系



    d_mu = 10  # 均值
    d_sigma = 2  # 标准差

    # 设置离散化的范围
    d_lower_bound = int(d_mu - 3 * d_sigma)  # 下界
    d_upper_bound = int(d_mu + 3 * d_sigma)  # 上界

    n1 = 10
    # 进行离散化并计算期望
    D_t, dprob = discretize_normal(d_mu, d_sigma, d_lower_bound, d_upper_bound,n1)
    max_D = int(max(D_t))
    # D_t = np.array([20, 20, 20])
    # max_D = int(max(D_t))
    # dprob = np.array([0, 1, 0])

    delta_mu = 0.5
    delta_sigma = 0.7
    delta_lower_bound = max(delta_mu - 3 * delta_sigma, 0)  # 下界
    delta_upper_bound = min(delta_mu + 3 * delta_sigma, 1)  # 上界
    n2 = 10
    delta_t, delta_prob = discretize_normal(delta_mu, delta_sigma, delta_lower_bound, delta_upper_bound, n2)

    v, a_qs = bm(T, B, K, D_t, dprob, delta_t, delta_prob, cu,p_t,s, max_D)
    # np.save(file='bm_v', arr=v)
    # np.save(file='bm_a', arr=a_qs)
    for t_i in range(1,T + 1):
        plot_fig_qs_x(a_qs[t_i, :])
    # plot_fig_qs_plus_x(a_qs[t_i, :])
    #
    # plot_fig_v_x(v[t_i, :])






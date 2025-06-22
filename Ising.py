import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ---------- 参数设置 ----------
L = 20                # 晶格边长
J = 1.0               # 耦合常数
n_steps = 1000000   # Monte Carlo 总步数
record_interval = 1000  # 每隔多少步记录一次磁化

def total_energy(spins):
    """计算系统总能量 E = -J * sum_{<i,j>} s_i s_j"""
    E = 0.0
    for i in range(L):
        for j in range(L):
            S = spins[i,j]
            # 仅右与下方以避免重复计数
            E -= J * S * spins[i, (j+1)%L]
            E -= J * S * spins[(i+1)%L, j]
    return E

def delta_energy(spins, i, j):
    """计算翻转 (i,j) 处自旋导致的能量差 Delta_E"""
    S = spins[i,j]
    # 周期边界的四邻域和
    neigh_sum = (
        spins[(i-1)%L, j] +
        spins[(i+1)%L, j] +
        spins[i, (j-1)%L] +
        spins[i, (j+1)%L]
    )
    # Delta_E = E_new - E_old = 2J * s_ij * sum(neigh)
    return 2 * J * S * neigh_sum

def run_ising(T):
    """在温度 T 下运行 Ising 模拟，返回记录的磁化列表"""
    # 随机初始化自旋阵列
    spins = np.random.choice([-1, 1], size=(L, L))
    M_list = []
    steps = []
    M = spins.sum()
    for step in range(1, n_steps+1):
        # 随机选格点并尝试翻转
        i = random.randrange(L)
        j = random.randrange(L)
        dE = delta_energy(spins, i, j)
        # Metropolis 算法
        if dE <= 0 or random.random() < math.exp(-dE / T):
            # 接受翻转
            spins[i,j] *= -1
            M += 2 * spins[i,j]  # 更新总磁化
        # 记录磁化
        if step % record_interval == 0:
            M_list.append(M)
            steps.append(step)
    return steps, M_list

def plot_magnetization(results):
    """绘制不同温度下磁化强度随步数变化曲线"""
    plt.figure(figsize=(8,6))
    for T, (steps, M_list) in results.items():
        plt.plot(steps, M_list, label=f"T = {T}")
    plt.xlabel("Monte Carlo steps")
    plt.ylabel("Total magnetization M")
    plt.legend()
    plt.title(f"Ising model (L={L}, J={J})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    temps = [1.0, 2.0, 3.0]
    results = {}
    for T in temps:
        print(f"Running for T = {T} ...")
        steps, M_list = run_ising(T)
        results[T] = (steps, M_list)
    plot_magnetization(results)

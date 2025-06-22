import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random, math

# ---------- 参数 ----------
L = 50
T0 = 1.0           # 初始温度
tau = 1000000.0      # 冷却常数
T_min = 1e-3       # 最低温度
steps_per_frame = 2000
max_frames = 300

# 状态表示
dimers = set()                         # 存放 frozenset({(i,j),(i2,j2)})  
occupancy = np.zeros((L, L), bool)     # 标记格点是否被占

def propose_move(T):
    """
    Metropolis 步骤：
    1) 随机选一对相邻格点
    2) 如果两点都空 => 试加；如果正好由同一二聚体占 => 试删；否则跳过
    3) 计算 ΔE，Metropolis 接受/拒绝
    """
    i, j = random.randrange(L), random.randrange(L)
    di, dj = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
    i2, j2 = (i+di)%L, (j+dj)%L
    bond = frozenset({(i,j),(i2,j2)})

    # 判断操作类型
    if not occupancy[i,j] and not occupancy[i2,j2]:
        # 试放置
        dE = -1  # 放置后 E_new - E_old = -1
        accept = (dE <= 0) or (random.random() < math.exp(-dE/T))
        if accept:
            dimers.add(bond)
            occupancy[i,j] = occupancy[i2,j2] = True

    elif bond in dimers:
        # 试移除
        dE = +1  # 移除后 ΔE = +1
        accept = (dE <= 0) or (random.random() < math.exp(-dE/T))
        if accept:
            dimers.remove(bond)
            occupancy[i,j] = occupancy[i2,j2] = False
    # 其他情况do nothing

def temperature(t):
    """指数冷却并施加下限 T_min"""
    T = T0 * math.exp(-t / tau)
    return T if T > T_min else T_min

def get_grid():
    """生成可视化矩阵：0空，1横向，2纵向"""
    grid = np.zeros((L, L), int)
    for bond in dimers:
        (x1,y1),(x2,y2) = list(bond)
        if x1 == x2:
            # 同行 => 横向
            y0, y1 = sorted((y1,y2))
            grid[x1, y0] = grid[x2, y1] = 1
        else:
            # 同列 => 纵向
            x0, x1 = sorted((x1,x2))
            grid[x0, y1] = grid[x1, y2] = 2
    return grid

# ---------- 可视化设定 ----------
fig, ax = plt.subplots(figsize=(6,6))
cmap = plt.cm.get_cmap('Set1', 3)
im = ax.imshow(get_grid(), cmap=cmap, vmin=0, vmax=2)
title = ax.set_title("Step: 0, Dimers: 0")

# MCMC 计数
total_steps = 0

def update(frame):
    global total_steps
    # 每帧做若干 MCMC
    for _ in range(steps_per_frame):
        T = temperature(total_steps)
        propose_move(T)
        total_steps += 1

    grid = get_grid()
    im.set_array(grid)
    title.set_text(f"Step: {total_steps}, Dimers: {len(dimers)}")
    return im, title

ani = animation.FuncAnimation(
    fig, update, frames=max_frames, interval=1, blit=False
)
plt.show()

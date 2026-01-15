# CartPole_DQN/tool/plot_distribution.py
# 负责绘制step1B的奖励分布图-呈现非正态分布

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator # 沿用你之前的风格设置

def plot_step1b_pure_matplotlib():
    # ==========================================
    # 1. 数据录入区 (请直接粘贴你的完整列表)
    # ==========================================
    raw_scores = [np.float64(293.5857142857143), np.float64(500.0), np.float64(385.1857142857143), np.float64(163.74285714285713), np.float64(216.68571428571428), np.float64(131.84285714285716), np.float64(500.0), np.float64(500.0), np.float64(500.0), np.float64(407.9142857142857), np.float64(114.15714285714286), np.float64(309.6714285714286), np.float64(456.72857142857146), np.float64(60.01428571428571), np.float64(499.0857142857143), np.float64(491.04285714285714), np.float64(500.0), np.float64(500.0), np.float64(500.0), np.float64(193.6)]

    # 转为 numpy 数组 (Matplotlib 处理这个速度最快)
    scores = np.array(raw_scores, dtype=float)

    # ==========================================
    # 2. 绘图逻辑 (纯 Matplotlib)
    # ==========================================
    # 保持和你 Step1A 一样的画布比例 (10x6) 和清晰度 (dpi=120)
    plt.figure(figsize=(10, 6), dpi=120)

    # 绘制直方图 (Histogram)
    # bins=10: 把数据分成10组，足以看清双峰
    # alpha=0.7: 让颜色稍微透明一点，更有质感
    # edgecolor='black': 给柱子加黑边，看清楚边界
    # zorder=3: 让柱子显示在网格线上面
    plt.hist(scores, bins=10, range=(0, 500), color='#1f77b4', edgecolor='black', alpha=0.75, zorder=3)

    # ==========================================
    # 3. 装饰与标注 (Style)
    # ==========================================
    plt.title("Step 1B: Distribution of Training Rewards (N=20)", fontsize=14)
    plt.xlabel("Average Training Reward", fontsize=12)
    plt.ylabel("Frequency (Count)", fontsize=12)

    # 开启网格 (跟 Step1A 一致)
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # 标注重要线条
    # 1. 满分线 (Ceiling)
    plt.axvline(x=500, color='red', linestyle='--', linewidth=1.5, label='Ceiling (500)')
    
    # 2. 均值线 (Mean)
    mean_val = np.mean(scores)
    plt.axvline(x=mean_val, color='green', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.1f})')

    # 添加图例
    plt.legend(loc='upper center')

    # ==========================================
    # 4. 保存与展示
    # ==========================================
    save_path = "step1B_distribution_plt.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"图片已生成并保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_step1b_pure_matplotlib()
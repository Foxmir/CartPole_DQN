# CartPole_DQN/tool/plot_forest.py
# 负责绘制 step3 的森林图，复现 Controller 的配对差值置信区间计算逻辑
# 只画seed = 30和40时刻的图, 30时发生剪枝，40是预算上限
# 使用时是将改代码核json原数据（step3controller生成的big_dict）放在根目录下直接运行即可

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# ================= 配置区域 =================
JSON_PATH = Path("big_dict.json")
OUTPUT_IMG = "step3_forest_plots.png"
N_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.99
# ============================================

def get_snapshot_stats(big_dict, target_n):
    """
    严格复现 Controller 的逻辑:
    1. 截取数据长度
    2. 计算均值并排名确定当前的 Top1
    3. 固定随机种子并生成索引矩阵 (与 Controller 结构完全一致)
    4. 计算所有存活者相对于 Top1 的 99% 配对差值 CI
    """
    # 📑【逻辑1】过滤当前存活的候选者 (n=40时自动排除c02)
    active_ids = [k for k, v in big_dict.items() if len(v) >= target_n]
    
    # 📑【逻辑2】排名确定当时的冠军
    candidate_means = {k: np.mean(big_dict[k][:target_n]) for k in active_ids}
    sorted_candidates = sorted(candidate_means.items(), key=lambda x: x[1], reverse=True)
    sorted_ids = [x[0] for x in sorted_candidates]
    
    top1_id = sorted_ids[0]
    top1_means_list = np.array(big_dict[top1_id][:target_n])
    
    # 📑【逻辑3】固定随机种子并生成索引矩阵 (结构复现)
    np.random.seed(target_n)
    index_matrix = np.random.randint(0, target_n, size=(N_RESAMPLES, target_n))
    
    results = []
    
    # 📑【逻辑4】计算配对差值 CI
    lower_percentile = (1 - CONFIDENCE_LEVEL) / 2 * 100
    upper_percentile = (1 + CONFIDENCE_LEVEL) / 2 * 100

    for comp_id in sorted_ids:
        if comp_id == top1_id:
            results.append({
                'id': f"{comp_id} (Leader)",
                'mean_diff': 0, 'ci_low': 0, 'ci_high': 0
            })
        else:
            comp_means_list = np.array(big_dict[comp_id][:target_n])
            differences_list = top1_means_list - comp_means_list
            resampled_means = differences_list[index_matrix].mean(axis=1)
            
            ci_low = np.percentile(resampled_means, lower_percentile)
            ci_high = np.percentile(resampled_means, upper_percentile)
            mean_diff = np.mean(differences_list)
            
            results.append({
                'id': comp_id,
                'mean_diff': mean_diff, 'ci_low': ci_low, 'ci_high': ci_high
            })
            
    return results

def draw_forest_plot(ax, stats, title, color):
    """使用 Matplotlib 绘制增强后的森林图"""
    ids = [s['id'] for s in stats]
    means = [s['mean_diff'] for s in stats]
    err_low = [s['mean_diff'] - s['ci_low'] for s in stats]
    err_high = [s['ci_high'] - s['mean_diff'] for s in stats]
    
    # 1. 绘制误差棒
    ax.errorbar(means, range(len(ids)), xerr=[err_low, err_high], 
                fmt='o', color=color, ecolor=color, capsize=3, 
                elinewidth=1.5, markeredgewidth=2, label='Candidate Performance')
    
    # 2. 为 c02 添加“被剪枝”的明确标注
    for i, s in enumerate(stats):
        if "c02" in s['id']:
            ax.text(s['ci_low'] - 3, i, f"LB={s['ci_low']:.2f}\n(Pruned)", 
                    va='center', ha='right', color='red', 
                    fontweight='bold', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # 3. 设置辅助线（明确标注图例）
    ax.axvline(x=10, color='black', linestyle='--', linewidth=1.5, 
               label='Pruning Threshold (LB > 10)')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    
    # 4. 坐标轴与标签美化
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids)
    ax.invert_yaxis()
    
    ax.set_title(f"{title}\n(N={N_RESAMPLES:,} Bootstrap Resamples)", fontweight='bold', fontsize=11)
    ax.set_xlabel("Paired-Difference (Leader - Competitor) [99% Bootstrap CI]", fontsize=10)
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    
    # 5. 显示图例（解释虚线意义）
    ax.legend(loc='lower right', fontsize=8)

def main():
    if not JSON_PATH.exists():
        print(f"Error: {JSON_PATH} not found.")
        return
    
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        big_dict = json.load(f)

    print("Calculating stats for n=30...")
    stats_30 = get_snapshot_stats(big_dict, 30)
    print("Calculating stats for n=40...")
    stats_40 = get_snapshot_stats(big_dict, 40)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11))
    
    draw_forest_plot(ax1, stats_30, "A. Snapshot at n=30 Seeds (Initial Pruning Moment)", "tab:blue")
    draw_forest_plot(ax2, stats_40, "B. Snapshot at n=40 Seeds (Final State Analysis)", "tab:orange")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300)
    print(f"Success! Forest plot saved to {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()

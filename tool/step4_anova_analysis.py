# CartPole_DQN/tool/step4_anova_analysis.py
# 用以对step4的cartpole敏感性分析数据进行ANOVA分析

'''
1.数据清洗：读取 CSV, 只留 4 因子 + final_mean。
    自检：确保数据行数符合预期（约补齐后）。
2.模型设定：
    因变量: final_mean
    自变量：全因子模型 (4个超参的主效应 + 所有交互)
    Type: Type II Sum of Squares
    噪音项: 由种子带来的变异自动填充(Residual), 不显式建模。
3.输出物：
    1. 一张 ANOVA 表：包含 columns [Source, Sum_sq(平方和), df(自由度), F, PR(>F)(P值), eta_sq (部分效应量)]。
        解读承诺：直接看 Eta_sq 排序，谁大谁就是跨种子的“真大哥”。
    2. 可视化：
        主效应图(4张): 每个因子的均值变化曲线（带置信区间，置信区间越窄说明跨种子越稳）。
        交互图（针对显著的）：展示参数之间怎么互相掣肘。

'''

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ================= 配置区域 =================
# 输入数据路径 (请确保文件存在)
INPUT_CSV_PATH = os.path.join("data", "step4", "step4_data_cleaned_flattened.csv")
# 输出结果目录
OUTPUT_DIR = os.path.join("data", "step4", "anova_results")
# 字体设置 (防止中文乱码，如果报错可尝试 'SimHei' 或 'Arial')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False
# ===========================================

def main():
    # 1. 准备环境
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已创建输出目录: {OUTPUT_DIR}")

    # 2. 读取数据
    print(f"🔄 正在读取数据: {INPUT_CSV_PATH} ...")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {INPUT_CSV_PATH}")
        return

    # 3. 数据重命名 (statsmodels 公式不支持带点的列名，如 agent.tau)
    # 映射字典: {原列名: 新列名}
    rename_map = {
        'agent.tau': 'tau',
        'agent.epsilon_decay': 'epsilon_decay',
        'agent.learning_rate': 'learning_rate',
        'training.batch_size': 'batch_size',
        'final_mean': 'final_mean'
    }
    
    # 检查列是否存在
    missing_cols = [col for col in rename_map.keys() if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误: CSV中缺少以下列，请检查数据清洗步骤:\n{missing_cols}")
        print(f"当前CSV列名: {list(df.columns)}")
        return

    df_clean = df.rename(columns=rename_map)
    print("✅ 列名已标准化 (移除点号).")

    # 4. 定义统计模型
    # 目的: final_mean ~ 4因子全交互
    # C(...) 表示 Categorical (分类变量)，确保库把它当离散水平处理
    formula = (
        "final_mean ~ "
        "C(learning_rate) * C(batch_size) * C(epsilon_decay) * C(tau)"
    )
    print(f"🧪 正在拟合 ANOVA 模型 (Type II Sum of Squares)...\n公式: {formula}")

    model = ols(formula, data=df_clean).fit()
    
    # 5. 生成 ANOVA 表 (Type II)
    # typ=2 是平衡设计下的标准选择
    anova_table = sm.stats.anova_lm(model, typ=2)

    # 6. 计算效应量 (Partial Eta Squared)
    # 公式: SS_effect / (SS_effect + SS_residual)
    # 注意: 在 statsmodels 的 type 2 表中，Residual 行的索引通常叫 'Residual'
    ss_residual = anova_table.loc['Residual', 'sum_sq']
    
    anova_table['eta_sq_partial'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + ss_residual)
    
    # 整理表格显示
    anova_table['F'] = anova_table['F'].round(2)
    anova_table['PR(>F)'] = anova_table['PR(>F)'].apply(lambda x: f"{x:.4f}" if x >= 0.001 else "<0.001")
    anova_table['eta_sq_partial'] = anova_table['eta_sq_partial'].round(4)
    
    # 排序: 按效应量从大到小 (Residual除外)
    results_sorted = anova_table.drop('Residual').sort_values('eta_sq_partial', ascending=False)
    results_sorted = pd.concat([results_sorted, anova_table.loc[['Residual']]])

    # 7. 打印并保存报告
    print("\n" + "="*50)
    print("📊 ANOVA 分析报告 (按效应量排序)")
    print("="*50)
    print(results_sorted[['sum_sq', 'df', 'F', 'PR(>F)', 'eta_sq_partial']])
    
    csv_save_path = os.path.join(OUTPUT_DIR, "step4_anova_report.csv")
    results_sorted.to_csv(csv_save_path)
    print(f"\n✅ 完整报告已保存至: {csv_save_path}")

    # 8. 可视化 - 主效应图
    print("\n🎨 正在绘制主效应图...")
    factors = ['learning_rate', 'batch_size', 'epsilon_decay', 'tau']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    fig.suptitle('Step 4 Main Effects (主效应 Analysis)', fontsize=16)

    for ax, factor in zip(axes, factors):
        # 绘制点图+置信区间 (默认95% CI)
        sns.pointplot(data=df_clean, x=factor, y='final_mean', 
                      errorbar='ci', capsize=0.1, ax=ax, color='#e74c3c')
        ax.set_title(f"Factor: {factor}")
        ax.set_ylabel("Final Mean Score")
        ax.set_xlabel("Level")
        ax.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(OUTPUT_DIR, "step4_main_effects.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ 主效应图已保存: {plot_path}")

    # 9. 可视化 - 交互效应 (只画效应量最大的前2个显著二阶交互，避免图太多)
    # 筛选: 包含 ':' (交互项), 排除三阶以上(冒号数量>1可能是高阶), 且 P值显著
    # 注意: 这里简单起见，我们手动指定通常最关心的交互，或者根据表里 top 2 自动画
    
    # 从排序后的表中找前2个交互项
    interaction_rows = [idx for idx in results_sorted.index if ':' in idx and results_sorted.loc[idx, 'eta_sq_partial'] > 0.01] 
    # 阈值 0.01 是为了只看有点意义的
    
    if interaction_rows:
        print(f"\n🎨 正在绘制 Top 交互效应: {interaction_rows[:2]} ...")
        for i, term in enumerate(interaction_rows[:2]): # 只画前2个
            # term 格式如 "C(learning_rate):C(batch_size)"
            # 需要解析出两个因子名
            try:
                # 粗暴解析: 移除 'C(' 和 ')'，然后 split ':'
                clean_term = term.replace("C(", "").replace(")", "")
                f1, f2 = clean_term.split(":")[:2] # 取前两个
                
                plt.figure(figsize=(8, 6))
                sns.pointplot(data=df_clean, x=f1, y='final_mean', hue=f2, 
                              errorbar=None, linestyle='-', marker='o') # 交互图通常不画CI以免太乱，或者画也行
                plt.title(f"Interaction: {f1} x {f2}")
                plt.grid(True, linestyle='--', alpha=0.3)
                
                int_path = os.path.join(OUTPUT_DIR, f"step4_interaction_{i+1}_{f1}_x_{f2}.png")
                plt.savefig(int_path, dpi=300)
                plt.close()
                print(f"   已保存交互图: {int_path}")
            except Exception as e:
                print(f"   ⚠️ 无法自动绘制交互项 {term}: {e}")
    else:
        print("\nℹ️ 没有发现效应量 > 0.01 的显著交互项，跳过绘制交互图。")

    print("\n🎉 分析完成!")

if __name__ == "__main__":
    main()

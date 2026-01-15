# CartPole_DQN/tool/sweeps_data_preprocessing.py
# 用以清洗从WandB下载下来的sweeps数据CSV文件(这里是针对step4的cartpole敏感性分析数据)

import pandas as pd
import ast
import os

# ================= 用户配置区域 =================
# 获取当前脚本所在目录 (CartPole_DQN/tool)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (CartPole_DQN)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 使用 os.path.join 自动处理 Windows/Linux 路径差异，并精准定位到 data 文件夹
INPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "step4", "step4-grid.csv")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "step4", "step4_data_cleaned_flattened.csv")
# ===============================================

def clean_wandb_csv():
    print(f"🔄 正在读取文件: {INPUT_CSV_PATH} ...")
    
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"❌ 错误: 找不到文件 {INPUT_CSV_PATH}，请检查路径。")
        return

    # 1. 读取原始 CSV
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"📊 原始数据包含 {len(df)} 行。")

    # 2. 定义解析函数：把字符串 "{'a':1}" 变成真正的字典 {'a':1}
    def parse_dict_string(x):
        try:
            if pd.isna(x) or x == "":
                return {}
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return {}

    print("🔨 正在拆解嵌套结构 (Config & Summary) ...")

    # 3. 处理 Config 列 (超参数)
    # 将字符串转为字典
    config_dicts = df['config'].apply(parse_dict_string)
    # 使用 json_normalize 将字典铺平 (例如 agent.learning_rate)
    config_flattened = pd.json_normalize(config_dicts)
    
    # 4. 处理 Summary 列 (结果指标)
    summary_dicts = df['summary'].apply(parse_dict_string)
    summary_flattened = pd.json_normalize(summary_dicts)

    # 5. 合并数据
    # 我们保留原始的 'name' 列，加上拆解后的 config 和 summary
    final_df = pd.concat([df['name'], config_flattened, summary_flattened], axis=1)

    # 6. (可选) 过滤掉无用的列
    # 删除 WandB 自动生成的内部列 (以 _wandb 或 _step 开头的)
    cols_to_drop = [c for c in final_df.columns if c.startswith('_wandb') or c.startswith('_step') or c.startswith('_runtime') or c.startswith('_timestamp')]
    final_df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 7. 保存
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"✅ 成功! 清洗后的数据已保存至: {OUTPUT_CSV_PATH}")
    print(f"   新文件包含 {len(final_df.columns)} 个列 (因子 + 指标)。")
    print("   👉 现在你可以直接用 Excel 打开它进行 ANOVA 分析了。")

if __name__ == "__main__":
    clean_wandb_csv()

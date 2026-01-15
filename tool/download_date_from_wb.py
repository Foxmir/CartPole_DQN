# CartPole_DQN/tool/download_date_from_wb.py
# 可以从WandB的Sweep中拉取所有运行数据，并保存为CSV文件
# 使用前，先登录
# 它是需要先从wb拉取（即下载）到内存中，下载完后，整合成一个DataFrame，最后时刻保存为CSV文件

import pandas as pd
import wandb
import sys

# 1. 配置基础信息
ENTITY = "foxmir-stanford-university"
PROJECT = "RL_Project_Data"
SWEEP_ID = "ri1jgd4l"  # 🔺 替换成你需要的sweepsID ,通过url连接可以看到id

# 2. 初始化 API
# 增加 timeout 防止无限期等待服务器响应
api = wandb.Api(timeout=60)

print(f"正在连接到 Sweep: {ENTITY}/{PROJECT}/{SWEEP_ID} ...")

try:
    sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
    # 注意：这里直接获取 runs 迭代器
    all_runs = sweep.runs
    
    summary_list, config_list, name_list = [], [], []

    print("开始拉取数据，这可能需要一点时间...")

    # 3. 遍历并打印进度，防止看起来像卡死
    for count, run in enumerate(all_runs):
        # 读取数据
        summary_list.append(run.summary._json_dict)
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
        name_list.append(run.name)
        
        # 每 50 条强制刷新一次进度到屏幕
        if (count + 1) % 50 == 0:
            print(f"已成功获取 {count + 1} 条记录...", flush=True)

    # 4. 构建 DataFrame
    if not name_list:
        print("警告：未找到任何运行记录！请检查 SWEEP_ID 是否正确。")
    else:
        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
        })

        print(f"数据拉取完毕！总计: {len(runs_df)} 条")

        # 5. 保存文件
        filename = "step4-grid.csv"
        runs_df.to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"文件已成功保存至: {filename}")

except Exception as e:
    print(f"运行过程中出错: {e}")
    sys.exit(1)

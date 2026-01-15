# CartPole_DQN / CartPole 强化学习实验工作流（DQN）

[English](#english-readme) | [中文](#chinese-readme)

<a name="english-readme"></a>
## English README

### 1) What this is
This repository is a Reinforcement Learning (RL) experimentation pipeline for **CartPole-v1** using **Deep Q-Networks (DQN)**.
It is designed for reproducible experiments and structured evaluation (precision analysis → sweep search → model selection → sensitivity analysis), with **Weights & Biases (W&B / wandb)** as the primary experiment tracking backend.

### 2) Tech stack
- **Language**: Python
- **Frameworks**: TensorFlow, Gymnasium
- **Config**: YAML
- **Experiment tracking**: Weights & Biases (wandb)

### 3) Important: secrets / W&B (required for most runs)
This project is tightly integrated with W&B.

If you do not want to use W&B, you will need to adjust the code/config to disable it. The default pipeline assumes W&B is available.

#### 3.1 Create `secrets.json`
- Create a file named `secrets.json` in the **repository root** (same level as `run.bat`).
- Minimal format:

```json
{
  "WANDB_API_KEY": "YOUR_WANDB_API_KEY_HERE"
}
```

#### 3.2 How to get the API key
1. Create/login to a W&B account.
2. Go to W&B settings → **API Keys**.
3. Copy your key into `secrets.json`.

#### 3.3 Security reminder (do not upload secrets)
- **Never commit** `secrets.json` to GitHub.
- This repo is configured to ignore secrets via `.gitignore`, but you should still double-check before pushing.
- If you need to share configuration, share a template (e.g., documentation), not the real key.

### 4) Project layout (core vs optional vs generated)

#### Core (required)
- `configs/`: YAML configuration files (training/sweep parameters)
- `src/`: core library code (agents, networks, replay buffer, env wrappers)
- `scripts/`: main executable pipeline scripts (Step1–Step4)
- `requirements.txt`: pinned dependencies

#### Optional (convenience)
- `tool/`: utility scripts (analysis, plotting, data conversion). Helpful but not required to run training.
- `notebooks/`: exploration notebooks (optional)

#### Generated / outputs (usually safe to delete)
- `data/`: experiment artifacts produced by the pipeline (depending on scripts)
- `logs/`, `wandb/`, `artifacts/`: runtime outputs / W&B local logs and caches
- `plot/`: generated figures

### 5) How to run
Typical flow:
1. Install dependencies: `pip install -r requirements.txt`
2. Create `secrets.json` (see Section 3)
3. Run pipeline scripts under `scripts/` (Step1 → Step4)

Recommended order (practical):
1. Start from the default entrypoint: `python scripts/main_cartpole_dqn.py`
2. Then run the staged pipeline (Step1A → Step1B → Step2 → Step3 → Step4)

Concrete entrypoints (recommended order):
- Default / main: `python scripts/main_cartpole_dqn.py`
- Step1A: `python scripts/step1A_cartpole_precision_analysis.py`
- Step1B: `python scripts/step1B_cartpole_presision_analysis.py`
- Step2: `python scripts/step2_cartpole_candidate_search.py`
- Step3 Controller: `python scripts/step3Controller_cartpole_model_selection.py`
- Step3 Worker: `python scripts/step3Worker_cartpole_model_selection.py`
- Step4: `python scripts/step4_cartpole_sensitivity_analysis.py`

Configs live in `configs/` (examples: `cartpole_dqn_defaults.yaml`, `grid_cartpole_dqn.yaml`, `bayes_cartpole_dqn.yaml`).

#### About `run.bat` and `worker.bat` (Windows helpers)
- These `.bat` files are **optional**.
- They are convenience wrappers for running multi-process / controller-worker style jobs on Windows.
- If you prefer, you can ignore them and run the Python scripts directly.

#### Controller–Worker note (Step 3)
- `scripts/step3Controller_cartpole_model_selection.py` is the main entry for the Step 3 architecture.
- The Worker script is **not** typically a standalone entrypoint: it is started/controlled by the Controller.
- In most cases you only need to run the Controller; it will orchestrate Worker processes as needed.
- Run the Worker manually only if you intentionally deploy Workers on separate machines/processes and know the expected arguments/environment.

### 6) Reports / results publication
Experiment reports (figures + analysis) are planned to be published on the author’s GitHub Pages / static blog.
The URL is not finalized yet; a link will be added later.

### 7) 30-second pre-release checklist
- Confirm there is **no** `secrets.json` committed/tracked, and there are no hardcoded API keys in code/config.
- Confirm `.gitignore` excludes secrets and common outputs (`wandb/`, `logs/`, `artifacts/`, `__pycache__/`, etc.).
- Delete generated outputs you don’t want to publish (`data/`, `plot/`, `wandb/` caches) before pushing.
- Quick sanity: run `python -m compileall .` to ensure no syntax errors.
- Optional: run Step1/Step2 on a small config to confirm the pipeline still starts.

---

<a name="chinese-readme"></a>
## 中文 README

### 1) 这是什么项目
这是一个用于 **CartPole-v1** 环境的强化学习实验工作流，核心算法为 **DQN (Deep Q-Networks)**。
项目强调可复现性与系统化实验流程（精度分析 → 候选搜索 → 模型筛选 → 敏感度分析），并且与 **Weights & Biases (W&B / wandb)** 深度耦合用于实验跟踪与可视化。

### 2) 技术栈
- **语言**：Python
- **框架**：TensorFlow, Gymnasium
- **配置**：YAML
- **实验跟踪**：Weights & Biases (wandb)

### 3) 重要：密钥 / W&B（多数运行需要）
本项目与 W&B 紧密耦合。

如果你不想使用 W&B，需要你自行在代码/配置层面关闭相关逻辑；默认工作流假设 W&B 可用。

#### 3.1 创建 `secrets.json`
- 在**仓库根目录**创建 `secrets.json`（与 `run.bat` 同级）。
- 最小格式如下：

```json
{
  "WANDB_API_KEY": "你的W&B_API_KEY"
}
```

#### 3.2 如何获取 API Key
1. 注册/登录 W&B。
2. 进入 W&B 设置页面 → **API Keys**。
3. 复制 key，填入 `secrets.json`。

#### 3.3 密钥管理提醒（不要上传到云端）
- **不要把** `secrets.json` 提交到 GitHub。
- 本项目通过 `.gitignore` 忽略密钥文件，但仍建议你 push 前再检查一遍。
- 分享项目时可以分享“格式模板/说明”，不要分享真实 key。

### 4) 目录结构（核心 / 可选 / 自动产出）

#### 核心（必须）
- `configs/`：YAML 配置（训练参数、扫描参数等）
- `src/`：核心库代码（agent、network、replay buffer、env 封装等）
- `scripts/`：实验工作流脚本（Step1–Step4）
- `requirements.txt`：依赖版本

#### 可选（工具性质）
- `tool/`：一些便利脚本（分析、绘图、格式转换）。可用可不用，不影响核心训练。
- `notebooks/`：笔记本（可选）

#### 自动生成/输出（通常可以删除）
- `data/`：脚本运行过程中生成的数据/中间产物
- `logs/`、`wandb/`、`artifacts/`：运行输出 / W&B 本地缓存
- `plot/`：生成的图表

### 5) 如何运行
典型流程：
1. 安装依赖：`pip install -r requirements.txt`
2. 创建 `secrets.json`（见第 3 节）
3. 运行 `scripts/` 下各阶段脚本（Step1 → Step4）

建议顺序（更符合实际使用）：
1. 先从默认主入口跑起来：`python scripts/main_cartpole_dqn.py`
2. 然后再按阶段跑完整工作流（Step1A → Step1B → Step2 → Step3 → Step4）

脚本入口（建议顺序）：
- 默认主入口：`python scripts/main_cartpole_dqn.py`
- Step1A：`python scripts/step1A_cartpole_precision_analysis.py`
- Step1B：`python scripts/step1B_cartpole_presision_analysis.py`
- Step2：`python scripts/step2_cartpole_candidate_search.py`
- Step3 Controller：`python scripts/step3Controller_cartpole_model_selection.py`
- Step3 Worker：`python scripts/step3Worker_cartpole_model_selection.py`
- Step4：`python scripts/step4_cartpole_sensitivity_analysis.py`

配置文件在 `configs/`（例如：`cartpole_dqn_defaults.yaml`, `grid_cartpole_dqn.yaml`, `bayes_cartpole_dqn.yaml`）。

#### 关于 `run.bat` 和 `worker.bat`（Windows 辅助脚本）
- 这两个 `.bat` 是**可选**的。
- 主要用于 Windows 下的多进程/Controller-Worker 管理。
- 不想用就忽略，直接运行 Python 脚本也可以。

#### Controller–Worker 说明（Step 3）
- Step3 的主入口是 `scripts/step3Controller_cartpole_model_selection.py`。
- Worker 脚本一般**不是**独立入口：它通常由 Controller 启动并进行控制。
- 大多数情况下只需要运行 Controller，它会按需拉起/管理 Worker 进程。
- 只有在你明确要把 Worker 部署到独立机器/独立进程，并清楚需要的参数与环境时，才建议手动运行 Worker。

### 6) 报告发布
实验报告（图表与分析）计划发布到作者个人 GitHub Pages / 静态博客。
目前尚未生成最终网址，后续会补充链接。

### 7) 发布前 30 秒检查清单
- 确认仓库内**没有**被提交/被追踪的 `secrets.json`，并且代码/配置里没有硬编码的 API key。
- 确认 `.gitignore` 已忽略密钥与常见输出目录（`wandb/`、`logs/`、`artifacts/`、`__pycache__/` 等）。
- 推送前删除不想公开的自动产出（如 `data/`、`plot/`、本地 `wandb/` 缓存等）。
- 快速自检：运行 `python -m compileall .`，确保没有语法错误。
- 可选：用一个小配置跑一下 Step1/Step2，确认工作流仍可启动。
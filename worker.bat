:: worker.bat
@echo off
:: 1. 强制内部编码为 UTF-8 (解决 python 读写乱码)
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

:: --- 设定WB离线/在线模式 ---或者在wband.init里加mode="offline"
:: 备注: 离线模式下不会上传数据到云端，只会保存在本地。
:: 如果需要联网上传，可以在联网时,在【平板服务器根目录cmd】运行 wandb sync 命令同步数据。命令如下：
    :: wandb sync --sync-all 【记得先登录+检查网络】如果你愿意，也可以指定某个文件夹同步，但通常这是不必要的
    :: dir wandb | findstr 1evhicuj 如果卡住在这个进程上，则可以在根目录下这样找到他，然后
    :: 会返回类似于wandb\offline-run-20251231_044909-b9zx5i0c
    :: wandb sync wandb/offline-run-20260101_065521-1evhicuj 继续这个命令就可以同步这个单独的对象
    :: 网格搜素用了sweeps功能，该功能不支持离线
set WANDB_MODE=online

:: 2. 设置 W&B 模式 (解决超时)
set WANDB_START_METHOD=thread

:: 3. 接收编号参数 (如果没有参数则默认为 1)
set "AGENT_ID=%1"
if "%AGENT_ID%"=="" set "AGENT_ID=1"

:: 4. 切换到脚本所在目录 (确保相对路径不出错)
cd /d "%~dp0"
set "PYTHONPATH=%CD%"

:: 5. 写入分隔符 (修改为对应编号的日志名)
echo. >> agent_%AGENT_ID%.log
echo ======================================================== >> agent_%AGENT_ID%.log
echo [WORKER %AGENT_ID% STARTED] %DATE% %TIME% >> agent_%AGENT_ID%.log
echo ======================================================== >> agent_%AGENT_ID%.log

:: 6. 正式运行指令库 (用哪个就取消哪行的 :: 注释，并确保其他行已注释)

:: [模式 A] 运行普通训练 (Step 1B 或 默认基线训练)
:: python -u -m scripts.main_cartpole_dqn --config cartpole_dqn_defaults.yaml >> agent_%AGENT_ID%.log 2>&1

:: [模式 B] 运行精度分析 (Step 1A - 需手动修改下方的模型 ID)
:: python -u -m scripts.step1A_cartpole_precision_analysis --model_artifact_name "你的用户/项目/model-ID:v版本" >> agent_%AGENT_ID%.log 2>&1

:: [模式 C] 运行 W&B Sweep Agent (Step 2 候选搜索 / Step 3 冠军验证 / Step 4 网格搜索)
:: 备注: Step 2/3/4 的区别仅在于你创建 Sweep 时用的 YAML 不同，Agent 命令是一样的
:: wandb agent foxmir-stanford-university/RL_Project_Data/a0yngt0o --count 80 >> agent_%AGENT_ID%.log 2>&1
:: wandb sweep configs/bayes_cartpole_dqn.yaml --project RL_Project_Data --entity foxmir-stanford-university

:: [模式 D] 运行模型选择控制器 (Step 3 Controller - 由它自己管控两个子进程 ）
:: python -u -m scripts.step3Controller_cartpole_model_selection >> agent_%AGENT_ID%.log 2>&1

:: [模式 E] 运行敏感性分析 (如果是用 Python 脚本跑循环而非 Sweep)
:: python -u -m scripts.step4_sensitivity_analysis >> agent_%AGENT_ID%.log 2>&1

:: [模式 F] 运行网格搜索 (Step 4 Grid Search - 需手动修改下方的配置文件名)
:: python -m tool.before_sweeps
:: set PYTHONUTF8=1 然后run.bat 1 , 2 分别开个窗口运行【该文件！！！】而不是下面这条命令！！🔺
:: wandb sweep configs/grid_cartpole_dqn.yaml --project RL_Project_Data --entity foxmir-stanford-university
wandb agent foxmir-stanford-university/RL_Project_Data/ri1jgd4l >> agent_%AGENT_ID%.log 2>&1

:: ========================================================
:: 第 1 步：SSH 连上平板，进入根目录。
:: 第 2 步：确保已登录（wandb login）。
:: 第 3 步：wb贝叶斯优化命令(可能需要set PYTHONUTF8=1 来先指定utf-8编码才能执行成功)---这一步无论贝叶斯或者网格搜素都一样
:: 低 4 步：拿到编号后修改本文档，然后依次开3个独立的cmd窗口处理以下命令
:: 第 3 步：发射 1 号卫星：输入 run.bat 1。系统提示 [REAL SUCCESS]。
:: 第 4 步：发射 2 号卫星：输入 run.bat 2。系统提示 [REAL SUCCESS]。
:: 第 5 步：发射 3 号卫星：输入 run.bat 3。系统提示 [REAL SUCCESS]。
:: 第 6 步：直接关闭 SSH 窗口，关机睡觉。

:: ========================================================
:: 检查cpu的频率(2000以上就是健康的)
:: wmic cpu get currentclockspeed
:: wmic cpu get currentclockspeed,maxclockspeed

:: 简易版任务管理器
:: powershell -command "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10"

:: 查询电脑温度,建议小于60，否则长期运行可能会降频甚至蓝屏
:: powershell "Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace 'root/wmi' | Select-Object InstanceName, @{N='Temp(C)';E={($_.CurrentTemperature - 2732)/10}}"

:: 检查电脑网络连接情况中是否有断线的情况记录，如果没有则返回空
:: wevtutil qe System /q:"*[System[(EventID=4204 or EventID=4205)]]" /f:text /c:100

:: ========================================================
:: ::: 平板维护指令库 (复制 -> 粘贴到终端 -> 回车) :::
::
:: [1. 监控类]
:: 查看实时日志(中文不乱码):
:: powershell -Command "chcp 65001; Get-Content agent_1.log -Wait -Tail 20"
::
:: 检查Python进程是否存活(看PID):
:: tasklist | findstr python
::
:: 检查网络是否通(测试Google):
:: curl -I https://www.google.com
:: crul -I https://wandb.ai
::
:: [2. 操作类]
:: 强制杀掉所有训练进程(停止任务):
:: taskkill /f /im python.exe
::
:: 远程重启平板电脑(慎用,会断连):
:: shutdown /r /t 0
::
:: [3. 调试类]
:: 查看最近的报错日志:
:: type agent_1.log
:: ========================================================

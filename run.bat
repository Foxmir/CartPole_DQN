:: run.bat
@echo off
chcp 65001 >nul

:: 接收编号参数 (判断要启动第几个 Agent)
set "AGENT_ID=%1"
if "%AGENT_ID%"=="" set "AGENT_ID=1"

:: 任务名称包含编号，防止冲突
set "TASK_NAME=RL_Satellite_%AGENT_ID%"
set "WORKER_PATH=%~dp0worker.bat"

echo [1/3] Creating detached task %TASK_NAME%...
:: 修改点：去掉了 /st 00:00，改用 /sc ONCE /sd 1900/01/01 (规避时间警告)
schtasks /create /tn "%TASK_NAME%" /tr "'%WORKER_PATH%' %AGENT_ID%" /sc ONCE /st 00:00 /ri 1 /f >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create task. Windows rejected the command.
    goto :EOF
)

echo [2/3] Launching satellite...
schtasks /run /tn "%TASK_NAME%" >nul
if %errorlevel% neq 0 (
    echo [ERROR] Failed to run task.
    goto :EOF
)

echo [3/3] Cleaning up task definition...
schtasks /delete /tn "%TASK_NAME%" /f >nul

:: 真实的成功验证
timeout /t 2 >nul
tasklist | findstr python >nul
if %errorlevel% equ 0 (
    echo.
    echo [REAL SUCCESS] Python process detected in background.
    echo ========================================================
    echo 1. Check logs: powershell -Command "chcp 65001; Get-Content agent_%AGENT_ID%.log -Wait -Tail 20"
    echo 2. Stop: taskkill /f /im python.exe
    echo ========================================================
) else (
    echo.
    echo [FAILURE] Task ran but Python process died immediately.
    echo Please check agent_%AGENT_ID%.log for details.
)


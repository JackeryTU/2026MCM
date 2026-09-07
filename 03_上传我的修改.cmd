@echo off
chcp 936 >nul
setlocal EnableExtensions DisableDelayedExpansion
title 2026MCM - 03_上传我的修改

set "REPO_URL=https://github.com/JackeryTU/2026MCM.git"
cd /d "%~dp0"

echo ============================================================
echo  保存并上传我在本机完成的修改
echo  注意：本脚本不会下载队友的新文件
echo ============================================================
echo.

call :check_repo
if errorlevel 1 goto :fail

git diff --check
if errorlevel 1 (
    echo [停止] Git 在上面列出的文件中发现了空白字符错误。
    echo 不确定如何处理时，请把完整窗口发给队长。
    goto :fail
)

set "JUST_COMMITTED=0"
git status --short
git status --porcelain | findstr . >nul
if errorlevel 1 goto :check_remote

echo.
set /p "CONFIRM=请检查上面的文件列表。确认保存全部修改请输入 Y："
if /i not "%CONFIRM%"=="Y" goto :cancel
set /p "COMMIT_MSG=请输入一句修改说明，例如 paper: 完成问题一模型描述："
if not defined COMMIT_MSG (
    echo [失败] 修改说明不能为空。
    goto :fail
)
rem 去掉双引号，避免用户输入的引号破坏 git commit 命令。
set "COMMIT_MSG=%COMMIT_MSG:"=%"
if not defined COMMIT_MSG (
    echo [失败] 修改说明不能只包含双引号。
    goto :fail
)

git add -A
if errorlevel 1 goto :git_fail
git commit -m "%COMMIT_MSG%"
if errorlevel 1 goto :git_fail
set "JUST_COMMITTED=1"

:check_remote
echo.
echo [检查] 正在确认队友是否已经上传了更新……
git fetch origin main
if errorlevel 1 goto :network_fail

for /f %%C in ('git rev-list --count HEAD..origin/main') do set "BEHIND_COUNT=%%C"
if not "%BEHIND_COUNT%"=="0" goto :behind

for /f %%C in ('git rev-list --count origin/main..HEAD') do set "AHEAD_COUNT=%%C"
if "%AHEAD_COUNT%"=="0" goto :already_synced
if "%JUST_COMMITTED%"=="1" goto :push_now

echo.
echo [等待上传] 下面这些本地版本尚未上传到 GitHub：
git log --oneline origin/main..HEAD
echo.
set /p "PUSH_CONFIRM=确认上传这些已有版本请输入 Y："
if /i not "%PUSH_CONFIRM%"=="Y" goto :cancel_push

:push_now
git push origin main
if errorlevel 1 goto :push_fail

echo.
echo [完成] 你的修改已经上传到 GitHub。
git status --short --branch
goto :success

:already_synced
echo [完成] 当前没有需要上传的新修改或本地版本。
goto :success

:behind
echo.
echo [停止] 队友已经先上传了更新。
echo 你的本地版本已经安全保存，不会丢失。
echo 请运行“04_获取队友最新文件”，成功后再运行一次“03_上传我的修改”。
goto :fail

:cancel_push
echo [已取消] 已有的本地版本没有上传，但仍安全保存在本机。
goto :success

:network_fail
echo [失败] 无法连接 GitHub。你的本地版本仍安全保存在本机，请在网络恢复后重试。
goto :fail

:push_fail
echo [失败] 上传被拒绝或网络中断。不要使用强制推送。
echo 请先运行“04_获取队友最新文件”；如有冲突，请联系队长；之后再上传。
goto :fail

:git_fail
echo [失败] Git 操作失败。请保留本窗口，并把完整报错发给队长。
goto :fail

:cancel
echo [已取消] 没有保存版本，也没有上传文件。
goto :success

:check_repo
where git >nul 2>nul
if errorlevel 1 (
    echo [失败] 没有检测到 Git。请先安装 Git for Windows。
    exit /b 1
)
if not exist ".git\" (
    echo [失败] 当前文件夹不是已经下载的 Git 项目。
    echo 队员第一次加入时，应先使用“02_队员首次下载项目.cmd”。
    exit /b 1
)
for /f "delims=" %%R in ('git remote get-url origin 2^>nul') do set "CURRENT_ORIGIN=%%R"
if not defined CURRENT_ORIGIN (
    echo [失败] 当前项目缺少 GitHub 仓库地址，请联系队长。
    exit /b 1
)
if /i not "%CURRENT_ORIGIN%"=="%REPO_URL%" (
    echo [停止] 当前文件夹连接的是另一个 GitHub 仓库：
    echo %CURRENT_ORIGIN%
    exit /b 1
)
if exist ".git\rebase-merge\" goto :unfinished_rebase
if exist ".git\rebase-apply\" goto :unfinished_rebase
git rev-parse -q --verify REBASE_HEAD >nul 2>nul
if not errorlevel 1 goto :unfinished_rebase
git rev-parse -q --verify MERGE_HEAD >nul 2>nul
if not errorlevel 1 goto :unfinished_merge
git rev-parse -q --verify CHERRY_PICK_HEAD >nul 2>nul
if not errorlevel 1 goto :unfinished_operation
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if /i not "%CURRENT_BRANCH%"=="main" (
    echo [停止] 当前不在 main 分支，不能使用本脚本上传。
    echo 当前分支：%CURRENT_BRANCH%
    echo 请联系队长检查，不要自行切换或强制推送。
    exit /b 1
)
git rev-parse -q --verify REBASE_HEAD >nul 2>nul
if not errorlevel 1 goto :unfinished_rebase
git rev-parse -q --verify MERGE_HEAD >nul 2>nul
if not errorlevel 1 goto :unfinished_merge
git rev-parse -q --verify CHERRY_PICK_HEAD >nul 2>nul
if not errorlevel 1 goto :unfinished_operation
exit /b 0

:unfinished_rebase
echo [停止] 上一次变基尚未处理完成，请联系队长处理。
exit /b 1

:unfinished_merge
echo [停止] 上一次合并尚未处理完成，请联系队长处理。
exit /b 1

:unfinished_operation
echo [停止] 上一次版本处理尚未完成，请联系队长处理。
exit /b 1

:fail
echo.
pause
exit /b 1

:success
echo.
pause
exit /b 0

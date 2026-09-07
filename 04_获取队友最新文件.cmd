@echo off
chcp 936 >nul
setlocal EnableExtensions DisableDelayedExpansion
title 2026MCM - 04_获取队友最新文件

set "REPO_URL=https://github.com/JackeryTU/2026MCM.git"
cd /d "%~dp0"

echo ============================================================
echo  从 GitHub 获取队友上传的最新文件
echo  注意：本脚本不会保存版本，也不会上传你的文件
echo ============================================================
echo.

call :check_repo
if errorlevel 1 goto :fail

git status --porcelain | findstr . >nul
if not errorlevel 1 (
    echo [停止] 你在本机还有尚未保存为版本的修改。
    echo 请先运行“03_上传我的修改”保存这些修改，再重新获取队友文件。
    goto :fail
)

git pull --rebase origin main
if errorlevel 1 goto :pull_fail

echo.
echo [完成] 当前文件夹已经是 GitHub 上的最新版本。
git status --short --branch
goto :success

:pull_fail
echo.
echo [失败] 获取失败，或者出现了需要人工处理的冲突。
echo 不要使用强制推送或强制重置。请保留本窗口，并把完整报错发给队长。
goto :fail

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
    echo [停止] 当前不在 main 分支，不能使用本脚本获取文件。
    echo 当前分支：%CURRENT_BRANCH%
    echo 请联系队长检查，不要自行切换或强制重置。
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

@echo off
chcp 936 >nul
setlocal EnableExtensions DisableDelayedExpansion
title 2026MCM - 01_队长首次发布项目

set "REPO_URL=https://github.com/JackeryTU/2026MCM.git"
cd /d "%~dp0"

echo ============================================================
echo  仅供队长使用：第一次把本项目发布到 GitHub
echo  当前文件夹：%CD%
echo ============================================================
echo.

where git >nul 2>nul
if errorlevel 1 goto :no_git

if exist ".git\" goto :retry_first_push

echo 本操作会初始化当前文件夹，并把其中的全部项目文件首次发布到 GitHub。
set /p "CONFIRM=确认由队长首次发布请输入 Y，取消请直接回车："
if /i not "%CONFIRM%"=="Y" goto :cancel

echo.
echo [检查] 正在检查 GitHub 仓库的当前文件，而不是只检查提交历史……
set "REMOTE_MODE="
git ls-remote --exit-code "%REPO_URL%" >nul 2>nul
set "REMOTE_RESULT=%ERRORLEVEL%"
if "%REMOTE_RESULT%"=="2" (
    set "REMOTE_MODE=NEW"
    goto :ask_identity
)
if not "%REMOTE_RESULT%"=="0" goto :remote_check_failed

git ls-remote --exit-code --heads "%REPO_URL%" refs/heads/main >nul 2>nul
set "MAIN_RESULT=%ERRORLEVEL%"
if not "%MAIN_RESULT%"=="0" goto :remote_exists

set "TEMP_CHECK=%TEMP%\2026MCM_remote_check_%RANDOM%_%RANDOM%"
if exist "%TEMP_CHECK%\" goto :remote_check_failed
git clone --quiet --no-checkout --branch main "%REPO_URL%" "%TEMP_CHECK%" >nul 2>nul
if errorlevel 1 goto :remote_clone_failed

git -C "%TEMP_CHECK%" ls-tree -r --name-only HEAD | findstr . >nul
set "REMOTE_HAS_FILES=%ERRORLEVEL%"
rmdir /s /q "%TEMP_CHECK%" >nul 2>nul
set "TEMP_CHECK="
if "%REMOTE_HAS_FILES%"=="0" goto :remote_exists

set "REMOTE_MODE=EMPTY_MAIN"
echo [通过] 远程 main 有历史记录，但当前版本没有文件。
echo        脚本会保留这些历史，再把本地模板作为新提交上传。

:ask_identity
set /p "GIT_NAME=请输入你的 GitHub 用户名："
if not defined GIT_NAME goto :missing_identity
set /p "GIT_EMAIL=请输入你的 GitHub 邮箱或隐私邮箱："
if not defined GIT_EMAIL goto :missing_identity

git init
if errorlevel 1 goto :git_fail
git remote add origin "%REPO_URL%"
if errorlevel 1 goto :git_fail
git config --local user.name "%GIT_NAME%"
if errorlevel 1 goto :git_fail
git config --local user.email "%GIT_EMAIL%"
if errorlevel 1 goto :git_fail
git config --local pull.rebase true
if errorlevel 1 goto :git_fail

if "%REMOTE_MODE%"=="EMPTY_MAIN" (
    git fetch origin main
    if errorlevel 1 goto :git_fail
    git checkout -B main origin/main
    if errorlevel 1 goto :git_fail
) else (
    git branch -M main
    if errorlevel 1 goto :git_fail
)

git add -A
if errorlevel 1 goto :git_fail
git diff --cached --quiet
if not errorlevel 1 goto :nothing_to_commit
git commit -m "docs: initialize modeling workspace"
if errorlevel 1 goto :git_fail
git push -u origin main
if errorlevel 1 goto :push_fail

echo.
echo [完成] 项目模板已经发布到 GitHub，并且没有覆盖远程历史。
echo 现在由队长在仓库设置中邀请队员，再让队员运行“02_队员首次下载项目.cmd”。
goto :success

:retry_first_push
echo [检查] 检测到本地 Git 仓库，正在判断是否为首次上传失败后的重试……
for /f "delims=" %%R in ('git remote get-url origin 2^>nul') do set "CURRENT_ORIGIN=%%R"
if not defined CURRENT_ORIGIN goto :existing_repo_stop
if /i not "%CURRENT_ORIGIN%"=="%REPO_URL%" goto :existing_repo_stop
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if /i not "%CURRENT_BRANCH%"=="main" goto :existing_repo_stop
git rev-parse --verify HEAD >nul 2>nul
if errorlevel 1 goto :existing_repo_stop
git status --porcelain | findstr . >nul
if not errorlevel 1 goto :existing_repo_dirty

git ls-remote --exit-code --heads "%REPO_URL%" refs/heads/main >nul 2>nul
set "RETRY_REMOTE_MAIN=%ERRORLEVEL%"
if not "%RETRY_REMOTE_MAIN%"=="0" goto :retry_push_confirm
git fetch origin main
if errorlevel 1 goto :remote_check_failed
for /f "delims=" %%L in ('git rev-parse HEAD 2^>nul') do set "LOCAL_HEAD=%%L"
for /f "delims=" %%R in ('git rev-parse origin/main 2^>nul') do set "REMOTE_HEAD=%%R"
if not defined LOCAL_HEAD goto :existing_repo_stop
if not defined REMOTE_HEAD goto :remote_check_failed
if /i "%LOCAL_HEAD%"=="%REMOTE_HEAD%" goto :already_published
git merge-base --is-ancestor origin/main HEAD >nul 2>nul
if errorlevel 1 goto :remote_diverged

:retry_push_confirm
echo.
echo [恢复] 本地已有首次提交，但尚未成功建立远程跟踪关系。
echo        将执行普通推送；如果 GitHub 已有更新，Git 会拒绝，不会强制覆盖。
set /p "RETRY_CONFIRM=确认重试上传请输入 Y，取消请直接回车："
if /i not "%RETRY_CONFIRM%"=="Y" goto :cancel
git push -u origin main
if errorlevel 1 goto :push_fail
echo.
echo [完成] 首次提交已经成功上传到 GitHub。
goto :success

:already_published
echo.
echo [完成] 本地首次提交已经存在于 GitHub，无需重复上传。
echo 日常协作请改用“03_上传我的修改”或“04_获取队友最新文件”。
goto :success

:remote_diverged
echo.
echo [停止] GitHub main 与本地首次提交已经出现不同更新，脚本不会覆盖任何一方。
echo 请保留当前文件夹，并联系队长检查后再决定如何合并。
goto :fail

:existing_repo_dirty
echo.
echo [停止] 本地仓库还有未提交的文件变化，不能自动重试首次上传。
echo 请保留当前文件并联系队长检查。
goto :fail

:existing_repo_stop
echo.
echo [停止] 当前文件夹已经是正常使用中的 Git 仓库，不应再运行 01。
echo 日常协作请使用“03_上传我的修改”或“04_获取队友最新文件”。
goto :fail

:remote_exists
echo.
echo [停止] GitHub 仓库当前版本确实存在文件，不能执行首次发布。
echo 如果这些文件需要保留，请使用“04_获取队友最新文件.cmd”合并协作。
echo 如果确认仓库应当为空，请先在 GitHub 检查 main 分支当前文件列表。
goto :fail

:remote_clone_failed
if defined TEMP_CHECK if exist "%TEMP_CHECK%\" rmdir /s /q "%TEMP_CHECK%" >nul 2>nul
set "TEMP_CHECK="
goto :remote_check_failed

:remote_check_failed
echo.
echo [停止] 无法完整检查 GitHub 仓库。请检查网络、仓库地址、登录账号和访问权限。
echo 脚本尚未初始化当前文件夹，也没有上传任何文件。
goto :fail

:no_git
echo [失败] 没有检测到 Git，请先安装 Git for Windows：
echo https://git-scm.com/download/win
goto :fail

:missing_identity
echo [失败] GitHub 用户名和邮箱都不能为空。
goto :fail

:nothing_to_commit
echo [失败] 当前文件夹中没有可以首次发布的文件。
goto :fail

:push_fail
echo.
echo [失败] 普通上传失败，没有执行强制覆盖。
echo 请检查 GitHub 登录状态、仓库权限和网络；保留本文件夹后可再次运行 01 重试。
goto :fail

:git_fail
echo.
echo [失败] Git 操作失败。请保留窗口内容，并把完整截图发给队长。
goto :fail

:cancel
echo [已取消] 没有进行任何修改。
goto :success

:fail
echo.
pause
exit /b 1

:success
echo.
pause
exit /b 0

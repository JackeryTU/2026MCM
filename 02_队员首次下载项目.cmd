@echo off
chcp 936 >nul
setlocal EnableExtensions DisableDelayedExpansion
title 2026MCM - 02_队员首次下载项目

set "REPO_URL=https://github.com/JackeryTU/2026MCM.git"
set "TARGET_DIR=%~dp02026MCM"
cd /d "%~dp0"

echo ============================================================
echo  供队员第一次加入项目时使用
echo  项目将下载到：%TARGET_DIR%
echo ============================================================
echo.

where git >nul 2>nul
if errorlevel 1 goto :no_git

if exist "%TARGET_DIR%" (
    echo [停止] 目标文件夹已经存在：
    echo %TARGET_DIR%
    echo 请把本脚本放到另一个空的父文件夹中，或者联系队长处理。
    goto :fail
)

echo 本操作会把私有仓库下载到新建的 2026MCM 文件夹中。
set /p "CONFIRM=确认第一次下载项目请输入 Y，取消请直接回车："
if /i not "%CONFIRM%"=="Y" goto :cancel

git clone --branch main --single-branch "%REPO_URL%" "%TARGET_DIR%"
if errorlevel 1 goto :clone_fail

set /p "GIT_NAME=请输入你的 GitHub 用户名："
if not defined GIT_NAME goto :identity_fail
set /p "GIT_EMAIL=请输入你的 GitHub 邮箱或隐私邮箱："
if not defined GIT_EMAIL goto :identity_fail

git -C "%TARGET_DIR%" config --local user.name "%GIT_NAME%"
if errorlevel 1 goto :git_fail
git -C "%TARGET_DIR%" config --local user.email "%GIT_EMAIL%"
if errorlevel 1 goto :git_fail
git -C "%TARGET_DIR%" config --local pull.rebase true
if errorlevel 1 goto :git_fail

echo.
echo [完成] 项目下载和个人 Git 身份配置已经完成。
echo 今后请只在下面的文件夹中工作：
echo %TARGET_DIR%
echo 开始工作前运行“04_获取队友最新文件”，完成修改后运行“03_上传我的修改”。
goto :success

:no_git
echo [失败] 没有检测到 Git。请先安装 Git for Windows：
echo https://git-scm.com/download/win
goto :fail

:clone_fail
echo.
echo [失败] 项目下载失败。请依次确认：
echo 1. 队长已经邀请你的 GitHub 账号加入私有仓库；
echo 2. 你已经接受邀请，并登录了被邀请的账号；
echo 3. 当前网络能够访问 GitHub。
goto :fail

:identity_fail
echo.
echo [失败] 项目已经下载，但用户名或邮箱为空。
echo 请打开 2026MCM 文件夹，并联系队长帮助完成 Git 身份配置。
goto :fail

:git_fail
echo.
echo [失败] 个人 Git 身份配置失败。请把完整报错发给队长。
goto :fail

:cancel
echo [已取消] 没有下载任何文件。
goto :success

:fail
echo.
pause
exit /b 1

:success
echo.
pause
exit /b 0

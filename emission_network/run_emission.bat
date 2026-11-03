@echo off
REM ============================================================
REM  船舶排放网络（月度/城市节点, 年±1 边界月处理）一键运行
REM  用法：改下面的 YEAR / MONTHS / CORES 后双击本文件
REM  - YEAR   : 自然年, 如 2020
REM  - MONTHS : 留空=全年12月; 或 1,2,3,4,5,6 只输出部分月
REM  - TYPES  : 留空=全4类商船; 或 Container 只跑某一类
REM  断点续传默认开（已完成的"年×船型"自动跳过）
REM ============================================================
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONFAULTHANDLER=1

set YEAR=2020
set MONTHS=1,2,3,4,5,6
set TYPES=
set CORES=8
set DBDIR=E:\SQL_AIS_SingleShip
set OUT=E:\transfer justice\results_test

set OPT=
if not "%MONTHS%"=="" set OPT=%OPT% --only-months %MONTHS%
if not "%TYPES%"==""  set OPT=%OPT% --only-ship-types %TYPES%

echo 运行: 年=%YEAR%  月=%MONTHS%  船型=%TYPES%  核=%CORES%
echo 输出: %OUT%
python emission_network_main.py --year %YEAR% %OPT% --db-dir "%DBDIR%" --results-root "%OUT%" --n-cores %CORES%

echo.
echo ===== 运行结束, 按任意键关闭 =====
pause >nul

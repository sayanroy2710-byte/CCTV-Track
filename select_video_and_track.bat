@echo off
echo ======================================================================
echo      AI CCTV Tracking - Graphical Video Selector Launcher
echo ======================================================================
cd /d "%~dp0"
call .\C_Yolo\Scripts\activate.bat
python main.py --gui
pause

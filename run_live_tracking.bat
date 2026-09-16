@echo off
echo ======================================================================
echo       Launching Martian CCTV Multi-Camera People Tracking System
echo ======================================================================
cd /d "%~dp0"
call .\C_Yolo\Scripts\activate.bat
python main.py
pause
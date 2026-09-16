@echo off
echo ======================================================================
echo       Launching Martian CCTV Web Administrator Portal (Streamlit)
echo ======================================================================
cd /d "%~dp0"
call .\C_Yolo\Scripts\activate.bat
streamlit run web_dashboard.py
pause
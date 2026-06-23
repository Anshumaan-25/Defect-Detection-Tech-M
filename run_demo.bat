@echo off
REM ============================================================
REM   Defect Detection - Live Demo launcher
REM   Double-click this file to start the web demo.
REM ============================================================
cd /d "%~dp0"

echo ============================================================
echo   Defect Detection - Live Demo
echo ------------------------------------------------------------
echo   Loading models (PatchCore + YOLO + OCR)...
echo   This takes ~15-20 seconds on first start.
echo.
echo   When you see "Running on local URL", open this in a browser:
echo       http://127.0.0.1:7860
echo.
echo   To share on the same WiFi, others open http://[your-laptop-IP]:7860
echo       (find your IP by running:  ipconfig )
echo ------------------------------------------------------------
echo   Keep this window open during the demo. Close it (or press
echo   Ctrl+C) to stop the server when you are done.
echo ============================================================
echo.

".venv\Scripts\python.exe" app.py

echo.
echo Server stopped. Press any key to close this window.
pause >nul

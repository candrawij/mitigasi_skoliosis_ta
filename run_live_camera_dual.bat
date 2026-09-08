@echo off
title Mitigasi Skoliosis - Live Dual-Camera Posture Detection
echo ======================================================================
echo   DETEKSI POSTUR DUAL-CAMERA REAL-TIME (FRONTAL + LATERAL)
echo ======================================================================
echo.
if "%1"=="" (
    python 04_scripts/inference/infer_realtime_2d.py --select
) else (
    python 04_scripts/inference/infer_realtime_2d.py %*
)
pause

@echo off
title Mitigasi Skoliosis - Live Camera Posture Detection
echo ======================================================================
echo   DETEKSI POSTUR REAL-TIME (MODEL DATASET PRIVAT - 24 SUBJEK)
echo ======================================================================
echo.
if "%1"=="" (
    python 04_scripts/inference/infer_realtime_2d.py --select
) else (
    python 04_scripts/inference/infer_realtime_2d.py %*
)
pause

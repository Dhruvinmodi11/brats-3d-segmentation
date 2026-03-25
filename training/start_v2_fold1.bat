@echo off
title V2 UNet3DAttnV2 - Fold 1/5 (200 epochs) + BDL + DSD + GradCkpt
cd /d E:\data\training
echo ============================================================
echo  V2 UNet3DAttnV2 (base=24) - Fold 1/5
echo  200 epochs, Deep Supervision + Boundary Loss (GPU EDT)
echo  Batch=3 Accum=3 (eff=9) ValBatch=4
echo  Cosine Annealing, Early Stop Patience 25
echo  Mixed BraTS 2020+2023 dataset
echo ============================================================
echo.
E:\data\.venv\Scripts\python.exe runs\v2_5fold.py --fold 1
echo.
echo Training finished (exit code: %ERRORLEVEL%).
pause

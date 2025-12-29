@echo off
setlocal enabledelayedexpansion

REM Set console to UTF-8 for correct display of Russian text
chcp 65001 >nul

REM Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Create folders if they don't exist
mkdir models reports 2>nul

REM Step 1: Training the model
echo Step 1: Training the model...
call env\Scripts\python ml_pipelines\training\train_model.py --data data\train.csv --output models\model_sklearn.pkl --report reports\classification_report_sklearn.txt
if errorlevel 1 (
    echo Error: Failed to train the model. Check the data and code.
    pause
    exit /b 1
)
echo Step 1 completed. Press Enter to continue to Step 2.
pause >nul

REM Step 2: Converting to ONNX
echo Step 2: Converting to ONNX...
call env\Scripts\python ml_pipelines\optimizations\onnx_conversion.py --model models\model_sklearn.pkl --onnx models\default_model.onnx --data data\train.csv
if errorlevel 1 (
    echo Error: Failed to convert to ONNX.
    pause
    exit /b 1
)
echo Step 2 completed. Press Enter to continue to Step 3.
pause >nul

REM Step 3: Optimizing ONNX (skip if module not installed or error occurs)
echo Step 3: Optimizing ONNX...
REM Run the script without redirection to see output
call env\Scripts\python ml_pipelines\optimizations\optimize_onnx.py --input_model models\default_model.onnx --output_model models\default_model_int.onnx
REM Check if the output model was created (better check than errorlevel)
if exist models\default_model_int.onnx (
    echo ONNX optimization completed successfully.
    set ONNX_MODEL=models\default_model_int.onnx
) else (
    echo Warning: ONNX optimization failed or model not created. Using non-optimized model.
    set ONNX_MODEL=models\default_model.onnx
)
echo Step 3 completed. Press Enter to continue to Step 4.
pause >nul

REM Step 4: Benchmarking (use test.csv if exists, otherwise train.csv)
echo Step 4: Benchmarking...
if exist data\test.csv (
    set DATA_FILE=data\test.csv
) else (
    set DATA_FILE=data\train.csv
)
call env\Scripts\python ml_pipelines\inference\inference.py --data %DATA_FILE% --onnx %ONNX_MODEL% --report-dir reports/
if errorlevel 1 (
    echo Error: Failed to perform benchmarking.
    pause
    exit /b 1
)

echo All scripts executed. Reports in reports/
pause
endlocal

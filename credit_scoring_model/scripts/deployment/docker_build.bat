@echo off
REM Скрипт для сборки Docker образов для ML пайплайна (Windows)

setlocal enabledelayedexpansion

REM Параметры
set IMAGE_NAME=credit-scoring-ml
set TAG=%1
if "%TAG%"=="" set TAG=latest
set DOCKERFILE=%2
if "%DOCKERFILE%"=="" set DOCKERFILE=Dockerfile.ml

echo [%date% %time%] Начинаем сборку Docker образа для ML пайплайна
echo Образ: %IMAGE_NAME%:%TAG%
echo Dockerfile: %DOCKERFILE%

REM Проверяем наличие Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: Docker не установлен. Пожалуйста, установите Docker.
    exit /b 1
)

REM Проверяем наличие Dockerfile
if not exist "%DOCKERFILE%" (
    echo [%date% %time%] ERROR: Dockerfile не найден: %DOCKERFILE%
    exit /b 1
)

REM Собираем образ
echo [%date% %time%] Сборка Docker образа...
docker build -f "%DOCKERFILE%" -t "%IMAGE_NAME%:%TAG%" .

if errorlevel 1 (
    echo [%date% %time%] ERROR: Ошибка при сборке Docker образа
    exit /b 1
)

echo [%date% %time%] Docker образ успешно собран: %IMAGE_NAME%:%TAG%

REM Показываем информацию об образе
echo [%date% %time%] Информация об образе:
docker images "%IMAGE_NAME%:%TAG%"

echo [%date% %time%] Сборка завершена успешно!
pause

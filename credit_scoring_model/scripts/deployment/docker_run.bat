@echo off
REM Скрипт для запуска ML пайплайна в Docker контейнере (Windows)

setlocal enabledelayedexpansion

REM Параметры по умолчанию
set IMAGE_NAME=credit-scoring-ml
set TAG=latest
set CONTAINER_NAME=credit-scoring-pipeline
set DATA_VOLUME=credit-scoring-data
set MODELS_VOLUME=credit-scoring-models
set INTERACTIVE=false
set STEPS=""

REM Парсинг аргументов
:parse_args
if "%1"=="" goto :run_container
if "%1"=="-i" (
    set IMAGE_NAME=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--image" (
    set IMAGE_NAME=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-t" (
    set TAG=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--tag" (
    set TAG=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-n" (
    set CONTAINER_NAME=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--name" (
    set CONTAINER_NAME=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-d" (
    set DATA_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--data-dir" (
    set DATA_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="-m" (
    set MODELS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--models-dir" (
    set MODELS_DIR=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--interactive" (
    set INTERACTIVE=true
    shift
    goto :parse_args
)
if "%1"=="--steps" (
    set STEPS=%2
    shift
    shift
    goto :parse_args
)
if "%1"=="--help" (
    echo Использование: %0 [ОПЦИИ]
    echo.
    echo Опции:
    echo   -i, --image IMAGE     Имя Docker образа (по умолчанию: %IMAGE_NAME%)
    echo   -t, --tag TAG         Тег образа (по умолчанию: %TAG%)
    echo   -n, --name NAME       Имя контейнера (по умолчанию: %CONTAINER_NAME%)
    echo   -d, --data-dir DIR    Локальная папка с данными
    echo   -m, --models-dir DIR  Локальная папка для моделей
    echo   --interactive         Интерактивный режим
    echo   --steps STEPS         Шаги для выполнения
    echo   --help                Показать эту справку
    echo.
    echo Примеры:
    echo   %0 --data-dir .\data --models-dir .\models
    echo   %0 --interactive --steps eda,preprocessing
    echo   %0 --image my-ml-image --tag v1.0
    exit /b 0
)
shift
goto :parse_args

:run_container
echo [%date% %time%] Запуск ML пайплайна в Docker контейнере
echo Образ: %IMAGE_NAME%:%TAG%
echo Контейнер: %CONTAINER_NAME%

REM Проверяем наличие Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: Docker не установлен. Пожалуйста, установите Docker.
    exit /b 1
)

REM Проверяем наличие образа
docker images "%IMAGE_NAME%:%TAG%" >nul 2>&1
if errorlevel 1 (
    echo [%date% %time%] ERROR: Docker образ не найден: %IMAGE_NAME%:%TAG%
    echo Пожалуйста, сначала соберите образ с помощью docker_build.bat
    exit /b 1
)

REM Останавливаем существующий контейнер, если он запущен
docker ps -q -f name="%CONTAINER_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [%date% %time%] Останавливаем существующий контейнер...
    docker stop "%CONTAINER_NAME%"
)

REM Удаляем существующий контейнер
docker ps -aq -f name="%CONTAINER_NAME%" >nul 2>&1
if not errorlevel 1 (
    echo [%date% %time%] Удаляем существующий контейнер...
    docker rm "%CONTAINER_NAME%"
)

REM Создаем Docker volumes для данных и моделей
echo [%date% %time%] Создание Docker volumes...

REM Создаем volume для данных
docker volume ls | findstr "%DATA_VOLUME%" >nul 2>&1
if errorlevel 1 (
    docker volume create "%DATA_VOLUME%"
    echo [%date% %time%] Создан volume для данных: %DATA_VOLUME%
)

REM Создаем volume для моделей
docker volume ls | findstr "%MODELS_VOLUME%" >nul 2>&1
if errorlevel 1 (
    docker volume create "%MODELS_VOLUME%"
    echo [%date% %time%] Создан volume для моделей: %MODELS_VOLUME%
)

REM Подготавливаем команду запуска
set DOCKER_CMD=docker run --name %CONTAINER_NAME%

REM Добавляем volumes
if defined DATA_DIR (
    if exist "%DATA_DIR%" (
        set DOCKER_CMD=!DOCKER_CMD! -v "%DATA_DIR%":/app/data
        echo [%date% %time%] Монтируем локальную папку данных: %DATA_DIR%
    ) else (
        set DOCKER_CMD=!DOCKER_CMD! -v %DATA_VOLUME%:/app/data
        echo [%date% %time%] Используем Docker volume для данных: %DATA_VOLUME%
    )
) else (
    set DOCKER_CMD=!DOCKER_CMD! -v %DATA_VOLUME%:/app/data
    echo [%date% %time%] Используем Docker volume для данных: %DATA_VOLUME%
)

if defined MODELS_DIR (
    if exist "%MODELS_DIR%" (
        set DOCKER_CMD=!DOCKER_CMD! -v "%MODELS_DIR%":/app/models
        echo [%date% %time%] Монтируем локальную папку моделей: %MODELS_DIR%
    ) else (
        set DOCKER_CMD=!DOCKER_CMD! -v %MODELS_VOLUME%:/app/models
        echo [%date% %time%] Используем Docker volume для моделей: %MODELS_VOLUME%
    )
) else (
    set DOCKER_CMD=!DOCKER_CMD! -v %MODELS_VOLUME%:/app/models
    echo [%date% %time%] Используем Docker volume для моделей: %MODELS_VOLUME%
)

REM Добавляем переменные окружения
set DOCKER_CMD=!DOCKER_CMD! -e PYTHONPATH=/app -e PYTHONUNBUFFERED=1

REM Добавляем режим запуска
if "%INTERACTIVE%"=="true" (
    set DOCKER_CMD=!DOCKER_CMD! -it
    echo [%date% %time%] Запуск в интерактивном режиме
) else (
    set DOCKER_CMD=!DOCKER_CMD! --rm
    echo [%date% %time%] Запуск в фоновом режиме
)

REM Добавляем команду
if not "%STEPS%"=="" (
    set DOCKER_CMD=!DOCKER_CMD! %IMAGE_NAME%:%TAG% python scripts/run_pipeline.py --steps %STEPS%
    echo [%date% %time%] Выполнение шагов: %STEPS%
) else (
    set DOCKER_CMD=!DOCKER_CMD! %IMAGE_NAME%:%TAG%
    echo [%date% %time%] Выполнение полного пайплайна
)

REM Запускаем контейнер
echo [%date% %time%] Запуск контейнера...
echo Команда: !DOCKER_CMD!

!DOCKER_CMD!

if errorlevel 1 (
    echo [%date% %time%] ERROR: Ошибка при выполнении ML пайплайна
    exit /b 1
) else (
    echo [%date% %time%] ML пайплайн выполнен успешно!
    
    if "%INTERACTIVE%"=="false" (
        echo [%date% %time%] Результаты сохранены в Docker volumes:
        echo   Данные: %DATA_VOLUME%
        echo   Модели: %MODELS_VOLUME%
        echo.
        echo [%date% %time%] Для получения результатов выполните:
        echo   docker run --rm -v %DATA_VOLUME%:/data -v %MODELS_VOLUME%:/models alpine ls -la /data /models
    )
)

pause

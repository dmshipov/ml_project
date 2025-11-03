#!/bin/bash

# Скрипт для запуска ML пайплайна в Docker контейнере

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Параметры по умолчанию
IMAGE_NAME="credit-scoring-ml"
TAG="latest"
CONTAINER_NAME="credit-scoring-pipeline"
DATA_VOLUME="credit-scoring-data"
MODELS_VOLUME="credit-scoring-models"

# Функция для отображения справки
show_help() {
    echo "Использование: $0 [ОПЦИИ]"
    echo ""
    echo "Опции:"
    echo "  -i, --image IMAGE     Имя Docker образа (по умолчанию: ${IMAGE_NAME})"
    echo "  -t, --tag TAG         Тег образа (по умолчанию: ${TAG})"
    echo "  -n, --name NAME       Имя контейнера (по умолчанию: ${CONTAINER_NAME})"
    echo "  -d, --data-dir DIR    Локальная папка с данными"
    echo "  -m, --models-dir DIR  Локальная папка для моделей"
    echo "  --interactive         Интерактивный режим"
    echo "  --steps STEPS         Шаги для выполнения (eda,preprocessing,training,tuning,validation)"
    echo "  --help                Показать эту справку"
    echo ""
    echo "Примеры:"
    echo "  $0 --data-dir ./data --models-dir ./models"
    echo "  $0 --interactive --steps eda,preprocessing"
    echo "  $0 --image my-ml-image --tag v1.0"
}

# Парсинг аргументов
DATA_DIR=""
MODELS_DIR=""
INTERACTIVE=false
STEPS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -n|--name)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -m|--models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --interactive)
            INTERACTIVE=true
            shift
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Неизвестная опция: $1"
            show_help
            exit 1
            ;;
    esac
done

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    error "Docker не установлен. Пожалуйста, установите Docker."
    exit 1
fi

# Проверяем наличие образа
if ! docker images "${IMAGE_NAME}:${TAG}" &> /dev/null; then
    error "Docker образ не найден: ${IMAGE_NAME}:${TAG}"
    log "Пожалуйста, сначала соберите образ с помощью docker_build.sh"
    exit 1
fi

log "Запуск ML пайплайна в Docker контейнере"
log "Образ: ${IMAGE_NAME}:${TAG}"
log "Контейнер: ${CONTAINER_NAME}"

# Останавливаем существующий контейнер, если он запущен
if docker ps -q -f name="${CONTAINER_NAME}" | grep -q .; then
    log "Останавливаем существующий контейнер..."
    docker stop "${CONTAINER_NAME}"
fi

# Удаляем существующий контейнер
if docker ps -aq -f name="${CONTAINER_NAME}" | grep -q .; then
    log "Удаляем существующий контейнер..."
    docker rm "${CONTAINER_NAME}"
fi

# Создаем Docker volumes для данных и моделей
log "Создание Docker volumes..."

# Создаем volume для данных
if ! docker volume ls | grep -q "${DATA_VOLUME}"; then
    docker volume create "${DATA_VOLUME}"
    log "Создан volume для данных: ${DATA_VOLUME}"
fi

# Создаем volume для моделей
if ! docker volume ls | grep -q "${MODELS_VOLUME}"; then
    docker volume create "${MODELS_VOLUME}"
    log "Создан volume для моделей: ${MODELS_VOLUME}"
fi

# Подготавливаем команду запуска
DOCKER_CMD="docker run"

# Добавляем имя контейнера
DOCKER_CMD="${DOCKER_CMD} --name ${CONTAINER_NAME}"

# Добавляем volumes
if [ -n "${DATA_DIR}" ] && [ -d "${DATA_DIR}" ]; then
    DOCKER_CMD="${DOCKER_CMD} -v ${DATA_DIR}:/app/data"
    log "Монтируем локальную папку данных: ${DATA_DIR}"
else
    DOCKER_CMD="${DOCKER_CMD} -v ${DATA_VOLUME}:/app/data"
    log "Используем Docker volume для данных: ${DATA_VOLUME}"
fi

if [ -n "${MODELS_DIR}" ] && [ -d "${MODELS_DIR}" ]; then
    DOCKER_CMD="${DOCKER_CMD} -v ${MODELS_DIR}:/app/models"
    log "Монтируем локальную папку моделей: ${MODELS_DIR}"
else
    DOCKER_CMD="${DOCKER_CMD} -v ${MODELS_VOLUME}:/app/models"
    log "Используем Docker volume для моделей: ${MODELS_VOLUME}"
fi

# Добавляем переменные окружения
DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=/app"
DOCKER_CMD="${DOCKER_CMD} -e PYTHONUNBUFFERED=1"

# Добавляем режим запуска
if [ "${INTERACTIVE}" = true ]; then
    DOCKER_CMD="${DOCKER_CMD} -it"
    log "Запуск в интерактивном режиме"
else
    DOCKER_CMD="${DOCKER_CMD} --rm"
    log "Запуск в фоновом режиме"
fi

# Добавляем команду
if [ -n "${STEPS}" ]; then
    DOCKER_CMD="${DOCKER_CMD} ${IMAGE_NAME}:${TAG} python scripts/run_pipeline.py --steps ${STEPS}"
    log "Выполнение шагов: ${STEPS}"
else
    DOCKER_CMD="${DOCKER_CMD} ${IMAGE_NAME}:${TAG}"
    log "Выполнение полного пайплайна"
fi

# Запускаем контейнер
log "Запуск контейнера..."
log "Команда: ${DOCKER_CMD}"

eval "${DOCKER_CMD}"

if [ $? -eq 0 ]; then
    log "ML пайплайн выполнен успешно!"
    
    if [ "${INTERACTIVE}" = false ]; then
        log "Результаты сохранены в Docker volumes:"
        log "  Данные: ${DATA_VOLUME}"
        log "  Модели: ${MODELS_VOLUME}"
        
        # Показываем как получить результаты
        log ""
        log "Для получения результатов выполните:"
        log "  docker run --rm -v ${DATA_VOLUME}:/data -v ${MODELS_VOLUME}:/models alpine ls -la /data /models"
    fi
else
    error "Ошибка при выполнении ML пайплайна"
    exit 1
fi

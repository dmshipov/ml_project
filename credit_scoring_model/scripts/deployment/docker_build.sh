#!/bin/bash

# Скрипт для сборки Docker образов для ML пайплайна

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

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    error "Docker не установлен. Пожалуйста, установите Docker."
    exit 1
fi

# Параметры
IMAGE_NAME="credit-scoring-ml"
TAG=${1:-"latest"}
DOCKERFILE=${2:-"Dockerfile.ml"}

log "Начинаем сборку Docker образа для ML пайплайна"
log "Образ: ${IMAGE_NAME}:${TAG}"
log "Dockerfile: ${DOCKERFILE}"

# Проверяем наличие Dockerfile
if [ ! -f "${DOCKERFILE}" ]; then
    error "Dockerfile не найден: ${DOCKERFILE}"
    exit 1
fi

# Собираем образ
log "Сборка Docker образа..."
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}:${TAG}" .

if [ $? -eq 0 ]; then
    log "Docker образ успешно собран: ${IMAGE_NAME}:${TAG}"
    
    # Показываем информацию об образе
    log "Информация об образе:"
    docker images "${IMAGE_NAME}:${TAG}"
    
    # Показываем размер образа
    SIZE=$(docker images --format "table {{.Size}}" "${IMAGE_NAME}:${TAG}" | tail -n 1)
    log "Размер образа: ${SIZE}"
    
    # Опционально: запускаем тест
    if [ "${3}" = "--test" ]; then
        log "Запуск тестового контейнера..."
        docker run --rm "${IMAGE_NAME}:${TAG}" python -c "print('ML пайплайн готов к работе!')"
    fi
    
else
    error "Ошибка при сборке Docker образа"
    exit 1
fi

log "Сборка завершена успешно!"

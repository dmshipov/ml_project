#!/bin/bash

# Credit Scoring System Deployment Script
# This script handles the deployment of the credit scoring system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="credit-scoring"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENVIRONMENT=${1:-development}

echo -e "${GREEN}Запуск развертывания системы кредитного скоринга${NC}"
echo -e "${YELLOW}Окружение: ${ENVIRONMENT}${NC}"

# Функция для вывода цветного текста
print_status() {
    echo -e "${GREEN}[OK] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

# Проверить, запущен ли Docker
if ! docker info > /dev/null 2>&1; then
    print_error "Docker не запущен. Запустите Docker и попробуйте снова."
    exit 1
fi

# Проверить доступность Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose не установлен. Установите Docker Compose и попробуйте снова."
    exit 1
fi

# Создать необходимые директории
print_status "Создание необходимых директорий..."
mkdir -p data/postgres
mkdir -p logs
mkdir -p models/trained
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/provisioning

# Установить правильные права доступа
chmod 755 data/postgres
chmod 755 logs
chmod 755 models/trained

# Скопировать файл окружения, если он не существует
if [ ! -f .env ]; then
    print_warning "Файл окружения не найден. Создание из шаблона..."
    cp env.example .env
    print_warning "Отредактируйте файл .env с вашей конфигурацией перед продолжением."
    read -p "Нажмите Enter для продолжения после редактирования файла .env..."
fi

# Собрать и запустить сервисы
print_status "Сборка и запуск сервисов..."
docker-compose -f ${DOCKER_COMPOSE_FILE} down --remove-orphans
docker-compose -f ${DOCKER_COMPOSE_FILE} build --no-cache
docker-compose -f ${DOCKER_COMPOSE_FILE} up -d

# Ожидать готовности сервисов
print_status "Ожидание готовности сервисов..."
sleep 30

# Проверить состояние сервисов
print_status "Проверка состояния сервисов..."

# Проверить состояние API
if curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1; then
    print_status "API сервис работает"
else
    print_error "API сервис не отвечает"
    exit 1
fi

# Проверить состояние Frontend
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    print_status "Frontend сервис работает"
else
    print_warning "Frontend сервис не отвечает"
fi

# Проверить состояние Prometheus
if curl -f http://localhost:9090 > /dev/null 2>&1; then
    print_status "Prometheus сервис работает"
else
    print_warning "Prometheus сервис не отвечает"
fi

# Проверить состояние Grafana
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_status "Grafana сервис работает"
else
    print_warning "Grafana сервис не отвечает"
fi

# Показать URL сервисов
echo ""
echo -e "${GREEN}Развертывание успешно завершено!${NC}"
echo ""
echo -e "${YELLOW}URL сервисов:${NC}"
echo -e "  API Документация: http://localhost:8000/docs"
echo -e "  Frontend Приложение: http://localhost:8501"
echo -e "  Prometheus Метрики: http://localhost:9090"
echo -e "  Grafana Дашборды: http://localhost:3000"
echo -e "  База данных: localhost:5432"
echo ""
echo -e "${YELLOW}Учетные данные по умолчанию:${NC}"
echo -e "  Grafana: admin/admin"
echo -e "  База данных: postgres/password"
echo ""

# Показать запущенные контейнеры
print_status "Запущенные контейнеры:"
docker-compose -f ${DOCKER_COMPOSE_FILE} ps

echo ""
print_status "Развертывание завершено!"

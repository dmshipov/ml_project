# Многоэтапная сборка для production-ready приложения кредитного скоринга
FROM python:3.13-slim as base

# Установить переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Установить системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Создать пользователя приложения
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Установить рабочую директорию
WORKDIR /app

# Установить Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать код приложения
COPY . .

# Создать необходимые директории
RUN mkdir -p /app/logs /app/models/trained /app/data/processed

# Изменить владельца директории приложения
RUN chown -R appuser:appuser /app

# Переключиться на пользователя без root прав
USER appuser

# Открыть порт
EXPOSE 8000

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Запустить приложение
CMD ["uvicorn", "api.simple_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Система Кредитного Скоринга

![Cистема кредитного скоринга для предсказания одобрения кредита и определения суммы займа](screen.png)



## Архитектура проекта

```
credit_scoring_project/
├── .github/               # GitHub Actions CI/CD
│   └── workflows/
│       └── ci-cd.yml     # Основной CI/CD пайплайн
├── app/                   # Основное приложение
│   ├── core/             # Ядро приложения (конфигурация, безопасность)
│   ├── services/         # Бизнес-логика
│   ├── utils/            # Утилиты
│   └── schemas/          # Pydantic схемы
├── api/                   # FastAPI приложение
│   ├── routes/           # API маршруты
│   └── dependencies/     # Зависимости API
├── data/                  # Данные
│   ├── raw/              # Исходные данные
│   ├── processed/        # Обработанные данные
│   └── external/         # Внешние данные
├── models/                # ML модели
│   ├── trained/          # Обученные модели (.pkl файлы)
│   ├── artifacts/        # Артефакты моделей (графики, отчеты)
│   └── checkpoints/      # Чекпоинты во время обучения
├── config/                # Конфигурация
│   └── settings/         # Настройки приложения
├── tests/                 # Тесты
│   ├── unit/             # Unit тесты
│   ├── integration/      # Интеграционные тесты
│   └── e2e/              # End-to-end тесты
├── frontend/              # Frontend приложение
│   ├── static/           # Статические файлы
│   ├── templates/        # HTML шаблоны
│   └── components/       # React компоненты
├── monitoring/            # Мониторинг
│   ├── metrics/          # Метрики
│   ├── alerts/           # Алерты
│   ├── dashboards/       # Дашборды
│   └── reports/          # Отчеты мониторинга
├── deployment/            # Развертывание
│   ├── docker/           # Docker конфигурации
│   ├── kubernetes/       # Kubernetes манифесты
│   └── scripts/          # Скрипты развертывания
├── docs/                  # Документация
│   ├── api/              # API документация
│   └── user_guide/       # Руководство пользователя
├── scripts/               # ML и DevOps скрипты
│   ├── data_processing/  # Обработка данных
│   │   ├── eda.py        # Исследовательский анализ данных
│   │   └── preprocessing.py # Предобработка данных
│   ├── model_training/   # Обучение моделей
│   │   ├── train_models.py # Обучение базовых моделей
│   │   ├── hyperparameter_tuning.py # Подбор гиперпараметров
│   │   ├── validation.py # Валидация моделей
│   │   └── mlflow_tracking.py # MLflow интеграция
│   ├── monitoring/       # Мониторинг
│   │   ├── model_monitoring.py # Мониторинг моделей
│   │   └── data_quality_monitor.py # Мониторинг данных
│   ├── deployment/       # Развертывание
│   │   ├── docker_build.sh/.bat # Сборка Docker образов
│   │   └── docker_run.sh/.bat # Запуск в Docker
│   ├── run_pipeline.py   # Главный скрипт пайплайна
│   └── setup_pre_commit.py # Настройка pre-commit
├── logs/                  # Логи приложения
├── .pre-commit-config.yaml # Pre-commit hooks
├── Dockerfile.ml         # Docker для ML пайплайна
├── docker-compose.yml    # Docker Compose конфигурация
├── requirements.txt      # Python зависимости
└── eda_script.py         # EDA утилиты
```

## Быстрый старт

### 1. Клонирование и установка зависимостей

```bash
git clone <repository-url>
cd credit_scoring_project
pip install -r requirements.txt
```

### 2. Настройка окружения

```bash
# Копируем пример конфигурации
cp env.example .env
# Отредактируйте .env файл с вашими настройками

# Настраиваем pre-commit hooks
python scripts/setup_pre_commit.py
```

### 3. Подготовка данных

```bash
# Поместите файл данных в папку data/raw/
# Файл должен называться: accepucicreditcard.csv
```

### 4. Запуск ML пайплайна

#### Вариант A: Полный пайплайн (рекомендуется)

```bash
# Запуск всех этапов последовательно
python scripts/run_pipeline.py
```

#### Вариант B: Пошаговый запуск

```bash
# 1. Исследовательский анализ данных (EDA)
python scripts/data_processing/eda.py

# 2. Предобработка данных
python scripts/data_processing/preprocessing.py

# 3. Обучение моделей
python scripts/model_training/train_models.py

# 4. Подбор гиперпараметров
python scripts/model_training/hyperparameter_tuning.py

# 5. Валидация моделей
python scripts/model_training/validation.py
```

#### Вариант C: Docker (для продакшена)

```bash
# Сборка Docker образа
./scripts/deployment/docker_build.sh  # Linux/Mac
# или
scripts\deployment\docker_build.bat   # Windows

# Запуск пайплайна в Docker
./scripts/deployment/docker_run.sh    # Linux/Mac
# или
scripts\deployment\docker_run.bat     # Windows
```

### 5. Запуск приложения

#### Вариант A: Через Makefile (рекомендуется)

```bash
# Запуск API
make start-api

# Запуск frontend
make start-frontend

# Запуск всех сервисов
make start-all
```

#### Вариант B: Прямой запуск

```bash
# Запуск API (упрощенная версия)
uvicorn api.simple_main:app --reload --host 0.0.0.0 --port 8000

# Запуск frontend
streamlit run frontend/app.py --server.port 8501
```

#### Вариант C: Docker Compose

```bash
# Запуск всех сервисов через Docker
docker-compose up -d
```

### 6. Мониторинг

```bash
# Мониторинг качества модели
python scripts/monitoring/model_monitoring.py

# Мониторинг качества данных
python scripts/monitoring/data_quality_monitor.py
```

## Функциональность

### Основные возможности:
- **Предсказание одобрения кредита** - ML модель для оценки кредитоспособности
- **Определение суммы займа** - Рекомендация оптимальной суммы кредита
- **Мониторинг моделей** - Отслеживание производительности в реальном времени
- **A/B тестирование** - Сравнение различных версий моделей
- **Автоматическое переобучение** - Обновление моделей на новых данных

### API Endpoints:
- `GET /` - Главная страница API
- `GET /docs` - Интерактивная документация Swagger
- `GET /api/v1/health` - Проверка здоровья системы
- `POST /api/v1/predict` - Предсказание кредитного скоринга
- `GET /api/v1/model/info` - Информация о модели
- `GET /api/v1/predictions/stats` - Статистика предсказаний

### Доступ к приложению:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:8501

## CI/CD и Автоматизация

### GitHub Actions

Проект включает полный CI/CD пайплайн с автоматическими проверками:

- **Code Quality** - Black, isort, flake8, mypy, bandit
- **Testing** - pytest с покрытием кода
- **Data Validation** - проверка качества данных
- **Model Training** - автоматическое обучение моделей
- **Deployment** - развертывание в staging/production
- **Monitoring** - отслеживание качества модели

### Pre-commit Hooks

```bash
# Установка hooks
python scripts/setup_pre_commit.py

# Ручной запуск
pre-commit run --all-files
```

### Docker

```bash
# Сборка образа
./scripts/deployment/docker_build.sh

# Запуск пайплайна
./scripts/deployment/docker_run.sh --data-dir ./data --models-dir ./models
```

## Тестирование

### Unit тесты

```bash
# Запуск всех unit тестов
pytest tests/unit/ -v

# Запуск с покрытием
pytest --cov=scripts --cov-report=html

# Запуск конкретного модуля
pytest tests/unit/test_eda.py -v
pytest tests/unit/test_preprocessing.py -v
pytest tests/unit/test_models.py -v
```

### Интеграционные тесты

```bash
# Запуск интеграционных тестов
pytest tests/integration/ -v
```

### E2E тесты

```bash
# Запуск end-to-end тестов
pytest tests/e2e/ -v
```

### Тестирование пайплайна

```bash
# Тестирование полного пайплайна
python scripts/run_pipeline.py --steps eda, preprocessing

# Или используйте Makefile для удобства
make test-all          # Все тесты
make test-unit         # Unit тесты
make test-integration  # Интеграционные тесты
make test-e2e          # E2E тесты
make test-fast         # Быстрые тесты
make ci                # CI пайплайн
```

## Мониторинг

### MLflow (Эксперименты и модели)
- **MLflow UI** - http://localhost:5000
- **Отслеживание экспериментов** - автоматическое логирование всех ML экспериментов
- **Model Registry** - версионирование и управление моделями
- **Метрики и параметры** - детальное отслеживание качества моделей

### Системный мониторинг
- **Prometheus** - Сбор метрик (http://localhost:9090)
- **Grafana** - Дашборды (http://localhost:3000)
- **Sentry** - Отслеживание ошибок

### Мониторинг качества
```bash
# Мониторинг качества модели
python scripts/monitoring/model_monitoring.py

# Мониторинг качества данных
python scripts/monitoring/data_quality_monitor.py
```

### Отчеты мониторинга
- **Отчеты сохраняются** в `monitoring/reports/`
- **Алерты** при деградации производительности
- **Детекция дрифта** данных и моделей

## Разработка

### Makefile команды

Проект включает удобный Makefile для всех основных операций:

```bash
# Установка и настройка
make install          # Установить зависимости
make setup            # Полная настройка проекта
make pre-commit-install # Установить pre-commit hooks

# Тестирование
make test-all         # Все тесты
make test-unit        # Unit тесты
make test-integration # Интеграционные тесты
make test-e2e         # E2E тесты
make test-fast        # Быстрые тесты
make ci               # CI пайплайн

# Качество кода
make format           # Форматировать код
make lint             # Проверить линтерами
make quality-check    # Полная проверка качества

# ML пайплайн
make pipeline         # Полный пайплайн
make pipeline-eda     # Только EDA
make pipeline-training # Только обучение
make monitoring-model  # Мониторинг модели

# Docker
make docker-build     # Собрать Docker образ
make docker-run       # Запустить в Docker

# Очистка
make clean            # Очистить временные файлы
make clean-all        # Полная очистка

# Справка
make help             # Показать все команды
```

### Code Style
```bash
# Форматирование кода
black .
isort .

# Проверка стиля
flake8 .
mypy .

# Или используйте Makefile
make format
make lint
```

### Pre-commit hooks
```bash
# Установка hooks
python scripts/setup_pre_commit.py

# Ручной запуск
pre-commit run --all-files

# Или используйте Makefile
make pre-commit-install
make pre-commit-run
```

## Документация

- [API Documentation](http://localhost:8000/docs) - Swagger UI
- [User Guide](docs/user_guide/) - Руководство пользователя
- [Architecture](docs/architecture.md) - Архитектура системы

## Развертывание

### Production
```bash
# Сборка и запуск
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
```

## Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Создайте Pull Request

## Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE) для деталей.

## Поддержка

Для вопросов и поддержки создайте issue в репозитории или свяжитесь с командой разработки.

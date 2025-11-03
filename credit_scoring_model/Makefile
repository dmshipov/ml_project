# Makefile для проекта кредитного скоринга

.PHONY: help install test test-unit test-integration test-e2e test-all lint format clean setup pre-commit pipeline docker-build docker-run start-api start-frontend start-all

# Переменные
PYTHON = python
PIP = pip
PYTEST = pytest
BLACK = black
ISORT = isort
FLAKE8 = flake8
MYPY = mypy
BANDIT = bandit
PRE_COMMIT = pre-commit

# Цвета для вывода
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Показать справку
	@echo "$(GREEN)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить зависимости
	@echo "$(GREEN)Установка зависимостей...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Зависимости установлены!$(NC)"

setup: install pre-commit-install ## Полная настройка проекта
	@echo "$(GREEN)Проект настроен!$(NC)"

pre-commit-install: ## Установить pre-commit hooks
	@echo "$(GREEN)Установка pre-commit hooks...$(NC)"
	$(PYTHON) scripts/setup_pre_commit.py --install-only
	@echo "$(GREEN)Pre-commit hooks установлены!$(NC)"

pre-commit-update: ## Обновить pre-commit hooks
	@echo "$(GREEN)Обновление pre-commit hooks...$(NC)"
	$(PYTHON) scripts/setup_pre_commit.py --update
	@echo "$(GREEN)Pre-commit hooks обновлены!$(NC)"

test-unit: ## Запустить unit тесты
	@echo "$(GREEN)Запуск unit тестов...$(NC)"
	$(PYTEST) tests/unit/ -v -m "unit" --cov=scripts --cov-report=html --cov-report=term-missing

test-integration: ## Запустить интеграционные тесты
	@echo "$(GREEN)Запуск интеграционных тестов...$(NC)"
	$(PYTEST) tests/integration/ -v -m "integration"

test-e2e: ## Запустить E2E тесты
	@echo "$(GREEN)Запуск E2E тестов...$(NC)"
	$(PYTEST) tests/e2e/ -v -m "e2e"

test-fast: ## Запустить быстрые тесты
	@echo "$(GREEN)Запуск быстрых тестов...$(NC)"
	$(PYTEST) tests/ -v -m "fast" --maxfail=3

test-slow: ## Запустить медленные тесты
	@echo "$(GREEN)Запуск медленных тестов...$(NC)"
	$(PYTEST) tests/ -v -m "slow"

test-all: ## Запустить все тесты
	@echo "$(GREEN)Запуск всех тестов...$(NC)"
	$(PYTEST) tests/ -v --cov=scripts --cov-report=html --cov-report=term-missing --cov-report=xml

test-ml: ## Запустить тесты машинного обучения
	@echo "$(GREEN)Запуск тестов ML...$(NC)"
	$(PYTEST) tests/ -v -m "ml"

test-data: ## Запустить тесты обработки данных
	@echo "$(GREEN)Запуск тестов данных...$(NC)"
	$(PYTEST) tests/ -v -m "data"

test-monitoring: ## Запустить тесты мониторинга
	@echo "$(GREEN)Запуск тестов мониторинга...$(NC)"
	$(PYTEST) tests/ -v -m "monitoring"

lint: ## Проверить код линтерами
	@echo "$(GREEN)Проверка кода линтерами...$(NC)"
	$(FLAKE8) scripts/ tests/
	$(MYPY) scripts/ --ignore-missing-imports
	$(BANDIT) -r scripts/ -f json -o bandit-report.json

format: ## Форматировать код
	@echo "$(GREEN)Форматирование кода...$(NC)"
	$(BLACK) scripts/ tests/
	$(ISORT) scripts/ tests/

format-check: ## Проверить форматирование кода
	@echo "$(GREEN)Проверка форматирования кода...$(NC)"
	$(BLACK) --check scripts/ tests/
	$(ISORT) --check-only scripts/ tests/

pre-commit-run: ## Запустить pre-commit на всех файлах
	@echo "$(GREEN)Запуск pre-commit на всех файлах...$(NC)"
	$(PRE_COMMIT) run --all-files

pre-commit-update: ## Обновить pre-commit hooks
	@echo "$(GREEN)Обновление pre-commit hooks...$(NC)"
	$(PRE_COMMIT) autoupdate

pipeline: ## Запустить полный ML пайплайн
	@echo "$(GREEN)Запуск полного ML пайплайна...$(NC)"
	$(PYTHON) scripts/run_pipeline.py

pipeline-eda: ## Запустить только EDA
	@echo "$(GREEN)Запуск EDA...$(NC)"
	$(PYTHON) scripts/data_processing/eda.py

pipeline-preprocessing: ## Запустить только предобработку
	@echo "$(GREEN)Запуск предобработки...$(NC)"
	$(PYTHON) scripts/data_processing/preprocessing.py

pipeline-training: ## Запустить только обучение моделей
	@echo "$(GREEN)Запуск обучения моделей...$(NC)"
	$(PYTHON) scripts/model_training/train_models.py

pipeline-tuning: ## Запустить только подбор гиперпараметров
	@echo "$(GREEN)Запуск подбора гиперпараметров...$(NC)"
	$(PYTHON) scripts/model_training/hyperparameter_tuning.py

pipeline-validation: ## Запустить только валидацию
	@echo "$(GREEN)Запуск валидации...$(NC)"
	$(PYTHON) scripts/model_training/validation.py

monitoring-model: ## Запустить мониторинг модели
	@echo "$(GREEN)Запуск мониторинга модели...$(NC)"
	$(PYTHON) scripts/monitoring/model_monitoring.py

monitoring-data: ## Запустить мониторинг данных
	@echo "$(GREEN)Запуск мониторинга данных...$(NC)"
	$(PYTHON) scripts/monitoring/data_quality_monitor.py

docker-build: ## Собрать Docker образ
	@echo "$(GREEN)Сборка Docker образа...$(NC)"
	./scripts/deployment/docker_build.sh

docker-run: ## Запустить пайплайн в Docker
	@echo "$(GREEN)Запуск пайплайна в Docker...$(NC)"
	./scripts/deployment/docker_run.sh

docker-run-interactive: ## Запустить пайплайн в Docker в интерактивном режиме
	@echo "$(GREEN)Запуск пайплайна в Docker (интерактивно)...$(NC)"
	./scripts/deployment/docker_run.sh --interactive

docker-clean: ## Очистить Docker образы и контейнеры
	@echo "$(GREEN)Очистка Docker...$(NC)"
	docker system prune -f
	docker image prune -f

clean: ## Очистить временные файлы
	@echo "$(GREEN)Очистка временных файлов...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "bandit-report.json" -delete
	find . -type f -name ".coverage" -delete
	rm -rf logs/*.log
	rm -rf monitoring/reports/*.json

clean-data: ## Очистить обработанные данные
	@echo "$(GREEN)Очистка обработанных данных...$(NC)"
	rm -rf data/processed/*
	rm -rf models/trained/*
	rm -rf models/artifacts/*

clean-all: clean clean-data ## Полная очистка проекта
	@echo "$(GREEN)Полная очистка завершена!$(NC)"

ci: lint test-all ## Запустить CI пайплайн
	@echo "$(GREEN)CI пайплайн завершен!$(NC)"

ci-fast: format-check test-fast ## Быстрый CI пайплайн
	@echo "$(GREEN)Быстрый CI пайплайн завершен!$(NC)"

# Специальные команды для разработки
dev-setup: install pre-commit-install ## Настройка для разработки
	@echo "$(GREEN)Настройка для разработки завершена!$(NC)"

dev-test: format test-fast ## Быстрое тестирование для разработки
	@echo "$(GREEN)Быстрое тестирование завершено!$(NC)"

dev-lint: format lint ## Линтинг для разработки
	@echo "$(GREEN)Линтинг завершен!$(NC)"

# Команды для мониторинга
mlflow-ui: ## Запустить MLflow UI
	@echo "$(GREEN)Запуск MLflow UI...$(NC)"
	mlflow ui --host 0.0.0.0 --port 5000

# Команды для документации
docs-serve: ## Запустить документацию
	@echo "$(GREEN)Запуск документации...$(NC)"
	@echo "Документация доступна в папке docs/"

# Команды для развертывания
deploy-staging: ## Развертывание в staging
	@echo "$(GREEN)Развертывание в staging...$(NC)"
	@echo "Staging развертывание не настроено"

deploy-production: ## Развертывание в production
	@echo "$(GREEN)Развертывание в production...$(NC)"
	@echo "Production развертывание не настроено"

# Команды для отладки
debug-pipeline: ## Отладка пайплайна
	@echo "$(GREEN)Отладка пайплайна...$(NC)"
	$(PYTHON) -m pdb scripts/run_pipeline.py

debug-test: ## Отладка тестов
	@echo "$(GREEN)Отладка тестов...$(NC)"
	$(PYTEST) tests/ -v --pdb

# Команды для профилирования
profile-pipeline: ## Профилирование пайплайна
	@echo "$(GREEN)Профилирование пайплайна...$(NC)"
	$(PYTHON) -m cProfile -o pipeline.prof scripts/run_pipeline.py

profile-test: ## Профилирование тестов
	@echo "$(GREEN)Профилирование тестов...$(NC)"
	$(PYTEST) tests/ --profile

# Команды для статистики
stats-lines: ## Статистика строк кода
	@echo "$(GREEN)Статистика строк кода:$(NC)"
	@find scripts/ -name "*.py" -exec wc -l {} + | tail -1
	@find tests/ -name "*.py" -exec wc -l {} + | tail -1

stats-files: ## Статистика файлов
	@echo "$(GREEN)Статистика файлов:$(NC)"
	@find scripts/ -name "*.py" | wc -l
	@find tests/ -name "*.py" | wc -l

# Команды для проверки качества
quality-check: format-check lint test-fast ## Полная проверка качества
	@echo "$(GREEN)Проверка качества завершена!$(NC)"

quality-fix: format lint ## Исправление проблем качества
	@echo "$(GREEN)Исправление проблем качества завершено!$(NC)"

# Команды для обновления зависимостей
update-deps: ## Обновить зависимости
	@echo "$(GREEN)Обновление зависимостей...$(NC)"
	$(PIP) install --upgrade -r requirements.txt

check-deps: ## Проверить устаревшие зависимости
	@echo "$(GREEN)Проверка устаревших зависимостей...$(NC)"
	$(PIP) list --outdated

# Команды для безопасности
security-check: ## Проверка безопасности
	@echo "$(GREEN)Проверка безопасности...$(NC)"
	$(BANDIT) -r scripts/ -f json -o bandit-report.json
	$(PIP) check

# Команды для производительности
benchmark: ## Бенчмарк производительности
	@echo "$(GREEN)Бенчмарк производительности...$(NC)"
	$(PYTEST) tests/ -v -m "slow" --benchmark-only

# Команды для отчета
report-coverage: ## Отчет о покрытии кода
	@echo "$(GREEN)Отчет о покрытии кода...$(NC)"
	$(PYTEST) tests/ --cov=scripts --cov-report=html --cov-report=term-missing

report-quality: ## Отчет о качестве кода
	@echo "$(GREEN)Отчет о качестве кода...$(NC)"
	$(FLAKE8) scripts/ --format=html --htmldir=flake8-report
	$(MYPY) scripts/ --html-report mypy-report --ignore-missing-imports

# Команды для очистки отчетов
clean-reports: ## Очистить отчеты
	@echo "$(GREEN)Очистка отчетов...$(NC)"
	rm -rf htmlcov/
	rm -rf flake8-report/
	rm -rf mypy-report/
	rm -f coverage.xml
	rm -f bandit-report.json
	rm -f pipeline.prof

# Команды для запуска сервисов
start-api: ## Запустить API сервер
	@echo "$(GREEN)Запуск API сервера...$(NC)"
	uvicorn api.simple_main:app --host 0.0.0.0 --port 8000 --reload

start-frontend: ## Запустить frontend
	@echo "$(GREEN)Запуск frontend...$(NC)"
	streamlit run frontend/app.py --server.port 8501

start-all: ## Запустить все сервисы
	@echo "$(GREEN)Запуск всех сервисов...$(NC)"
	@echo "API: http://localhost:8000"
	@echo "Frontend: http://localhost:8501"
	@echo "API Docs: http://localhost:8000/docs"

# Команды для помощи
help-install: ## Помощь по установке
	@echo "$(GREEN)Помощь по установке:$(NC)"
	@echo "1. make install - установить зависимости"
	@echo "2. make setup - полная настройка проекта"
	@echo "3. make pre-commit-install - установить pre-commit hooks"

help-test: ## Помощь по тестированию
	@echo "$(GREEN)Помощь по тестированию:$(NC)"
	@echo "1. make test-unit - unit тесты"
	@echo "2. make test-integration - интеграционные тесты"
	@echo "3. make test-e2e - E2E тесты"
	@echo "4. make test-all - все тесты"

help-pipeline: ## Помощь по пайплайну
	@echo "$(GREEN)Помощь по пайплайну:$(NC)"
	@echo "1. make pipeline - полный пайплайн"
	@echo "2. make pipeline-eda - только EDA"
	@echo "3. make pipeline-preprocessing - только предобработка"
	@echo "4. make pipeline-training - только обучение"

# Команда по умолчанию
.DEFAULT_GOAL := help
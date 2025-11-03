"""Настройки конфигурации приложения."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict  # Импорт обновлен для V2


class Settings(BaseSettings):
    """Настройки приложения."""

    # Конфигурация API
    api_host: str = Field(default="0.0.0.0")  # Убрано env="API_HOST" – теперь автоматически из env
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    debug: bool = Field(default=False)

    # Конфигурация базы данных
    database_url: str = Field()  # Без default, поскольку env обязателен (или добавьте default=None если опционально, но в V1 был env без default)
    database_host: str = Field(default="localhost")
    database_port: int = Field(default=5432)
    database_name: str = Field(default="credit_scoring_db")
    database_user: str = Field(default="user")
    database_password: str = Field(default="password")

    # Конфигурация модели
    model_path: str = Field(default="models/trained/credit_scoring_model.pkl")
    model_version: str = Field(default="1.0.0")
    model_threshold: float = Field(default=0.5)

    # Безопасность
    secret_key: str = Field()  # Обязательное, без default – если env не установлен, возникнет ошибка
    jwt_secret_key: str = Field()  # Аналогично
    access_token_expire_minutes: int = Field(default=30)

    # Мониторинг
    prometheus_port: int = Field(default=9090)
    grafana_port: int = Field(default=3000)
    sentry_dsn: Optional[str] = Field(default=None)

    # Внешние API
    external_credit_api_url: Optional[str] = Field(default=None)
    external_credit_api_key: Optional[str] = Field(default=None)

    # Логирование
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Флаги функций
    enable_model_monitoring: bool = Field(default=True)
    enable_automatic_retraining: bool = Field(default=False)
    enable_a_b_testing: bool = Field(default=False)

    # Конфигурация заменена для V2 совместимости
    model_config = SettingsConfigDict(
        env_file=".env",  # Чтение из .env файла
        case_sensitive=False,  # Игнор регистра в именах env переменных
        populate_by_name=True,  # Поддержка псевдонимов, если нужны
    )


# Глобальный экземпляр настроек – остается без изменений
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Получить настройки приложения."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

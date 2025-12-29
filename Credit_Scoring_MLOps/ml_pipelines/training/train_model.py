import argparse
import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Конфигурация логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_preprocess_data(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Загрузка и начальная подготовка данных из CSV-файла."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл данных отсутствует: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("В файле нет колонки 'target' с целевой меткой")
    
    # Работа с пропусками: замещение средними значениями для численных характеристик
    imputer = SimpleImputer(strategy='mean')
    feature_cols = [col for col in df.columns if col != "target"]
    df[feature_cols] = imputer.fit_transform(df[feature_cols])
    
    x = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=int)
    logger.info(f"Данные успешно загружены: {x.shape[0]} образцов, {x.shape[1]} атрибутов")
    return x, y


def create_model_pipeline(hidden_sizes: Tuple[int, ...], activation: str, max_iters: int) -> Pipeline:
    """Формирование конвейера модели с указанными параметрами."""
    classifier = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        activation=activation,
        random_state=42,
        max_iter=max_iters,
        early_stopping=True,  # Чтобы избежать переобучения
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
    return pipeline


def perform_training_and_evaluation(
    x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, pipeline: Pipeline
) -> str:
    """Тренировка модели и анализ производительности."""
    try:
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_prob = pipeline.predict_proba(x_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
        
        # Базовый отчёт
        report = classification_report(y_test, y_pred, digits=4)
        
        # Дополняющие показатели
        if y_prob is not None:
            auc = roc_auc_score(y_test, y_prob)
            report += f"\nROC-AUC: {auc:.4f}"
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        report += f"\nМатрица ошибок:\n{conf_matrix}"
        
        logger.info("Анализ модели выполнен")
        return report
    except Exception as e:
        logger.error(f"Проблема во время тренировки или оценки: {e}")
        raise


def execute_training(data_path: Path, model_path: Path, report_path: Path, 
                     hidden_sizes: Tuple[int, ...], activation: str, max_iters: int, use_grid_search: bool) -> None:
    """Главная процедура тренировки модели."""
    x, y = load_and_preprocess_data(data_path)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    
    pipeline = create_model_pipeline(hidden_sizes, activation, max_iters)
    
    if use_grid_search:
        # Базовая сетка параметров для примера; возможно расширение
        param_grid = {
            'classifier__hidden_layer_sizes': [(64, 32), (128, 64)],
            'classifier__activation': ['relu', 'tanh'],
        }
        pipeline = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        logger.info("Инициирование GridSearchCV для выбора оптимальных параметров")
    
    report = perform_training_and_evaluation(x_train, x_test, y_train, y_test, pipeline)
    
    # Запись модели
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info(f"Модель записана по адресу {model_path}")
    
    # Фиксация отчета
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.is_dir():
        report_path = report_path / "classification_report_sklearn.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Отчет зафиксирован в {report_path}")
    
    print("Сводка по эффективности модели:")
    print(report)


def setup_arguments() -> argparse.Namespace:
    """Установка параметров командной строки."""
    parser = argparse.ArgumentParser(description="Тренировка и анализ модели кредитного скоринга с настраиваемыми параметрами")
    parser.add_argument(
        "--data", type=Path, required=True, help="Адрес CSV-файла с тренировочным набором данных"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/model_default.pkl"), 
        help="Местоположение для записи готовой модели"
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/classification_report_sklearn.txt"), 
        help="Путь к текстовому файлу с анализом качества"
    )
    parser.add_argument(
        "--hidden_sizes", type=int, nargs='+', default=[64, 32], 
        help="Объемы скрытых уровней (к примеру, 64 32)"
    )
    parser.add_argument(
        "--activation", type=str, default="relu", choices=["relu", "tanh", "logistic"], 
        help="Активационная функция"
    )
    parser.add_argument(
        "--max_iters", type=int, default=120, help="Предельное количество эпох"
    )
    parser.add_argument(
        "--grid_search", action="store_true", help="Активировать GridSearchCV для оптимизации параметров"
    )
    return parser.parse_args()


def main() -> None:
    args = setup_arguments()
    execute_training(
        args.data, args.output, args.report, 
        tuple(args.hidden_sizes), args.activation, args.max_iters, args.grid_search
    )


if __name__ == "__main__":
    main()

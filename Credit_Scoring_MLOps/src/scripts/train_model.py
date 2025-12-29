import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл данных отсутствует: {csv_path}")

    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError("В файле отсутствует столбец 'target' для целевой переменной")

    x = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy()
    return x, y


def build_pipeline() -> Pipeline:
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        random_state=42,
        max_iter=120,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", model),
        ],
    )
    return pipeline


def train_model(data_path: Path, model_path: Path, report_path: Path) -> None:
    x, y = read_dataset(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, digits=4)
    print("Отчет о качестве модели (sklearn):")
    print(report)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Модель записана по пути {model_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Текстовый отчет записан по адресу {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Тренировка базовой нейронной сети для кредитного скоринга",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Расположение CSV-файла с данными для обучения",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/model_default.pkl"),
        help="Место для хранения обученной модели",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/classification_report_sklearn.txt"),
        help="Адрес для текстового отчета о качестве",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(args.data, args.output, args.report)


if __name__ == "__main__":
    main()

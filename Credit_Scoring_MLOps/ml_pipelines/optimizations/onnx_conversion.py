import argparse
from pathlib import Path

import joblib
import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def determine_input_dimensions(csv_path: Path) -> int:
    df = pd.read_csv(csv_path, nrows=10)
    if "target" not in df.columns:
        raise ValueError("В наборе данных нет колонки 'target'")
    return df.drop(columns=["target"]).shape[1]


def perform_onnx_conversion(model_path: Path, onnx_path: Path, input_dim: int) -> None:
    model = joblib.load(model_path)

    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f_out:
        f_out.write(onnx_model.SerializeToString())

    print(f"Преобразованная модель ONNX записана в файл {onnx_path}")

    loaded_model = onnx.load(onnx_path)
    onnx.checker.check_model(loaded_model)
    print("Проверка структуры модели ONNX прошла удачно")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Конвертация тренированной модели sklearn в ONNX-формат для улучшения производительности",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/model_default.pkl"),
        help="Путь к файлу с подготовленной моделью sklearn",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/default_model.onnx"),
        help="Место для записи ONNX-файла",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Файл CSV с данными для вычисления количества входных признаков",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    n_features = determine_input_dimensions(args.data)
    perform_onnx_conversion(args.model, args.onnx, n_features)


if __name__ == "__main__":
    main()

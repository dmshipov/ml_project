import argparse
from pathlib import Path

import joblib
import onnx
import pandas as pd
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def infer_input_dim(csv_path: Path) -> int:
    df = pd.read_csv(csv_path, nrows=10)
    if "target" not in df.columns:
        raise ValueError("Столбец 'target' отсутствует в наборе данных")
    return df.drop(columns=["target"]).shape[1]


def convert_to_onnx(model_path: Path, onnx_path: Path, input_dim: int) -> None:
    model = joblib.load(model_path)

    initial_type = [("input", FloatTensorType([None, input_dim]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f_out:
        f_out.write(onnx_model.SerializeToString())

    print(f"Модель в формате ONNX сохранена по пути {onnx_path}")

    loaded = onnx.load(onnx_path)
    onnx.checker.check_model(loaded)
    print("Валидация структуры модели ONNX завершена успешно")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Преобразование обученной модели в формат ONNX",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/model_default.pkl"),
        help="Расположение файла с обученной моделью sklearn",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/default_model.onnx"),
        help="Место для сохранения файла модели ONNX",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Файл CSV для определения размерности входных данных",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_features = infer_input_dim(args.data)
    convert_to_onnx(args.model, args.onnx, n_features)


if __name__ == "__main__":
    main()

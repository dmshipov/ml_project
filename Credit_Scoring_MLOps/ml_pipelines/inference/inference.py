import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import classification_report


def load_dataset(csv_path: Path, n_rows: int = 2000):
    df = pd.read_csv(csv_path, nrows=n_rows)
    if "target" not in df.columns:
        raise ValueError("В наборе данных отсутствует колонка 'target'")
    x = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy()
    return x, y


def benchmark_sklearn(model_path: Path, x: np.ndarray, y: np.ndarray, runs: int = 20):
    model = joblib.load(model_path)
    start = time.perf_counter()
    for _ in range(runs):
        preds = model.predict(x)
    elapsed = time.perf_counter() - start
    avg_time = elapsed / runs

    report = classification_report(y, preds, digits=4)
    return avg_time, report


def benchmark_onnx(onnx_path: Path, x: np.ndarray, y: np.ndarray, runs: int = 20):
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    for _ in range(runs):
        raw = session.run(None, {input_name: x.astype("float32")})[0]
    elapsed = time.perf_counter() - start
    avg_time = elapsed / runs

    if raw.ndim == 2 and raw.shape[1] >= 2:
        preds_labels = np.argmax(raw, axis=1)
    else:
        preds_labels = (raw > 0.5).astype(int).ravel()

    report = classification_report(y, preds_labels, digits=4)
    return avg_time, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Оценка эффективности и правильности работы моделей sklearn и ONNX "
            "на одном и том же наборе примеров"
        ),
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Файл CSV с тестовыми примерами",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/model_default.pkl"),
        help="Расположение модели sklearn",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=Path("models/default_model.onnx"),
        help="Расположение модели ONNX",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports/benchmarking"),
        help="Папка для хранения результатов тестирования",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, y = load_dataset(args.data)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    sk_time, sk_report = benchmark_sklearn(args.model, x, y)
    onnx_time, onnx_report = benchmark_onnx(args.onnx, x, y)

    print(f"Средняя длительность прогноза sklearn: {sk_time:.6f} сек")
    print(f"Средняя длительность прогноза ONNX:   {onnx_time:.6f} сек")
    if onnx_time > 0:
        print(f"Показатель ускорения sklearn / ONNX: {sk_time / onnx_time:.2f}x")

    (report_dir / "sklearn_bench.txt").write_text(sk_report, encoding="utf-8")
    (report_dir / "onnx_bench.txt").write_text(onnx_report, encoding="utf-8")


if __name__ == "__main__":
    main()

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import classification_report


def load_dataset(csv_path: Path, n_rows: int = 2000):
    """
    Загружает датасет из CSV-файла, предполагая наличие столбца 'target' для меток.
    Возвращает признаки (X) и метки (y) в виде массивов NumPy.
    """
    df = pd.read_csv(csv_path, nrows=n_rows)
    if "target" not in df.columns:
        raise ValueError("В датасете должен быть столбец 'target' для меток.")
    X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
    y = df["target"].to_numpy()
    return X, y


def benchmark_onnx(onnx_path: Path, X: np.ndarray, y: np.ndarray, runs: int = 20):
    """
    Бенчмаркинг ONNX-модели: измеряет среднее время инференса и генерирует отчет классификации.
    Предполагает бинарную классификацию с сигмоидным выходом или мультиклассовую с softmax.
    """
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    for _ in range(runs):
        raw_output = session.run(None, {input_name: X})[0]
    elapsed = time.perf_counter() - start
    avg_time = elapsed / runs

    # Обработка выхода: предполагаем бинарный (сигмоид) или мультиклассовый (argmax)
    if raw_output.ndim == 2 and raw_output.shape[1] > 1:
        preds = np.argmax(raw_output, axis=1)
    else:
        preds = (raw_output.ravel() > 0.5).astype(int)

    report = classification_report(y, preds, digits=4)
    return avg_time, report


def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов командной строки для бенчмаркинга ONNX-моделей.
    """
    parser = argparse.ArgumentParser(
        description="Бенчмаркинг и сравнение двух ONNX-моделей (например, оригинальной и квантованной) на одном датасете.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Путь к CSV-файлу с тестовыми данными, содержащему столбец 'target'.",
    )
    parser.add_argument(
        "--original-onnx",
        type=Path,
        default=Path("models/default_model.onnx"),
        help="Путь к оригинальной ONNX-модели.",
    )
    parser.add_argument(
        "--quantized-onnx",
        type=Path,
        default=Path("models/default_model_int.onnx"),
        help="Путь к квантованной ONNX-модели.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Каталог для сохранения файлов отчетов бенчмаркинга.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Количество запусков инференса для усреднения времени.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Основная функция для запуска сравнительного бенчмаркинга.
    """
    args = parse_args()
    X, y = load_dataset(args.data)

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # Бенчмаркинг оригинальной модели
    orig_time, orig_report = benchmark_onnx(args.original_onnx, X, y, runs=args.runs)
    
    # Бенчмаркинг квантованной модели
    quant_time, quant_report = benchmark_onnx(args.quantized_onnx, X, y, runs=args.runs)

    print(f"Среднее время инференса для оригинальной ONNX: {orig_time:.6f} сек")
    print(f"Среднее время инференса для квантованной ONNX: {quant_time:.6f} сек")
    if quant_time > 0:
        speedup = orig_time / quant_time
        print(f"Ускорение (оригинальная / квантованная): {speedup:.2f}x")

    # Сохранение отчетов
    (report_dir / "benchmark_original_onnx.txt").write_text(orig_report, encoding="utf-8")
    (report_dir / "benchmark_quantized_onnx.txt").write_text(quant_report, encoding="utf-8")
    print(f"Отчеты сохранены в {report_dir}")


if __name__ == "__main__":
    main()

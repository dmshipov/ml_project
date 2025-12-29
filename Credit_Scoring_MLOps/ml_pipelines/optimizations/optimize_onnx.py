import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def validate_onnx_model(model_path: Path) -> None:
    """Проверка правильности ONNX-модели."""
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    print(f"Модель {model_path} прошла валидацию и признана корректной.")


def quantize_onnx_model(input_path: Path, output_path: Path, quant_type: QuantType) -> None:
    """Применение динамического квантования к ONNX-модели."""
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=quant_type,
    )
    print(f"Квантованная модель сохранена по пути {output_path}.")


def configure_arguments() -> argparse.Namespace:
    """Конфигурация параметров командной строки."""
    parser = argparse.ArgumentParser(
        description="Квантование ONNX-модели для повышения эффективности (оптимизация графа пропущена для совместимости).",
    )
    parser.add_argument(
        "--input_model",
        type=Path,
        default=Path("models/model_default.onnx"),
        help="Расположение входной ONNX-модели для квантования.",
    )
    parser.add_argument(
        "--output_model",
        type=Path,
        default=Path("models/default_model_int.onnx"),
        help="Расположение для записи квантованной модели.",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        choices=["QInt8", "QUInt8"],
        default="QInt8",
        help="Вид квантования (QInt8 или QUInt8).",
    )
    return parser.parse_args()


def main() -> None:
    args = configure_arguments()
    
    if not args.input_model.exists():
        raise FileNotFoundError(
            f"ONNX-модель отсутствует: {args.input_model}. "
            "Сначала выполните onnx_conversion.py для преобразования модели.",
        )
    
    # Проверка начальной модели
    validate_onnx_model(args.input_model)
    
    # Квантование (оптимизация графа пропущена для избежания ошибок импорта)
    quant_type = QuantType.QInt8 if args.quant_type == "QInt8" else QuantType.QUInt8
    try:
        quantize_onnx_model(args.input_model, args.output_model, quant_type)
        validate_onnx_model(args.output_model)
    except Exception as e:
        print(f"Проблема при квантовании: {e}")
        raise
    
    print(f"Квантование завершено. Итоговая модель: {args.output_model}")


if __name__ == "__main__":
    main()

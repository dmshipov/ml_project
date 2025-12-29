from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    source = Path("models/model_default.onnx")
    target = Path("models/default_model_int.onnx")

    if not source.exists():
        raise FileNotFoundError(
            f"Файл ONNX-модели отсутствует: {source}. "
            f"Сначала выполните скрипт onnx_conversion.py",
        )

    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        weight_type=QuantType.QInt8,
    )

    print(f"Квантизированная модель записана по адресу {target}")


if __name__ == "__main__":
    main()

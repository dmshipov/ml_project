from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def build_drift_report(
    reference_csv: Path,
    current_csv: Path,
    output_html: Path,
) -> None:
    """Генерация отчета о смещении данных."""
    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(output_html)
    print(f"Отчет о дрейфе данных сохранен по адресу {output_html}")


def main() -> None:
    """Основная функция для выполнения анализа дрейфа."""
    base_dir = Path("monitoring_reports")
    base_dir.mkdir(parents=True, exist_ok=True)

    reference_csv = Path("data/processed/train.csv")
    current_csv = Path("data/processed/current.csv")
    output_html = base_dir / "drift_report.html"

    if not reference_csv.exists() or not current_csv.exists():
        raise FileNotFoundError(
            "Необходимо подготовить файлы data/processed/train.csv и current.csv",
        )

    build_drift_report(reference_csv, current_csv, output_html)


if __name__ == "__main__":
    main()
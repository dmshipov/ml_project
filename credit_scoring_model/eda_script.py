"""
Упрощенный EDAProcessor для базового анализа данных кредитного скоринга.

Этот класс предоставляет методы для:
- Анализа пропусков
- Проверки дубликатов
- Базовой статистики
- Корреляционного анализа
- Визуализации
"""

import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Настройка для корректного вывода на Windows
sys.stdout.reconfigure(encoding='utf-8')

class EDAProcessor:
    def __init__(self, df: pd.DataFrame):
        """
        Инициализация EDAProcessor.

        Args:
            df: DataFrame для анализа
        """
        self.df = df.copy()
        self.output_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Анализирует пропуски в данных.

        Returns:
            DataFrame с информацией о пропусках
        """
        missing_info = pd.DataFrame({
            'missing_count': self.df.isnull().sum(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        missing_info = missing_info[missing_info['missing_count'] > 0].sort_values('missing_percentage', ascending=False)
        return missing_info

    def check_duplicates(self) -> Dict[str, Any]:
        """
        Проверяет дубликаты в данных.

        Returns:
            Dict с информацией о дубликатах
        """
        duplicate_rows = self.df.duplicated().sum()
        duplicate_columns = []

        # Проверяем дублированные столбцы
        for i, col1 in enumerate(self.df.columns):
            for col2 in self.df.columns[i+1:]:
                if self.df[col1].equals(self.df[col2]):
                    duplicate_columns.append((col1, col2))

        return {
            'duplicate_count': duplicate_rows,
            'duplicates': self.df[self.df.duplicated()] if duplicate_rows > 0 else pd.DataFrame(),
            'duplicate_columns': duplicate_columns
        }

    def basic_statistics(self) -> pd.DataFrame:
        """
        Вычисляет базовую статистику для числовых признаков.

        Returns:
            DataFrame со статистикой
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return pd.DataFrame()
        
        stats = self.df[numeric_cols].describe().transpose()
        stats['skewness'] = self.df[numeric_cols].skew()
        stats['kurtosis'] = self.df[numeric_cols].kurtosis()
        return stats

    def correlation_analysis(self) -> pd.DataFrame:
        """
        Вычисляет корреляционную матрицу.

        Returns:
            DataFrame с корреляциями
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return pd.DataFrame()
        
        corr_matrix = self.df[numeric_cols].corr()
        return corr_matrix

    def plot_distributions(self) -> None:
        """
        Строит графики распределений для числовых признаков.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:5]  # Ограничиваем до 5 для скорости
        
        if len(numeric_cols) == 0:
            print("Нет числовых признаков для построения графиков.")
            return
        
        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5*len(numeric_cols), 4))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols):
            sns.histplot(self.df[col].dropna(), ax=axes[i], kde=True)
            axes[i].set_title(f'Распределение {col}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'distributions.png', dpi=100, bbox_inches='tight')
        plt.close()

    def plot_correlation_heatmap(self) -> None:
        """
        Строит тепловую карту корреляций.
        """
        corr_matrix = self.correlation_analysis()
        if corr_matrix.empty:
            print("Недостаточно числовых признаков для корреляционного анализа.")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Корреляционная матрица')
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=100, bbox_inches='tight')
        plt.close()

    def generate_eda_summary(self) -> Dict[str, Any]:
        """
        Генерирует полный сводный отчет EDA.

        Returns:
            Dict с результатами анализа
        """
        print("Генерация сводного отчета EDA...")
        
        summary = {
            'summary': self.analyze_missing_values(),
            'duplicate_info': self.check_duplicates(),
            'basic_stats': self.basic_statistics(),
            'correlation_matrix': self.correlation_analysis()
        }
        
        # Сохраняем текстовые отчеты
        with open(self.output_dir / 'eda_summary.txt', 'w', encoding='utf-8') as f:
            f.write("ОТЧЕТ ПО ИССЛЕДОВАТЕЛЬСКОМУ АНАЛИЗУ ДАННЫХ\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("ИНФОРМАЦИЯ О ПРОПУСКАХ:\n")
            f.write(summary['summary'].to_string())
            f.write("\n\n")
            
            dup_info = summary['duplicate_info']
            f.write(f"КОЛИЧЕСТВО ДУБЛИРОВАННЫХ СТРОК: {dup_info['duplicate_count']}\n")
            if dup_info['duplicate_columns']:
                f.write("ДУБЛИРОВАННЫЕ СТОЛБЦЫ:\n")
                for col1, col2 in dup_info['duplicate_columns']:
                    f.write(f"- {col1} и {col2}\n")
            f.write("\n")
            
            f.write("БАЗОВАЯ СТАТИСТИКА:\n")
            f.write(summary['basic_stats'].to_string())
            f.write("\n\n")
        
        # Строим графики
        self.plot_distributions()
        self.plot_correlation_heatmap()
        
        print(f"Отчет сохранен в: {self.output_dir}")
        
        return {
            'summary': summary['summary'],
            'duplicate_count': summary['duplicate_info']['duplicate_count'],
            'duplicates': summary['duplicate_info']['duplicates'],
            'duplicate_columns': summary['duplicate_info']['duplicate_columns'],
            'basic_stats': summary['basic_stats'],
            'correlation_matrix': summary['correlation_matrix']
        }

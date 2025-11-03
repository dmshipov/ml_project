"""Streamlit frontend application for credit scoring."""

import json
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Конфигурация страницы
st.set_page_config(
    page_title="Система Кредитного Скоринга",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Конфигурация API
API_BASE_URL = "http://localhost:8000/api/v1"

# Пользовательский CSS для красивого дизайна
st.markdown(
    """
<style>
    /* Импорт шрифта */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    /* Основные стили */
    body {
        font-family: 'Roboto', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
    }
    
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.2);
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .approved {
        background: linear-gradient(135deg, #a8e6cf 0%, #dcedc8 100%);
        border-left: 5px solid #28a745;
        color: #155724;
    }
    
    .rejected {
        background: linear-gradient(135deg, #ffcccc 0%, #ff9999 100%);
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    
    /* Стили для вкладок */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        width: 150px;
        background: transparent;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255,255,255,0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #1f77b4 !important;
    }
    
    /* Стили для боковой панели */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f77b4 0%, #4f9cf9 100%);
        color: white !important;
    }
    
    /* Кнопка предсказания */
    .stButton button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        border: none;
        border-radius: 25px;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Ползунки и инпуты */
    .stSlider, .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #1f77b4, #764ba2);
        border-radius: 10px;
    }
    
    /* Подзаголовки */
    h2, h3 {
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Графики Plotly */
    .js-plotly-plot {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Метрики */
    .stMetric {
        background: rgba(255,255,255,0.9);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

def check_api_health():
    """Проверить доступность API."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def make_prediction(request_data):
    """Выполнить запрос предсказания к API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict", json=request_data, timeout=30
        )
        return response.json(), response.status_code
    except Exception as e:
        return {"error": str(e)}, 500

def get_model_info():
    """Получить информацию о модели из API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=10)
        return response.json(), response.status_code
    except:
        return {"error": "API недоступен"}, 500

def get_prediction_stats():
    """Получить статистику предсказаний из API."""
    try:
        response = requests.get(f"{API_BASE_URL}/predictions/stats", timeout=10)
        return response.json(), response.status_code
    except:
        return {"error": "API недоступен"}, 500

def map_employment_length(emp_length_ru):
    """Преобразовать русское значение стажа работы в английское."""
    mapping = {
        "< 1 года": "< 1 year",
        "1 год": "1 year",
        "2 года": "2 years",
        "3 года": "3 years",
        "4 года": "4 years",
        "5 лет": "5 years",
        "6 лет": "6 years",
        "7 лет": "7 years",
        "8 лет": "8 years",
        "9 лет": "9 years",
        "10+ лет": "10+ years",
        "н/д": "n/a",
    }
    return mapping.get(emp_length_ru, emp_length_ru)

def map_home_ownership(home_ownership_ru):
    """Преобразовать русское значение жилищных условий в английское."""
    mapping = {
        "АРЕНДА": "RENT",
        "СОБСТВЕННОСТЬ": "OWN",
        "ИПОТЕКА": "MORTGAGE",
        "ДРУГОЕ": "OTHER",
    }
    return mapping.get(home_ownership_ru, home_ownership_ru)

def map_loan_purpose(purpose_ru):
    """Преобразовать русское значение цели кредита в английское."""
    mapping = {
        "Консолидация долгов": "debt_consolidation",
        "Кредитная карта": "credit_card",
        "Ремонт дома": "home_improvement",
        "Крупная покупка": "major_purchase",
        "Малый бизнес": "small_business",
        "Другое": "other",
    }
    return mapping.get(purpose_ru, purpose_ru)

def map_term(term_str):
    """Преобразовать срок кредита в численное значение."""
    if "36" in term_str:
        return 36
    elif "60" in term_str:
        return 60
    else:
        return 36  # default

def main():
    """Основная функция приложения."""

    # Заголовок
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #2E86AB; /* Синий цвет для профессионального вида */
        display: flex;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Сам заголовок с иконкой
    st.markdown(
        '<div class="main-header"><i class="fas fa-chart-bar"></i> Система Кредитного Скоринга</div>',
        unsafe_allow_html=True,
    )
    # Проверка состояния API
    if not check_api_health():
        st.error("❌ API недоступен. Убедитесь, что backend сервис запущен.")
        st.stop()

    # Боковая панель
    with st.sidebar:
        st.header("📊 Статус Системы")

        # Проверка состояния API
        if check_api_health():
            st.success("🟢 API Онлайн")
        else:
            st.error("🔴 API Офлайн")

        # Информация о модели
        model_info, status = get_model_info()
        if status == 200:
            st.info(f"🧠 Версия модели: {model_info.get('model_version', 'Неизвестно')}")
            st.info(f"📈 Тип модели: {model_info.get('model_type', 'Неизвестно')}")

        # Статистика предсказаний
        stats, status = get_prediction_stats()
        if status == 200:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📉 Всего предсказаний", stats.get("total_predictions", 0))
            with col2:
                st.metric("✅ Процент одобрения", f"{stats.get('approval_rate', 0):.1%}")

    # Основной контент
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Предсказание", "📊 Аналитика", "ℹ️ Информация о модели", "⚙️ Настройки"]
    )

    with tab1:
        st.header("🔮 Предсказание Кредитного Скоринга")

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 👤 Личная Информация")
                annual_inc = st.number_input(
                    "💰 Годовой доход ($)",
                    min_value=0.0,
                    value=50000.0,
                    step=1000.0,
                    help="Введите ваш годовой доход",
                )

                emp_length = st.selectbox(
                    "⏱️ Стаж работы",
                    [
                        "< 1 года",
                        "1 год",
                        "2 года",
                        "3 года",
                        "4 года",
                        "5 лет",
                        "6 лет",
                        "7 лет",
                        "8 лет",
                        "9 лет",
                        "10+ лет",
                        "н/д",
                    ],
                    index=5,
                )

                home_ownership = st.selectbox(
                    "🏠 Жилищные условия",
                    ["АРЕНДА", "СОБСТВЕННОСТЬ", "ИПОТЕКА", "ДРУГОЕ"],
                    index=2,
                )

            with col2:
                st.markdown("### 💼 Информация о Кредите")
                loan_amnt = st.number_input(
                    "💵 Сумма кредита ($)",
                    min_value=1000.0,
                    value=10000.0,
                    step=1000.0,
                    help="Введите запрашиваемую сумму кредита",
                )

                # Изменено: слайдер для срока кредита в диапазоне от 12 месяцев (1 год)
                term = st.slider(
                    "📅 Срок кредита (месяцы)",
                    min_value=12,  # От 1 года
                    max_value=120,  # До 10 лет
                    value=36,  # Значение по умолчанию
                    step=12,  # Шаг 1 год
                    help="Выберите срок кредита от 1 года (12 месяцев)",
                )

                purpose = st.selectbox(
                    "🎯 Цель кредита",
                    [
                        "Консолидация долгов",
                        "Кредитная карта",
                        "Ремонт дома",
                        "Крупная покупка",
                        "Малый бизнес",
                        "Другое",
                    ],
                    index=0,
                )

        st.markdown("### 💳 Кредитная Информация")
        col3, col4 = st.columns(2)

        with col3:
            fico_low = st.slider(
                "📊 Диапазон FICO Score (Нижний)",
                min_value=300,
                max_value=850,
                value=700,
                step=10,
            )

            fico_high = st.slider(
                "📈 Диапазон FICO Score (Верхний)",
                min_value=300,
                max_value=850,
                value=750,
                step=10,
            )

            dti = st.slider(
                "📉 Коэффициент долг/доход (%)",
                min_value=0.0,
                max_value=100.0,
                value=15.5,
                step=0.1,
            )

        with col4:
            revol_util = st.slider(
                "🔄 Использование кредитных лимитов (%)",
                min_value=0.0,
                max_value=100.0,
                value=25.0,
                step=1.0,
            )

            inq_last_6mths = st.number_input(
                "🔍 Запросы за последние 6 месяцев", min_value=0, value=2, step=1
            )

            delinq_2yrs = st.number_input(
                "⚠️ Просрочки за последние 2 года", min_value=0, value=0, step=1
            )

        if st.button(
            "🚀 Предсказать Кредитный Скоринг", type="primary", use_container_width=True
        ):
            request_data = {
                "annual_inc": annual_inc,
                "emp_length": map_employment_length(emp_length),
                "home_ownership": map_home_ownership(home_ownership),
                "loan_amnt": loan_amnt,
                "term": f"{term} months",  # Преобразуем в строку, как ожидает API
                "purpose": map_loan_purpose(purpose),
                "fico_range_low": fico_low,
                "fico_range_high": fico_high,
                "dti": dti,
                "revol_util": revol_util,
                "inq_last_6mths": inq_last_6mths,
                "delinq_2yrs": delinq_2yrs,
                "pub_rec": 0,
            }

            with st.spinner("🔄 Выполняется предсказание..."):
                result, status_code = make_prediction(request_data)

            if status_code == 200 and result.get("success"):
                prediction = result["prediction"]

                if prediction["prediction"] == 0:
                    st.markdown(
                        f'<div class="prediction-result approved">'
                        f"<h2>✅ Кредит Одобрен!</h2>"
                        f'<p><strong>🎯 Уверенность:</strong> {prediction["confidence"].title()}</p>'
                        f'<p><strong>🔥 Оценка риска:</strong> {prediction["risk_score"]:.1f}/100</p>'
                        f'<p><strong>💰 Рекомендуемая сумма:</strong> ${prediction.get("recommended_amount", loan_amnt):,.2f}</p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-result rejected">'
                        f"<h2>❌ Кредит Отклонен</h2>"
                        f'<p><strong>🎯 Уверенность:</strong> {prediction["confidence"].title()}</p>'
                        f'<p><strong>🔥 Оценка риска:</strong> {prediction["risk_score"]:.1f}/100</p>'
                        f"<p><strong>⚠️ Причина:</strong> Обнаружен высокий риск</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Вероятность предсказания", f"{prediction['probability']:.1%}")
                with col2:
                    st.metric("⏱️ Время обработки", f"{result['processing_time_ms']:.0f}мс")
                with col3:
                    st.metric("🧠 Версия модели", prediction["model_version"])

                if prediction.get("features_importance"):
                    st.markdown("### 📈 Важность Признаков")
                    importance_df = pd.DataFrame(
                        list(prediction["features_importance"].items()),
                        columns=["Признак", "Важность"],
                    ).sort_values("Важность", ascending=True)

                    fig = px.bar(
                        importance_df,
                        x="Важность",
                        y="Признак",
                        orientation="h",
                        title="🎨 Оценки Важности Признаков",
                        color="Важность",
                        color_continuous_scale=px.colors.sequential.Viridis,
                    )
                    st.plotly_chart(fig, use_container_width=True)

            else:
                error_msg = result.get('error', 'Неизвестная ошибка')
                st.error(f"❌ Ошибка предсказания: {error_msg}")
                # Дополнительная отладка: показать код статуса и полный результат
                with st.expander("🔍 Детали ошибки (для отладки)"):
                    st.write(f"Код статуса: {status_code}")
                    st.json(result)

    with tab2:
        st.header("📊 Панель Аналитики")

        st.info("🚀 Панель аналитики будет реализована здесь")

        col1, col2 = st.columns(2)

        with col1:
            data = pd.DataFrame(
                {
                    "Месяц": ["Янв", "Фев", "Мар", "Апр", "Май", "Июн"],
                    "Процент одобрения": [0.75, 0.78, 0.72, 0.80, 0.76, 0.79],
                }
            )

            fig = px.line(
                data,
                x="Месяц",
                y="Процент одобрения",
                title="📈 Ежемесячный процент одобрения",
                markers=True,
                line_shape="spline",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            risk_data = pd.DataFrame(
                {
                    "Уровень риска": ["Низкий", "Средний", "Высокий"],
                    "Количество": [45, 30, 25],
                }
            )

            fig = px.pie(
                risk_data,
                values="Количество",
                names="Уровень риска",
                title="🍰 Распределение рисков",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("ℹ️ Информация о Модели")

        model_info, status = get_model_info()
        if status == 200:
            col1, col2 = st.columns(2)

            with col1:
                st.metric("🧠 Версия модели", model_info.get("model_version", "Неизвестно"))
                st.metric("📊 Тип модели", model_info.get("model_type", "Неизвестно"))
                st.metric("🔶 Порог", model_info.get("threshold", "Неизвестно"))

            with col2:
                st.metric("🔢 Количество признаков", model_info.get("features_count", "Неизвестно"))
                st.metric("📅 Последнее обновление", model_info.get("last_updated", "Неизвестно"))
        else:
            st.error("Не удалось загрузить информацию о модели")

    with tab4:
        st.header("⚙️ Настройки")

        st.markdown("### 🌐 Конфигурация API")
        api_url = st.text_input("Базовый URL API", value=API_BASE_URL)

        st.markdown("### 📏 Конфигурация Модели")
        threshold = st.slider("Порог предсказания", 0.0, 1.0, 0.5, 0.01)

        if st.button("💾 Сохранить настройки", key="save_settings"):
            st.success("✅ Настройки сохранены!")

if __name__ == "__main__":
    main()

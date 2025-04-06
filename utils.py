import pandas as pd
from dataclasses import dataclass
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional
import streamlit as st
import logging
import time
import aiofiles
import aiohttp
import os
from io import StringIO


logger = logging.getLogger('logger')

WINDOW = 30
THRESHOLD = 2
CACHE_TTL = 300
PLOT_FIGSIZE = (40, 30)
PLOT_DPI = 200

def plot_temperature_time_series(
    data: pd.DataFrame,
    city: str,
    ax: plt.Axes = None
    ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    else:
        fig = ax.figure

    ax.plot(
            data['timestamp'],
            data['temperature'],
            color='blue',
            alpha=0.5,
            label='Температура'
    )

    ax.plot(
            data['timestamp'],
            data['rolling_mean'],
            color='black',
            label='Скользящее среднее'
    )

    anomalies = data[data['is_anomaly']]
    ax.scatter(
            anomalies['timestamp'],
            anomalies['temperature'],
            color='red',
            label='Аномалии'
    )

    ax.set_title(f'Временной ряд температур для {city}')
    ax.set_xlabel('Дата')
    ax.set_ylabel('Температура')
    ax.legend()

    return fig


def plot_seasonal_boxplot(
        data: pd.DataFrame,
        city: str,
        ax: plt.Axes = None
    ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    else:
        fig = ax.figure

    sns.boxplot(
        data=data,
        x='season',
        y='temperature',
        ax=ax
    )

    ax.set_title(f'Распределение температур по сезонам для {city}')
    ax.set_ylabel('Температура')

    return fig


def plot_temperature_distribution(
    data: pd.DataFrame,
    city: str,
    ax: plt.Axes = None) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    else:
        fig = ax.figure

    sns.histplot(
            data=data,
            x='temperature',
            hue='season',
            multiple="stack",
            ax=ax
        )

    ax.set_title(f'Распределение температур для {city}')
    ax.set_xlabel('Температура')
    ax.set_ylabel('Дни')

    return fig


def plot_anomalies_heatmap(
    data: pd.DataFrame,
    city: str,
    ax: plt.Axes = None
    ) -> plt.Figure:

    if ax is None:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    else:
        fig = ax.figure

    data = data.copy()
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month

    anomalies_pivot = data.pivot_table(
            values='is_anomaly',
            index='year',
            columns='month',
            aggfunc='sum'
    ).astype(int)

    sns.heatmap(
            anomalies_pivot,
            ax=ax
    )

    ax.set_title(f'Количество аномалий по месяцам для города {city}')
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Год')

    return fig


def validate_and_prepare_dataframe(
    df: pd.DataFrame
) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """Проверка и подготовка DataFrame"""
    required_columns = {'city', 'timestamp', 'temperature', 'season'}

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        return False, f"Отсутствуют обязательные колонки: {missing}", None

    try:
        prepared_df = df.copy()
        prepared_df['timestamp'] = pd.to_datetime(prepared_df['timestamp'])

        if not pd.api.types.is_numeric_dtype(prepared_df['temperature']):
            return (
                False,
                "Колонка 'temperature' должна содержать числовые значения",
                None
            )

        valid_seasons = {'winter', 'spring', 'summer', 'autumn'}
        if not set(prepared_df['season'].unique()).issubset(valid_seasons):
            return (
                False,
                f"Недопустимые значения в колонке 'season'. "
                f"Допустимые значения: {valid_seasons}",
                None
            )

        return True, "Данные успешно загружены и подготовлены", prepared_df

    except Exception as e:
        return False, f"Ошибка при подготовке данных: {str(e)}", None


async def load_csv_async(file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """Асинхронная загрузка CSV"""
    content = file.getvalue().decode('utf-8')

    df = pd.read_csv(StringIO(content))
    return validate_and_prepare_dataframe(df)


@dataclass
class TemperatureAnalysisClass:
    """здесь сохраняем результаты анализа температуры"""
    city: str
    seasonal_stats: pd.DataFrame
    data: pd.DataFrame
    anomalies_count: int


async def analyze_all_cities_temperature(
    df: pd.DataFrame
) -> Dict[str, TemperatureAnalysisClass]:
    """Анализ температур для всех городов."""
    cities = df['city'].unique()
    analyses = {}
    for city in cities:
        analysis = await analyze_city_temperature(df, city)
        analyses[city] = analysis
    return analyses

 
async def analyze_city_temperature(
    df: pd.DataFrame,
    city: str
) -> TemperatureAnalysisClass:
    """Анализ температур для конкретного города."""
    city_data = df[df['city'] == city].copy()

    # Вычисление скользящего среднего
    city_data['rolling_mean'] = city_data['temperature'].rolling(
        window=WINDOW,
        center=True
    ).mean()

    # Расчет сезонной статистики
    seasonal_stats = city_data.groupby('season').agg({
        'temperature': ['mean', 'std']
    }).round(2)

    # Определение аномалий
    city_data['is_anomaly'] = False
    for season in city_data['season'].unique():
        season_data = seasonal_stats.loc[season]
        mean = season_data[('temperature', 'mean')]
        std = season_data[('temperature', 'std')]

        season_mask = city_data['season'] == season
        city_data.loc[season_mask, 'is_anomaly'] = (
            (city_data.loc[season_mask, 'temperature'] >
             mean + THRESHOLD * std) |
            (city_data.loc[season_mask, 'temperature'] <
             mean - THRESHOLD * std)
        )

    return TemperatureAnalysisClass(
        city=city,
        seasonal_stats=seasonal_stats,
        data=city_data,
        anomalies_count=city_data['is_anomaly'].sum()
    )


@dataclass
class WeatherInfo:
    temperature: float
    is_anomaly: bool
    error: Optional[str] = None


async def get_current_temperature(
    city: str,
    city_analysis: TemperatureAnalysisClass,
    api_key: str
) -> WeatherInfo:
    """Асинхронное получение текущей температуры с кэшированием."""
    cache_key = f"{city}:{api_key}"
    current_time = time.time()

    if cache_key in st.session_state.weather_cache:
        cache_age = current_time - \
            st.session_state.weather_cache_time[cache_key]
        if cache_age < CACHE_TTL:
            return st.session_state.weather_cache[cache_key]
    else:
        logger.info(f"Cache miss для {city}")

    result = await fetch_temperature(city, city_analysis, api_key)

    st.session_state.weather_cache[cache_key] = result
    st.session_state.weather_cache_time[cache_key] = current_time
    logger.info(f"Обновлен кэш для {city}")

    return result


def is_temperature_anomaly(
    temperature: float,
    seasonal_stats: pd.DataFrame,
    current_season: str
) -> bool:
    """Проверка на аномалию температуры."""
    mean = seasonal_stats.loc[current_season, ('temperature', 'mean')]
    std = seasonal_stats.loc[current_season, ('temperature', 'std')]
    return (temperature > mean + 2 * std) or (temperature < mean - 2 * std)

async def fetch_temperature(
    city: str,
    city_analysis: TemperatureAnalysisClass,
    api_key: str
) -> WeatherInfo:
    """Асинхронное получение текущей температуры."""
    logger.info(f"Запрос текущей температуры для города {city}")
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        async with aiohttp.ClientSession() as session:
            logger.debug(
                f"Отправка запроса к OpenWeatherMap API для {city}"
            )
            async with session.get(
                "http://api.openweathermap.org/data/2.5/weather",
                params=params
            ) as response:
                data = await response.json()

                if response.status == 200:
                    logger.info(f"200: получены данные для {city}")
                    is_anomaly = is_temperature_anomaly(
                        data['main']['temp'],
                        city_analysis.seasonal_stats,
                        city_analysis.data['season'].iloc[-1]
                    )
                    return WeatherInfo(
                        temperature=data['main']['temp'],
                        is_anomaly=is_anomaly
                    )

                error_message = data.get('message', 'Неизвестная ошибка')
                if response.status == 401:
                    logger.error(
                        f"401: Неверный API ключ при запросе для {city}"
                    )
                    error_message = "Invalid API key"
                elif response.status == 404:
                    logger.error(f"404: Город {city} не найден")
                    error_message = f"Город {city} не найден"
                else:
                    logger.error(f"Ошибка API: {error_message}")

                return WeatherInfo(
                    temperature=0,
                    is_anomaly=False,
                    error=error_message
                )

    except aiohttp.ClientError as e:
        logger.error(f"Ошибка сети при запросе для {city}: {str(e)}")
        return WeatherInfo(
            temperature=0,
            is_anomaly=False,
            error=f"Ошибка сети: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при запросе для {city}")
        return WeatherInfo(
            temperature=0,
            is_anomaly=False,
            error=f"Неожиданная ошибка: {str(e)}"
        )
#!/bin/bash
# Скрипт для запуска style transfer проекта

echo "🔧 Установка зависимостей через Poetry..."
poetry install

echo "🚀 Запуск Streamlit-приложения с правильным PYTHONPATH..."
PYTHONPATH=src poetry run streamlit run src/style_transfer/app.py

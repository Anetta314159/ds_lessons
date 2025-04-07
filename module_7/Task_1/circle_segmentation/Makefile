# Makefile for circle_segmentation project

# Путь до модели и логов
MODEL_PATH=src/model_weights.pth
LOG_PATH=src/train_log.json

.PHONY: help dev train api streamlit mlflow clean

help:
	@echo "Доступные команды:"
	@echo "  make dev       - Запустить все сервисы (API, Streamlit, MLflow)"
	@echo "  make train     - Запустить обучение модели внутри контейнера"
	@echo "  make api       - Перезапустить только FastAPI"
	@echo "  make streamlit - Перезапустить только Streamlit"
	@echo "  make mlflow    - Перезапустить только MLflow"
	@echo "  make clean     - Удалить собранные образы и контейнеры"

dev:
	docker compose up --build

train:
	docker compose run --rm api poetry run python src/train.py

api:
	docker compose restart api

streamlit:
	docker compose restart streamlit

mlflow:
	docker compose restart mlflow

clean:
	docker compose down --volumes --remove-orphans
	docker system prune -f
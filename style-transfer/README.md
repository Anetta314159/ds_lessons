# Neural Style Transfer (Gatys-style + Fast Style Transfer)

Этот проект реализует перенос художественного стиля на изображение с использованием двух подходов:

- **Gatys-style** — медленный, но точный (оптимизация на основе VGG-фичей).
- **Fast Style Transfer** — быстрый, обучаемый подход на основе сверточной сети.

---

## Возможности

- Перенос стиля двумя способами: оптимизацией или через обученную сеть.
- Интерактивный веб-интерфейс на **Streamlit**.
- API на **FastAPI** (два эндпоинта: `stylize`, `train-gatys`).
- Логирование метрик и артефактов с помощью **MLflow**.
- Поддержка запуска в **Docker**.

---
## Структура проекта
```text
style-transfer/
├── src/
│   └── style_transfer/
│       ├── app.py               # Streamlit-интерфейс
│       ├── api/                 # FastAPI backend
│       ├── fast_model.py        # TransformNet
│       ├── model.py             # Gatys-style logic
│       ├── perceptual_loss.py   # Перцептивный лосс
│       ├── utils.py             # Загрузка/сохранение изображений
│       └── ...
├── data/                        # Изображения и датасеты
├── fast_models/                 # Обученные модели
├── checkpoints-*                # Чекпойнты и превью
├── Dockerfile
├── docker-compose.yml
└── README.md
```
## Docker
Проект поддерживает запуск в контейнере Docker — это позволяет быстро поднять FastAPI-сервер 
без необходимости локальной настройки окружения.
docker-compose up --build

### Клонирование проекта

```bash
git clone https://github.com/anetta314159/style-transfer.git
cd style-transfer


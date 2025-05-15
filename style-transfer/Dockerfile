FROM python:3.10-slim

# Установим Poetry
RUN pip install poetry

# Установка системных библиотек (если нужно)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Создаём директорию приложения
WORKDIR /app

# Poetry config
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root

# Копируем код
COPY . .

# Устанавливаем переменную окружения для поиска модулей
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Открываем порт
EXPOSE 8000

# Команда запуска FastAPI
CMD ["poetry", "run", "uvicorn", "style_transfer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
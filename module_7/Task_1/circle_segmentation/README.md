# Circle Segmentation

This project demonstrates synthetic circle segmentation using a UNet model. It provides a FastAPI backend for predictions, a Streamlit frontend for visualization, and MLflow for experiment tracking. Everything is containerized with Docker Compose.

---

## Quick Start

### Prerequisites

Make sure you have Docker installed:

```bash
docker --version
```

---

### Run the project

```bash
docker compose up --build
```

After startup:

- **Streamlit**: http://localhost:8502 
-  **MLflow UI**: http://localhost:5001  
-  **FastAPI** (Docs): http://localhost:8000/docs  

---

### Train the model

```bash
make train
```

---

## Project Structure

```bash
.
├── src/
│   ├── api/           # FastAPI application
│   ├── app.py         # Streamlit interface
│   ├── model/         # UNet model definition
│   ├── utils/         # Metrics, data generators, etc.
│   ├── train.py       # Training script
├── Dockerfile         # Base Dockerfile
├── docker-compose.yml
├── pyproject.toml     # Poetry dependencies
├── README.md          # ← This file
└── Makefile           # make commands (train, test, etc.)
```

---

## Features

- Visualizes input image, ground truth and prediction side-by-side
- Logs metrics and parameters with MLflow
- Microservice architecture (FastAPI + Streamlit)
- Docker & Poetry support

---

## Dependencies

- Python 3.9+
- torch
- torchvision
- fastapi
- streamlit
- uvicorn
- mlflow
- opencv-python
- poetry

---

##  Contact

Author: [anetta314159](https://github.com/anetta314159)
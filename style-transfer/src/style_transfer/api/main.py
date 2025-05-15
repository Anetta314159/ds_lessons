from fastapi import FastAPI
from style_transfer.api.routes import router

app = FastAPI(title="Style Transfer API")
app.include_router(router)
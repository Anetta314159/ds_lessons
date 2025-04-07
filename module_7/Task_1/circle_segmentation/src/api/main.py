from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from models.unet import UNet
from utils.metrics import compute_iou
import torch
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load("src/model_weights.pth", map_location=device))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("L").resize((128, 128))
    img = np.array(image).astype(np.float32) / 255.0
    x = torch.tensor(img[None, None, ...], dtype=torch.float32).to(device)

    with torch.no_grad():
        pred = model(x)
        mask = (pred[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

    return JSONResponse({"prediction": mask.tolist()})
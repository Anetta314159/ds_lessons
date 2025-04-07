
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.unet import UNet
from data.dataset import CircleSegmentationDataset
import numpy as np
import json
import os
import mlflow
import mlflow.pytorch

# IoU метрика
def compute_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = ((pred + target) > 0).float().sum(dim=(1,2,3))
    iou = (intersection / union.clamp(min=1e-6)).mean().item()
    return iou

# Params
epochs = 17
batch_size = 5
lr = 1e-3
image_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
dataset = CircleSegmentationDataset(200, image_size)
val_size = test_size = len(dataset) // 5
train_size = len(dataset) - val_size - test_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# Model
model = UNet().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses, val_losses, test_losses = [], [], []
val_ious, test_ious = [], []

# MLflow
mlflow.set_experiment("circle_segmentation")

with mlflow.start_run():
    mlflow.set_tag("model", f"UNet_lr_{lr}_bs_{batch_size}")
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size
    })

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        total_test_loss = 0
        val_iou_total = 0
        test_iou_total = 0
        val_batches = 0
        test_batches = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = loss_fn(output, y)
                total_val_loss += loss.item()
                val_iou_total += compute_iou(torch.sigmoid(output), y)
                val_batches += 1

            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = loss_fn(output, y)
                total_test_loss += loss.item()
                test_iou_total += compute_iou(torch.sigmoid(output), y)
                test_batches += 1

        train_loss = total_train_loss / len(train_loader)
        val_loss = total_val_loss / val_batches
        test_loss = total_test_loss / test_batches
        val_iou = val_iou_total / val_batches
        test_iou = test_iou_total / test_batches

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        val_ious.append(val_iou)
        test_ious.append(test_iou)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f} (IoU={val_iou:.3f}), test={test_loss:.4f} (IoU={test_iou:.3f})")

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_iou": val_iou,
            "test_iou": test_iou
        }, step=epoch)

    # Save model weights
    model_path = "model_weights.pth"
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)

    # Save loss and IoU
    log_path = "train_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "train": train_losses,
            "val": val_losses,
            "test": test_losses,
            "val_iou": val_ious,
            "test_iou": test_ious
        }, f)
    mlflow.log_artifact(log_path)

    # Saved example from validation
    X_val, Y_val, preds_val = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            pred = model(x)
            X_val.append(x.cpu().numpy())
            Y_val.append(y.cpu().numpy())
            preds_val.append(pred.cpu().numpy())

    X_val = np.concatenate(X_val)
    Y_val = np.concatenate(Y_val)
    preds_val = np.concatenate(preds_val)

    val_np_path = "val_examples.npy"
    np.save(val_np_path, {
        "images": X_val,
        "labels": Y_val,
        "preds": preds_val
    })
    mlflow.log_artifact(val_np_path)

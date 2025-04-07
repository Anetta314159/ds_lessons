
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
import torch

def compute_iou(pred, target):
    pred = (pred > 0.5).float()
    return jaccard_score(target.cpu().numpy().flatten(), pred.cpu().numpy().flatten())

def visualize_predictions(images, true_masks, predicted_masks, num_samples=4):
    fig, axs = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    for i in range(num_samples):
        axs[i, 0].imshow(images[i][0].cpu(), cmap='gray')
        axs[i, 0].set_title("Input Image")
        axs[i, 1].imshow(true_masks[i][0].cpu(), cmap='gray')
        axs[i, 1].set_title("Ground Truth")
        axs[i, 2].imshow(predicted_masks[i][0].cpu(), cmap='gray')
        axs[i, 2].set_title("Prediction")
        for j in range(3):
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()

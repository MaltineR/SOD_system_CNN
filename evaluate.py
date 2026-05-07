import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import UNetSODModel
from data_loader import create_loaders


MODEL_PATH = "saved_models/unet_model.pth"
OUTPUT_DIR = "outputs/unet"


def compute_metrics(pred, mask):
    pred = torch.clamp(pred, 0, 1)
    pred_bin = (pred > 0.5).float()
    mask_bin = (mask > 0.5).float()

    tp = (pred_bin * mask_bin).sum()
    fp = (pred_bin * (1 - mask_bin)).sum()
    fn = ((1 - pred_bin) * mask_bin).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    iou = tp / (tp + fp + fn + 1e-6)
    mae = torch.abs(pred - mask_bin).mean()

    return iou.item(), precision.item(), recall.item(), f1.item(), mae.item()


def visualize(img, mask, pred, save_path):
    img = img.permute(1, 2, 0).cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().detach().numpy()

    img = np.clip(img, 0, 1)
    pred = np.clip(pred, 0, 1)

    pred_binary = (pred > 0.5).astype(np.float32)

    overlay = img.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_binary)

    plt.figure(figsize=(12, 3))

    plt.subplot(1, 4, 1)
    plt.title("Input")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Prediction")
    plt.imshow(pred_binary, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, _, test_loader = create_loaders()

    model = UNetSODModel().to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_mae = 0

    num_batches = 0

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            iou, precision, recall, f1, mae = compute_metrics(preds, masks)

            total_iou += iou
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_mae += mae

            num_batches += 1

    avg_iou = total_iou / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches
    avg_f1 = total_f1 / num_batches
    avg_mae = total_mae / num_batches

    print("\n===== UNET TEST RESULTS =====")
    print(f"IoU       : {avg_iou:.4f}")
    print(f"Precision : {avg_precision:.4f}")
    print(f"Recall    : {avg_recall:.4f}")
    print(f"F1-score  : {avg_f1:.4f}")
    print(f"MAE       : {avg_mae:.4f}")

    print("\nSaving visual examples...")

    saved = 0
    max_samples = 10

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)

            for i in range(imgs.size(0)):
                if saved >= max_samples:
                    break

                visualize(
                    imgs[i],
                    masks[i],
                    preds[i],
                    os.path.join(OUTPUT_DIR, f"sample_{saved}.png")
                )

                saved += 1

            if saved >= max_samples:
                break

    print(f"Saved {saved} visual examples in '{OUTPUT_DIR}' folder.")


if __name__ == "__main__":
    evaluate()
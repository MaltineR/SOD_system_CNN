import os
import torch
import torch.nn as nn

from model import UNetSODModel
from data_loader import create_loaders


MODEL_NAME = "unet"

MODEL_SAVE_PATH = f"saved_models/{MODEL_NAME}_model.pth"
CHECKPOINT_PATH = f"checkpoints/{MODEL_NAME}_checkpoint.pth"

LEARNING_RATE = 1e-3
EPOCHS = 30


def iou(pred, mask):
    pred = (pred > 0.5).float()

    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum() - inter

    return (inter / (union + 1e-6)).item()


def precision_recall_f1(pred, mask):
    pred = (pred > 0.5).float()

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()
    fn = ((1 - pred) * mask).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision.item(), recall.item(), f1.item()


def train():
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _ = create_loaders()

    model = UNetSODModel().to(device)

    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best = 1e9
    patience = 10
    wait = 0
    start_epoch = 0

    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = checkpoint["epoch"] + 1
        best = checkpoint["best_val_loss"]
        wait = checkpoint["wait"]

        print(f"Checkpoint found. Resuming training from epoch {start_epoch + 1}.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    
    for epoch in range(start_epoch, EPOCHS):
        model.train()

        train_loss = 0
        train_iou = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            inter = (pred * y).sum()
            union = pred.sum() + y.sum() - inter
            iou_loss = 1 - inter / (union + 1e-6)

            loss = bce(pred, y) + 0.5 * iou_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += iou(pred, y)

        model.eval()

        val_loss = 0
        val_iou = 0
        val_f1 = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                inter = (pred * y).sum()
                union = pred.sum() + y.sum() - inter
                iou_loss = 1 - inter / (union + 1e-6)

                loss = bce(pred, y) + 0.5 * iou_loss

                val_loss += loss.item()
                val_iou += iou(pred, y)

                _, _, f1 = precision_recall_f1(pred, y)
                val_f1 += f1

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        avg_train_iou = train_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Train IoU: {avg_train_iou:.4f}")
        print(f"Validation IoU: {avg_val_iou:.4f}")
        print(f"Validation F1-score: {avg_val_f1:.4f}")

        
        if val_loss < best:
            best = val_loss

            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print(f"Best model saved: {MODEL_SAVE_PATH}")

            wait = 0
        else:
            wait += 1

        
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best,
            "wait": wait
        }

        torch.save(checkpoint, CHECKPOINT_PATH)

        print(f"Checkpoint saved: {CHECKPOINT_PATH}")

        
        if wait >= patience:
            print("Early stopping activated.")
            break

    print("\nTraining finished.")


if __name__ == "__main__":
    train()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.face_dataset import FaceDataset
from models.backbone_ir50 import IR50
from models.arcface_head import ArcFace
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import re

# ================= CONFIG =================
DATA_ROOT = "data/train"
BATCH_SIZE = 128
EPOCHS = 40  # Target total epochs (will auto-extend if resuming from epoch >= 20)
ADDITIONAL_EPOCHS = 20  # Epochs to add if resumed at or past target
LR = 0.01
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
EMBEDDING_SIZE = 512
NUM_WORKERS = 2 if os.name != 'nt' else 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESUME_CKPT = None
# =========================================

def get_start_epoch(path):
    m = re.search(r"epoch_(\d+)", path)
    return int(m.group(1)) if m else 0

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Target total epochs")
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path to resume from")
    args = parser.parse_args()

    target_epochs = args.epochs

    print(f"[INFO] Using device: {DEVICE}")

    dataset = FaceDataset(DATA_ROOT)
    num_classes = len(dataset.class_to_idx)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True
    )

    backbone = IR50(EMBEDDING_SIZE).to(DEVICE)
    arcface = ArcFace(EMBEDDING_SIZE, num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(
        list(backbone.parameters()) + list(arcface.parameters()),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler("cuda")
    os.makedirs("checkpoints", exist_ok=True)

    start_epoch = 0

    # -------- RESUME (AUTO/SAFE) --------
    resume_path = args.resume or RESUME_CKPT
    if not resume_path:
        ckpt_files = [f for f in os.listdir("checkpoints") if f.endswith(".pth") and "epoch_" in f]
        if ckpt_files:
            ckpt_files.sort(key=get_start_epoch, reverse=True)
            resume_path = os.path.join("checkpoints", ckpt_files[0])
            print(f"[INFO] Auto-detected latest checkpoint: {resume_path}")

    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Loading checkpoint to resume: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=DEVICE)
        
        if isinstance(checkpoint, dict) and "backbone" in checkpoint:
            backbone.load_state_dict(checkpoint["backbone"])
            if "arcface" in checkpoint:
                arcface.load_state_dict(checkpoint["arcface"])
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception:
                    pass
            start_epoch = checkpoint.get("epoch", get_start_epoch(resume_path))
        else:
            backbone.load_state_dict(checkpoint)
            start_epoch = get_start_epoch(resume_path)
            
        print(f"[INFO] Resuming training from Epoch {start_epoch + 1}")

    if start_epoch >= target_epochs:
        target_epochs = start_epoch + ADDITIONAL_EPOCHS
        print(f"[INFO] Current epoch ({start_epoch}) >= target epochs. Automatically extending training to Epoch {target_epochs}!")

    # Ensure each optimizer param group has 'initial_lr' for PyTorch LRScheduler compatibility
    for group in optimizer.param_groups:
        if 'initial_lr' not in group:
            group['initial_lr'] = group.get('lr', LR)

    scheduler = CosineAnnealingLR(optimizer, T_max=target_epochs, last_epoch=start_epoch - 1 if start_epoch > 0 else -1)

    backbone.train()
    arcface.train()

    print(f"[INFO] Training from Epoch {start_epoch + 1} to Epoch {target_epochs}...\n")

    # -------- TRAIN LOOP --------
    for epoch in range(start_epoch, target_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{target_epochs}")

        for images, labels in progress:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                embeddings = backbone(images)
                logits = arcface(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(arcface.parameters()), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                train_acc=f"{100 * correct / total:.2f}%",
                lr=f"{scheduler.get_last_lr()[0]:.6f}"
            )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{target_epochs} | "
            f"Avg Loss: {total_loss / len(loader):.4f} | "
            f"Train Acc: {100 * correct / total:.2f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # -------- SAVE UNIFIED CHECKPOINT --------
        torch.save({
            "epoch": epoch + 1,
            "backbone": backbone.state_dict(),
            "arcface": arcface.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, f"checkpoints/arcface_epoch_{epoch+1}.pth")

    print("🔥 Training complete.")

if __name__ == "__main__":
    main()

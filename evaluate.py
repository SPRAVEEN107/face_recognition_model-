import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.face_dataset import FaceDataset
from models.backbone_ir50 import IR50
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from tqdm import tqdm
import numpy as np
import os
from collections import defaultdict

# ================= CONFIG =================
DATA_ROOT = "data/train"
DEFAULT_CHECKPOINT = "checkpoints/arcface_epoch_15.pth"
BATCH_SIZE = 256
EMBEDDING_SIZE = 512
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

def get_embeddings(loader, model, device):
    """
    Extracts embeddings for the entire dataset.
    Returns:
        embeddings: Tensor of shape (N, 512)
        labels: Tensor of shape (N,)
    """
    all_embeddings = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting Embeddings"):
            images = images.to(device)
            embedding = model(images)
            all_embeddings.append(embedding.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_embeddings), torch.cat(all_labels)

def compute_centroids(embeddings, labels):
    """
    Compute the mean embedding (centroid) for each class.
    """
    print("[INFO] Computing Class Centroids...")
    class_embs = defaultdict(list)
    
    # Organize embeddings by class
    for emb, lab in zip(embeddings, labels):
        class_embs[lab.item()].append(emb.numpy())
    
    centroids = {}
    for lab, embs in class_embs.items():
        embs = np.stack(embs)
        centroid = embs.mean(axis=0)
        # Normalize centroid to lie on the hypersphere
        centroid = centroid / np.linalg.norm(centroid)
        centroids[lab] = centroid
    
    return centroids

import argparse

def evaluate(checkpoint_path, test_dir=None):
    print(f"[INFO] Using device: {DEVICE}")

    # 1. Load Training Dataset (for Centroids)
    if not os.path.exists(DATA_ROOT):
        print(f"[ERROR] Training data root {DATA_ROOT} not found. Need it for centroids.")
        return

    train_dataset = FaceDataset(DATA_ROOT)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 2. Load Model
    backbone = IR50(EMBEDDING_SIZE).to(DEVICE)
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        # Handle unified checkpoint format vs standalone backbone
        if isinstance(checkpoint, dict) and "backbone" in checkpoint:
            backbone.load_state_dict(checkpoint["backbone"])
            print(f"[INFO] Loaded backbone from unified checkpoint (Epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            backbone.load_state_dict(checkpoint)
            print("[INFO] Loaded standalone backbone state dict")
    else:
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return

    backbone.eval()

    # 3. Compute Centroids from Training Data
    print("\n[INFO] Extracting Training Embeddings for Centroids...")
    train_embeddings, train_labels = get_embeddings(train_loader, backbone, DEVICE)
    train_embeddings = F.normalize(train_embeddings, p=2, dim=1)
    
    centroids = compute_centroids(train_embeddings, train_labels)

    centroid_labels = list(centroids.keys())
    centroid_matrix = torch.tensor(
        np.stack([centroids[k] for k in centroid_labels]),
        dtype=torch.float32
    ).to(DEVICE)

    # 4. Prepare Query Set (Train or Test)
    if test_dir:
        print(f"\n[INFO] Loading Test Dataset from {test_dir}...")
        if not os.path.isdir(test_dir):
            print(f"[ERROR] Test directory {test_dir} does not exist.")
            return

        test_dataset = FaceDataset(test_dir)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        print("[INFO] Extracting Test Embeddings...")
        query_embeddings, query_labels = get_embeddings(test_loader, backbone, DEVICE)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    else:
        print("\n[INFO] No test_dir provided. Using Training Embeddings as Query Set (Self-Consistency Check)...")
        query_embeddings = train_embeddings
        query_labels = train_labels

    num_images = query_embeddings.shape[0]
    print(f"\n[INFO] Running Centroid Matching on {num_images} images...")

    predicted_labels = []
    batch_size = 2000

    # 5. Centroid Matching
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            end_idx = min(i + batch_size, num_images)
            query_batch = query_embeddings[i:end_idx].to(DEVICE)

            # Cosine similarity matching
            sim = torch.mm(query_batch, centroid_matrix.t())
            best_idx = sim.argmax(dim=1)

            preds = torch.tensor(
                [centroid_labels[idx] for idx in best_idx.cpu().numpy()]
            )
            predicted_labels.append(preds)

    predicted_labels = torch.cat(predicted_labels)

    # 6. Calculate Metrics
    y_true = query_labels.numpy()
    y_pred = predicted_labels.numpy()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    print("\n" + "="*30)
    print("      EVALUATION RESULTS      ")
    print("="*30)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("="*30)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint file")
    parser.add_argument("--test_dir", type=str, default="data/test", help="Path to test dataset")
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.test_dir)


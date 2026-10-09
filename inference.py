import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models.backbone_ir50 import IR50
import os
import argparse
from datasets.face_dataset import FaceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from collections import defaultdict

# ================= CONFIG =================
DATA_ROOT = "data/train"
CHECKPOINT_PATH = "checkpoints/arcface_epoch_15.pth"
EMBEDDING_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.4  # Recommended range: 0.35 - 0.5
# =========================================

class FacePredictor:
    def __init__(self, threshold=0.5, checkpoint_path=None, compute_centroids=True):
        self.device = DEVICE
        self.threshold = threshold
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  # IR50 expects 112x112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self._load_model(checkpoint_path)
        if compute_centroids:
            self._compute_centroids()

    def _load_model(self, checkpoint_path=None):
        if checkpoint_path:
            path = checkpoint_path
            print(f"[INFO] Loading Model on {self.device} from {path}...")
            self.model = IR50(EMBEDDING_SIZE).to(self.device)
            if os.path.exists(path):
                checkpoint = torch.load(path, map_location=self.device)
                # Handle unified checkpoint format
                if isinstance(checkpoint, dict) and "backbone" in checkpoint:
                    self.model.load_state_dict(checkpoint["backbone"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        else:
             print(f"[INFO] Initializing with random weights (No Checkpoint)")
             self.model = IR50(EMBEDDING_SIZE).to(self.device)
             
        self.model.eval()

    def _compute_centroids(self):
        """
        Re-computes centroids from the training data.
        In a production app, you would save these to a file (e.g., centroids.npy).
        """
        print("[INFO] Computing Centroids from Training Data (One-time setup)...")
        dataset = FaceDataset(DATA_ROOT)
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
        
        class_embs = defaultdict(list)
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Indexing Identities"):
                images = images.to(self.device)
                embeddings = self.model(images)
                
                for emb, lab in zip(embeddings, labels):
                    class_embs[lab.item()].append(emb.cpu().numpy())
        
        self.centroids = []
        self.class_names = []
        
        # We need to map class indices back to names
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        
        sorted_indices = sorted(class_embs.keys())
        for idx in sorted_indices:
            embs = np.stack(class_embs[idx])
            centroid = embs.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            
            self.centroids.append(centroid)
            self.class_names.append(idx_to_class[idx])
            
        self.centroids = torch.tensor(np.stack(self.centroids), dtype=torch.float32).to(self.device)
        print(f"[INFO] Indexed {len(self.class_names)} identities.")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return "Error: File not found", 0.0

        # Preprocess with 5-Point Landmark Face Alignment
        try:
            from utils.face_align import align_face_5pts
            aligned_img = align_face_5pts(image_path, output_type="pil")
            img_tensor = self.transform(aligned_img).unsqueeze(0).to(self.device)
        except Exception as e:
            return f"Error processing image: {e}", 0.0

        # Inference
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)

            # Cosine Similarity
            # [1, 512] x [512, Num_Classes] -> [1, Num_Classes]
            sims = torch.mm(embedding, self.centroids.t())
            
            max_sim, max_idx = sims.max(dim=1)
            score = max_sim.item()
            identity = self.class_names[max_idx.item()]

            if score < self.threshold:
                return "Unknown / Low Confidence", score
            
            return identity, score

def main():
    parser = argparse.ArgumentParser(description="Inference for Face Recognition")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.4, help="Confidence threshold for identification (default: 0.4, recommended: 0.35-0.5)")
    args = parser.parse_args()

    predictor = FacePredictor(threshold=args.threshold, checkpoint_path=args.checkpoint)
    identity, score = predictor.predict(args.image)
    
    print("\n" + "="*30)
    print("      INFERENCE RESULT      ")
    print("="*30)
    print(f"Image    : {args.image}")
    print(f"Identity : {identity}")
    print(f"Score    : {score:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()

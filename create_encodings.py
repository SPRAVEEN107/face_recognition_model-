import os
import cv2
import torch
import numpy as np
from PIL import Image
from pymongo import MongoClient
from inference import FacePredictor
import glob

# Custom .env parser
def load_env(env_path=".env"):
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

# Load environment variables
env = load_env()
MONGO_URI = env.get("MONGO_URI", "mongodb://localhost:27017/AttendanceDB")

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("AttendanceDB")
    collection = db.students
    print(f"[INFO] Connected to MongoDB at {MONGO_URI}")
except Exception as e:
    print(f"[ERROR] Could not connect to MongoDB: {e}")
    exit(1)

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not files:
        return None
    try:
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except:
        files.sort()
    return os.path.join(checkpoint_dir, files[-1])

chkpt_path = get_latest_checkpoint()
if not chkpt_path:
    print("[ERROR] No checkpoints found in 'checkpoints/' directory.")
    exit(1)

print(f"[INFO] Using latest checkpoint for encoding: {chkpt_path}")
# Initialize Predictor (skip training centroids as we use student DB)
predictor = FacePredictor(checkpoint_path=chkpt_path, compute_centroids=False)
model = predictor.model
transform = predictor.transform
device = predictor.device

from utils.face_align import align_face_5pts

def get_embedding(image_path):
    try:
        # Align face using 5-point landmark detection
        face_img = align_face_5pts(image_path, output_type="pil")
        img_tensor = transform(face_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = model(img_tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
        return embedding.cpu().numpy().flatten().tolist()

    except Exception as e:
        print(f"[ERROR] Processing {image_path}: {e}")
        return None

def main():
    images_dir = "images"
    if not os.path.exists(images_dir):
        print(f"[ERROR] Images directory '{images_dir}' not found.")
        return

    # 1. Process subfolders (folder name = identity)
    student_folders = [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]
    
    # 2. Process files in root (filename = identity)
    direct_images = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"[INFO] Found {len(student_folders)} student folders and {len(direct_images)} direct images.")

    # Process Folders
    for student_name in student_folders:
        student_path = os.path.join(images_dir, student_name)
        image_files = glob.glob(os.path.join(student_path, "*.*"))
        
        embeddings = []
        print(f"[INFO] Processing folder: {student_name} ({len(image_files)} images)...")
        
        for img_file in image_files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                emb = get_embedding(img_file)
                if emb:
                    embeddings.append(emb)
        
        if embeddings:
            save_to_db(student_name, np.mean(embeddings, axis=0))

    # Process Direct Images
    for img_name in direct_images:
        filename = os.path.splitext(img_name)[0]
        img_file = os.path.join(images_dir, img_name)
        
        # Try to split by '_' to get RollNo and Name
        if "_" in filename:
            roll_no, student_name = filename.split("_", 1)
            roll_no = roll_no.strip()
            student_name = student_name.strip()
        else:
            roll_no = None
            student_name = filename

        print(f"[INFO] Processing direct image: {img_name} -> RollNo: {roll_no}, Name: {student_name}")
        emb = get_embedding(img_file)
        if emb:
            save_to_db(student_name, emb, roll_no=roll_no)

def save_to_db(student_name, embedding, roll_no=None):
    # Upsert into MongoDB
    student_data = {
        "name": student_name,
        "encoding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
        "updated_at": cv2.getTickCount()
    }
    
    if roll_no:
        student_data["rollNo"] = roll_no

    try:
        # We search by rollNo if available, otherwise by name
        query = {"rollNo": roll_no} if roll_no else {"name": student_name}
        
        collection.update_one(
            query,
            {"$set": student_data},
            upsert=True
        )
        print(f"[SUCCESS] Saved encoding for {student_name} ({roll_no})")
    except Exception as e:
        print(f"[ERROR] Saving to DB for {student_name}: {e}")

if __name__ == "__main__":
    main()

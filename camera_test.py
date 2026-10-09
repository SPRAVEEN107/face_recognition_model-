import os
import cv2
import torch
import numpy as np
from PIL import Image
from pymongo import MongoClient
from inference import FacePredictor
import torch.nn.functional as F

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

# Connect to MongoDB and load encodings
try:
    client = MongoClient(MONGO_URI)
    db = client.get_database("AttendanceDB")
    collection = db.students
    print(f"[INFO] Connected to MongoDB at {MONGO_URI}")
    
    # Load all students from DB
    students_data = list(collection.find({}, {"name": 1, "encoding": 1}))
    if not students_data:
        print("[WARN] No encodings found in MongoDB. Please run create_encodings.py first.")
    else:
        print(f"[INFO] Loaded {len(students_data)} student(s) from database.")
except Exception as e:
    print(f"[ERROR] MongoDB Error: {e}")
    students_data = []

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not files:
        return None
    # Sort by epoch number assuming format 'arcface_epoch_N.pth'
    try:
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except:
        files.sort()
    return os.path.join(checkpoint_dir, files[-1])

# Get checkpoint
LATEST_CHECKPOINT = get_latest_checkpoint()

# Initialize Predictor (skip training centroids as we use student DB)
predictor = FacePredictor(checkpoint_path=LATEST_CHECKPOINT, compute_centroids=False)
if LATEST_CHECKPOINT:
    print(f"[INFO] Initialized with checkpoint: {LATEST_CHECKPOINT}")
else:
    print("[WARN] No checkpoints found. Initializing with random weights.")
model = predictor.model
transform = predictor.transform
device = predictor.device

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def match_face(embedding, students, threshold=0.45):
    """Matches the given embedding against stored student encodings using Cosine Similarity."""
    if not students:
        return "Unknown", 0.0
    
    emb_tensor = torch.tensor(embedding).to(device).unsqueeze(0)
    
    best_match = "Unknown"
    max_sim = 0.0
    
    for student in students:
        stored_emb = torch.tensor(student["encoding"]).to(device).unsqueeze(0)
        # Cosine similarity
        sim = F.cosine_similarity(emb_tensor, stored_emb).item()
        
        if sim > max_sim:
            max_sim = sim
            best_match = student["name"]
            
    if max_sim < threshold:
        return "Unknown", max_sim
        
    return best_match, max_sim

def get_camera():
    for index in range(3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Check for orange-ish dummy feed (High Red, Medium Green, Low Blue)
                mean = np.mean(frame, axis=(0, 1))
                std = np.std(frame)
                
                is_orange = (mean[2] > 200 and mean[1] > 100 and mean[0] < 100)
                
                if std > 5 and not is_orange:
                    print(f"[INFO] Successfully opened camera index {index}")
                    return cap, index
                else:
                    reason = "orange dummy feed" if is_orange else f"low variation (std={std:.2f})"
                    print(f"[WARN] Skipping camera index {index} because it looks like {reason}.")
            cap.release()
    return None, None

def main():
    cap, index = get_camera()
    if cap is None:
        print("[ERROR] Could not open a working webcam (tried 0, 1, 2).")
        return

    print(f"[INFO] Starting Camera Test (Index {index}). Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # detection expects grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        img_h, img_w = frame.shape[:2]
        for (x, y, w, h) in faces:
            # Add 15% padding margin for accurate ArcFace feature extraction
            mx = int(w * 0.15)
            my = int(h * 0.15)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(img_w, x + w + mx)
            y2 = min(img_h, y + h + my)

            face_roi_bgr = frame[y1:y2, x1:x2]
            from utils.face_align import align_face_5pts
            aligned_pil = align_face_5pts(face_roi_bgr, output_type="pil")
            
            img_tensor = transform(aligned_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model(img_tensor)
                embedding = F.normalize(embedding, p=2, dim=1)
                emb_np = embedding.cpu().numpy().flatten()
            
            # Match
            name, score = match_face(emb_np, students_data)
            
            # Draw
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({score:.2f})"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Face Recognition Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

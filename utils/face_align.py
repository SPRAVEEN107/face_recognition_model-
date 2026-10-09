import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import os

# Standard 112x112 ArcFace Reference Canonical Landmarks
REFERENCE_5PTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose tip
    [41.5493, 92.3655],  # left mouth corner
    [70.7299, 92.2041]   # right mouth corner
], dtype=np.float32)

# Lazy loading of MediaPipe FaceMesh for performance & thread safety
_face_mesh = None

def get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    return _face_mesh

# Fallback Haar Cascades
_cascades = None
def get_cascades():
    global _cascades
    if _cascades is None:
        _cascades = [
            (cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'), 1.1, 4),
            (cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'), 1.1, 3),
        ]
    return _cascades

def align_face_5pts(img_input, output_type="pil"):
    """
    Aligns a face image to standard ArcFace 112x112 canonical coordinates using 5 facial landmarks.
    
    Args:
        img_input: PIL.Image or numpy.ndarray (BGR or RGB) or file path string.
        output_type: "pil" (returns PIL.Image in RGB) or "cv2" (returns numpy.ndarray in RGB).
        
    Returns:
        Aligned image (112x112) or resized crop fallback.
    """
    if isinstance(img_input, str):
        if not os.path.exists(img_input):
            raise FileNotFoundError(f"Image file not found: {img_input}")
        pil_img = Image.open(img_input).convert("RGB")
        cv_rgb = np.array(pil_img)
    elif isinstance(img_input, Image.Image):
        pil_img = img_input.convert("RGB")
        cv_rgb = np.array(pil_img)
    elif isinstance(img_input, np.ndarray):
        if len(img_input.shape) == 2:
            cv_rgb = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
        elif img_input.shape[2] == 3:
            # Assume input from cv2 is BGR unless specified
            cv_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_rgb)
    else:
        raise ValueError(f"Unsupported image input type: {type(img_input)}")

    h, w, _ = cv_rgb.shape

    # 1. Try MediaPipe 5-Point Landmark Detection & Affine Warp
    try:
        mesh = get_face_mesh()
        results = mesh.process(cv_rgb)
        
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face = results.multi_face_landmarks[0]
            # Key points: left iris (468), right iris (473), nose tip (1), left mouth (61), right mouth (291)
            pts = np.array([
                [face.landmark[468].x * w, face.landmark[468].y * h],
                [face.landmark[473].x * w, face.landmark[473].y * h],
                [face.landmark[1].x * w, face.landmark[1].y * h],
                [face.landmark[61].x * w, face.landmark[61].y * h],
                [face.landmark[291].x * w, face.landmark[291].y * h],
            ], dtype=np.float32)

            M, _ = cv2.estimateAffinePartial2D(pts, REFERENCE_5PTS)
            if M is not None:
                aligned_rgb = cv2.warpAffine(cv_rgb, M, (112, 112), borderValue=0)
                if output_type == "pil":
                    return Image.fromarray(aligned_rgb)
                return aligned_rgb
    except Exception:
        pass

    # 2. Fallback: Multi-Stage Haar Cascade Face Crop
    gray = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2GRAY)
    for cascade, scale, min_n in get_cascades():
        faces = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=min_n)
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            mx = int(fw * 0.15)
            my = int(fh * 0.15)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(w, x + fw + mx)
            y2 = min(h, y + fh + my)
            
            crop_rgb = cv_rgb[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop_rgb, (112, 112), interpolation=cv2.INTER_AREA)
            if output_type == "pil":
                return Image.fromarray(crop_resized)
            return crop_resized

    # 3. Last Resort Fallback: Direct Center Crop / Resize
    crop_resized = cv2.resize(cv_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    if output_type == "pil":
        return Image.fromarray(crop_resized)
    return crop_resized

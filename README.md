# ArcFace Face Recognition System

This project implements a face recognition system using the ArcFace loss function and a ResNet-50 (IR50) backbone. It provides tools for training, evaluation, and inference.

## 1. Prerequisites (Setup)

Ensure you have Python installed (3.8+ recommended).

**Install Dependencies:**
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

*(If you don't use the requirements file, install manually:)*
```bash
pip install torch torchvision pillow scikit-learn tqdm
```

## 2. Project Structure
- `data/train`: Training images (organized by identity folder).
- `data/test`: Test/Validation images for evaluation.
- `checkpoints/`: Stores trained model weights (e.g., `backbone_epoch_5.pth`).
- `train.py`: Script to train the model.
- `evaluate.py`: Script to verify model accuracy.
- `inference.py`: Script to predict the identity of a single image.

## 3. Usage Guide

### A. Training
To train the model from scratch on the data in `data/train`:
```bash
python train.py
```
*Configuration (epochs, batch size) can be edited at the top of `train.py`.*

### B. Evaluation
To check the accuracy of the model.

**1. Internal Consistency (Training Set)**
Checks how well the model clusters the training faces.
```bash
python evaluate.py
```

**2. Generalization Test (Test Set)**
Checks accuracy on unseen images in `data/test`.
```bash
python evaluate.py --test_dir data/test
```

### C. Inference (Single Image)
To recognize a person in a specific photo:
```bash
python inference.py --image "path/to/your_image.jpg"
```
*Example:*
```bash
python inference.py --image "data/test/n000002/0004_01.jpg"
```

### D. Creating a Test Split
To automatically move 20% of your training data to the test folder:
```bash
python create_test_split.py
```
*(Warning: This moves files. Run only once.)*

## 4. How it Works
1.  **Model**: Uses `IR50` backbone pretrained/trained with `ArcFace` loss.
2.  **Inference Logic**:
    - Calculates the "Center" (Centroid) embedding for every person in the training set.
    - When a new image comes in, extracting its embedding.
    - Compares it to all Centroids using Cosine Similarity.
    - Returns the identity with the highest similarity score.

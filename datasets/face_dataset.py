import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.face_align import align_face_5pts

class FaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.class_to_idx = {}

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self._prepare_dataset()

    def _prepare_dataset(self):
        class_folders = sorted(os.listdir(self.root_dir))

        for idx, class_name in enumerate(class_folders):
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            self.class_to_idx[class_name] = idx

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, idx))

        print(f"[INFO] Total identities: {len(self.class_to_idx)}")
        print(f"[INFO] Total images: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        # Align face to 112x112 canonical ArcFace coordinates
        aligned_img = align_face_5pts(img_path, output_type="pil")
        img_tensor = self.transform(aligned_img)
        return img_tensor, label


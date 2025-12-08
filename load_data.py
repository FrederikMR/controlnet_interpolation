import os
import zipfile
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# -----------------------------
# 1. Helper transform
# -----------------------------
def get_transform(size=256):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# -----------------------------
# 2. Generic image folder dataset
# -----------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder, n_images=None, transform=None):
        self.transform = transform or get_transform()
        # list all images
        self.img_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png'))
        ]
        if n_images is not None and n_images < len(self.img_paths):
            self.img_paths = random.sample(self.img_paths, n_images)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)

# -----------------------------
# 3. ZIP-based loader (for FFHQ or any zipped images)
# -----------------------------
class ZipImageDataset(ImageFolderDataset):
    def __init__(self, zip_path, n_images=None, transform=None):
        extract_dir = os.path.splitext(zip_path)[0]
        if not os.path.exists(extract_dir):
            print(f"Extracting {zip_path} …")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print("Extraction complete.")
        super().__init__(extract_dir, n_images=n_images, transform=transform)

# -----------------------------
# 4. AFHQ loader
# -----------------------------
class AFHQ(ImageFolderDataset):
    def __init__(self, root_dir, n_images=None, transform=None):
        img_paths = []
        for sub in os.listdir(root_dir):
            sub_path = os.path.join(root_dir, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.lower().endswith(('.jpg', '.png')):
                        img_paths.append(os.path.join(sub_path, f))
        if n_images is not None and n_images < len(img_paths):
            img_paths = random.sample(img_paths, n_images)
        self.transform = transform or get_transform()
        self.img_paths = img_paths

# -----------------------------
# 5. Unified loader
# -----------------------------
def load_dataset(name, n_images=None, image_size=256):
    transform = get_transform(image_size)
    
    if name.lower() == "ffhq":
        zip_path = "/work3/fmry/Data/ffhq/00000-20251208T180936Z-3-001.zip"
        return ZipImageDataset(zip_path, n_images=n_images, transform=transform)
    
    elif name.lower() == "afhq":
        root_dir = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        return AFHQ(root_dir, n_images=n_images, transform=transform)
    
    elif name.lower() == "coco":
        zip_path = "/work3/fmry/Data/coco/train2017.zip"
        return ZipImageDataset(zip_path, n_images=n_images, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")

# -----------------------------
# 6. Example usage
# -----------------------------
if __name__ == "__main__":
    # Load 8 images from FFHQ
    ffhq_ds = load_dataset("ffhq", n_images=8, image_size=256)
    print("FFHQ dataset length:", len(ffhq_ds))
    
    # Load 10 images from AFHQ
    afhq_ds = load_dataset("afhq", n_images=10, image_size=256)
    print("AFHQ dataset length:", len(afhq_ds))
    
    # Load 5 images from COCO
    coco_ds = load_dataset("coco", n_images=5, image_size=256)
    print("COCO dataset length:", len(coco_ds))

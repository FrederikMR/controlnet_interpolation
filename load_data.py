import os
import random
import pickle
import zipfile
import numpy as np
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
# 2. Convert CelebA-HQ .pkl to PNGs
# -----------------------------
def convert_celeba_hq_pkl_to_images(pkl_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("Converting CelebA-HQ .pkl to images...")
    
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")  # important for StyleGAN2 pickles
    
    images = data["images"]  # shape [N, H, W, C], float32 [0,1]
    for i, img in enumerate(images):
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.save(os.path.join(out_dir, f"{i:05d}.png"))
    
    print(f"Conversion complete! {len(images)} images saved to {out_dir}")

# -----------------------------
# 3. Generic image folder dataset
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
# 4. COCO loader
# -----------------------------
class COCOImages(ImageFolderDataset):
    def __init__(self, zip_path, n_images=None, transform=None):
        extract_dir = os.path.splitext(zip_path)[0]
        if not os.path.exists(extract_dir):
            print("Extracting COCO zip...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print("Extraction complete.")
        super().__init__(extract_dir, n_images=n_images, transform=transform)

# -----------------------------
# 5. AFHQ loader
# -----------------------------
class AFHQ(ImageFolderDataset):
    def __init__(self, root_dir, n_images=None, transform=None):
        # gather all images in subfolders
        img_paths = []
        for sub in os.listdir(root_dir):
            sub_path = os.path.join(root_dir, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.lower().endswith(('.jpg', '.png')):
                        img_paths.append(os.path.join(sub_path, f))
        # sample N
        if n_images is not None and n_images < len(img_paths):
            img_paths = random.sample(img_paths, n_images)
        self.transform = transform or get_transform()
        self.img_paths = img_paths

# -----------------------------
# 6. Generic loader function
# -----------------------------
def load_dataset(name, n_images=None, image_size=256):
    transform = get_transform(image_size)
    
    if name.lower() == "coco":
        zip_path = "/work3/fmry/Data/coco/train2017.zip"
        return COCOImages(zip_path, n_images=n_images, transform=transform)
    
    elif name.lower() == "celeba-hq":
        img_folder = "/work3/fmry/Data/celeba_hq/images"  # after conversion
        # convert if folder does not exist
        if not os.path.exists(img_folder):
            pkl_path = "/work3/fmry/Data/celeba_hq/karras2018iclr-celebahq-1024x1024.pkl"
            convert_celeba_hq_pkl_to_images(pkl_path, img_folder)
        return ImageFolderDataset(img_folder, n_images=n_images, transform=transform)
    
    elif name.lower() == "afhq":
        root_dir = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        return AFHQ(root_dir, n_images=n_images, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")

# -----------------------------
# 7. Example usage
# -----------------------------
if __name__ == "__main__":
    # Load 8 images from CelebA-HQ
    celeba_ds = load_dataset("celeba-hq", n_images=8, image_size=256)
    print("CelebA-HQ dataset length:", len(celeba_ds))
    
    # Load 10 images from AFHQ
    afhq_ds = load_dataset("afhq", n_images=10, image_size=256)
    print("AFHQ dataset length:", len(afhq_ds))
    
    # Load 5 images from COCO
    coco_ds = load_dataset("coco", n_images=5, image_size=256)
    print("COCO dataset length:", len(coco_ds))

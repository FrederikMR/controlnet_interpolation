#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 18:48:28 2025

@author: fmry
"""

import os
import zipfile
import random
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# -----------------------------
# 1. Helper transforms
# -----------------------------
def get_transform(size=256):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# -----------------------------
# 2. COCO loader
# -----------------------------
class COCOImages(Dataset):
    def __init__(self, zip_path, n_images=None, transform=None, extract_dir=None):
        self.transform = transform or get_transform()
        
        # Determine extraction directory
        if extract_dir is None:
            extract_dir = os.path.splitext(zip_path)[0]
        if not os.path.exists(extract_dir):
            print("Extracting COCO zip...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print("Extraction complete.")
        
        # List all images
        self.img_paths = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png')):
                    self.img_paths.append(os.path.join(root, f))
        
        # Sample N images if requested
        if n_images is not None and n_images < len(self.img_paths):
            self.img_paths = random.sample(self.img_paths, n_images)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)

# -----------------------------
# 3. CelebA-HQ loader
# -----------------------------
class CelebAHQ(Dataset):
    def __init__(self, pkl_path, n_images=None, transform=None):
        self.transform = transform or get_transform()
        print("Loading CelebA-HQ .pkl file...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        images = data['images']  # shape [N, H, W, C], values [0,1]
        print(f"Total images in pkl: {len(images)}")
        
        # Convert to PIL images
        self.imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in images]
        
        if n_images is not None and n_images < len(self.imgs):
            self.imgs = random.sample(self.imgs, n_images)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return self.transform(self.imgs[idx])

# -----------------------------
# 4. AFHQ loader
# -----------------------------
class AFHQ(Dataset):
    def __init__(self, root_dir, n_images=None, transform=None):
        self.transform = transform or get_transform()
        self.img_paths = []
        # Iterate over subfolders (cat, dog, wild)
        for sub in os.listdir(root_dir):
            sub_path = os.path.join(root_dir, sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.lower().endswith(('.jpg', '.png')):
                        self.img_paths.append(os.path.join(sub_path, f))
        
        # Sample N images
        if n_images is not None and n_images < len(self.img_paths):
            self.img_paths = random.sample(self.img_paths, n_images)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)

# -----------------------------
# 5. Generic loader function
# -----------------------------
def load_dataset(name, n_images=None, image_size=256):
    transform = get_transform(image_size)
    if name.lower() == "coco":
        zip_path = "/work3/fmry/Data/coco/train2017.zip"
        return COCOImages(zip_path, n_images=n_images, transform=transform)
    elif name.lower() == "celeba-hq":
        pkl_path = "/work3/fmry/Data/celeba_hq/karras2018iclr-celebahq-1024x1024.pkl"
        return CelebAHQ(pkl_path, n_images=n_images, transform=transform)
    elif name.lower() == "afhq":
        root_dir = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        return AFHQ(root_dir, n_images=n_images, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")

# -----------------------------
# 6. Example usage
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

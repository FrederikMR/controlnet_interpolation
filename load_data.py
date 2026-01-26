import os
import zipfile
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np

# ================================================================
# 0. Global seed
# ================================================================
SEED = 2712

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Dedicated RNG for dataset sampling (deterministic)
rng = random.Random(SEED)

# ================================================================
# 1. Transform helper
# ================================================================
def get_transform(size=256):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

# ================================================================
# 2. Base class: always returns (PIL_image, Tensor)
# ================================================================
class BaseImageDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform or get_transform()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        pil_img = Image.open(self.img_paths[idx]).convert("RGB")
        tensor_img = self.transform(pil_img)
        return pil_img, tensor_img

# ================================================================
# 3. ImageFolder loader
# ================================================================
class ImageFolderDataset(BaseImageDataset):
    def __init__(self, folder, n_images=None, transform=None):
        img_paths = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        )

        if n_images is not None and n_images < len(img_paths):
            img_paths = rng.sample(img_paths, n_images)

        super().__init__(img_paths, transform)

# ================================================================
# 4. ZIP-based loader (FFHQ, COCO, etc.)
# ================================================================
class ZipImageDataset(BaseImageDataset):
    def __init__(self, zip_path, n_images=None, transform=None):
        extract_dir = os.path.splitext(zip_path)[0]

        if not os.path.exists(extract_dir):
            print(f"Extracting {zip_path}…")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print("Extraction complete.")

        img_paths = []
        for root, _, files in os.walk(extract_dir):
            for f in sorted(files):
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_paths.append(os.path.join(root, f))
        img_paths = sorted(img_paths)

        if n_images is not None and n_images < len(img_paths):
            img_paths = rng.sample(img_paths, n_images)

        if len(img_paths) == 0:
            print(f"WARNING: No images found in {extract_dir}")

        super().__init__(img_paths, transform)

# ================================================================
# 5. AFHQ subclass loader: cat, dog, wild
# ================================================================
class AFHQClassDataset(BaseImageDataset):
    def __init__(self, root_dir, subclass, n_images=None, transform=None):
        subclass = subclass.lower()
        subclass_dir = os.path.join(root_dir, subclass)

        if not os.path.isdir(subclass_dir):
            raise ValueError(f"AFHQ subclass not found: {subclass_dir}")

        img_paths = sorted(
            os.path.join(subclass_dir, f)
            for f in os.listdir(subclass_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        if n_images is not None and n_images < len(img_paths):
            img_paths = rng.sample(img_paths, n_images)

        if len(img_paths) == 0:
            print(f"WARNING: No images found in {subclass_dir}")

        super().__init__(img_paths, transform)

# ================================================================
# 6. AFHQ generic (all subfolders)
# ================================================================
class AFHQ(BaseImageDataset):
    def __init__(self, root_dir, n_images=None, transform=None):
        img_paths = []
        for sub in sorted(os.listdir(root_dir)):
            sp = os.path.join(root_dir, sub)
            if os.path.isdir(sp):
                for f in sorted(os.listdir(sp)):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_paths.append(os.path.join(sp, f))

        img_paths = sorted(img_paths)

        if n_images is not None and n_images < len(img_paths):
            img_paths = rng.sample(img_paths, n_images)

        super().__init__(img_paths, transform)


# ================================================================
# 7. Prompts (Optimized)
# ================================================================
AFHQ_PROMPTS = {
    "cat":  "A photo of a cat",
    "dog":  "A photo of a dog",
    "wild": "A photo of a wild animal",
}

AFHQ_GENERIC_PROMPT = "A photo of a cat"

FFHQ_PROMPT = "A photo of a man with a beard"

COCO_PROMPT = "A photo of a real-world scene"

# ---------------------------
# Target prompts (identity preserving)
# ---------------------------
AFHQ_TARGET_PROMPTS = {
    "cat":  "A photo of a cat wearing sunglasses",
    "dog":  "A photo of a dog wearing sunglasses",
    "wild": "A photo of a wild animal wearing sunglasses",
}

AFHQ_GENERIC_TARGET = "A photo of an animal wearing sunglasses"

FFHQ_TARGET_PROMPT = "A photo of a man with sunglasses"

COCO_TARGET_PROMPT = "A photo of a real-world scene in the desert"

# -------------------------------------------------------------
# Identity-preserving target map for fallback single-image datasets
# -------------------------------------------------------------
TARGET_MAP = {
    "cat":        "A photo of a tiger",
    "aircraft":   "A photo of a old aircraft",
    "apple":      "A photo of a shiny red apple",
    "banana":     "A photo of a ripe banana on a table",
    "bedroom":    "A photo of a luxurious bedroom interior",
    "bee":        "A close-up photo of a bee on a flower",
    "bird":       "A photo of a bird in flight",
    "car":        "A photo of a modern car",
    "cherry":     "A photo of fresh cherries",
    "cup":        "A photo of a ceramic cup",
    "eagle":      "A photo of an eagle in flight",
    "face":       "A photo of a human face with open mouth",
    "flower":     "A photo of a colorful flower",
    "forest":     "A photo of a dense forest",
    "grape":      "A photo of a bunch of grapes",
    "horse":      "A photo of a running horse",
    "house":      "A photo of an old-fashioned mansion",
    "lion_tiger": "A photo of a tiger",
    "mountain":   "A photo of a snow-capped mountain",
    "panda":      "A photo of a panda eating bamboo",
    "peach":      "A photo of a fresh peach",
    "pumpkin":    "A photo of a pumpkin",
    "shoes":      "A photo of a pair of boots",
    "spider":     "A close-up photo of a spider on a web",
    "sushi":      "A photo of sushi on a plate",
    "tree":       "A photo of a large tree",
}

# ================================================================
# 8. Negative prompts (clean, not overly aggressive)
# ================================================================
GENERIC_N_PROMPT = (
    'text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'
)

AFHQ_N_PROMPT = GENERIC_N_PROMPT + ", unnatural fur texture, misaligned eyes"
FFHQ_N_PROMPT = GENERIC_N_PROMPT + ", unnatural facial features, misplaced eyes"
COCO_N_PROMPT = GENERIC_N_PROMPT + ", distorted objects, inconsistent perspective"


# ================================================================
# 9. Unified loader
# ================================================================
def load_dataset(name, n_images=None, image_size=768):
    transform = get_transform(image_size)
    name = name.lower()
    imgs = []
    
    # ---- Paths ----
    af_root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
    ffhq_zip = "/work3/fmry/Data/ffhq/00000-20251208T180936Z-3-001.zip"
    coco_zip = "/work3/fmry/Data/coco/train2017.zip"
    
    # ---- Dataset selection ----
    if name == "ffhq":
        ds = ZipImageDataset(ffhq_zip, n_images=n_images, transform=transform)
        prompt = FFHQ_PROMPT
        target_prompt = FFHQ_TARGET_PROMPT
        n_prompt = FFHQ_N_PROMPT
        
    elif name.startswith("afhq"):
        if name == "afhq-cat":
            cls = "cat"
        elif name == "afhq-dog":
            cls = "dog"
        elif name == "afhq-wild":
            cls = "wild"
        else:
            cls = None
        
        if cls is not None:
            ds = AFHQClassDataset(af_root, cls, n_images=n_images, transform=transform)
            prompt = AFHQ_PROMPTS[cls]
            target_prompt = AFHQ_TARGET_PROMPTS[cls]
            n_prompt = AFHQ_N_PROMPT
        else:
            ds = AFHQ(af_root, n_images=n_images, transform=transform)
            prompt = AFHQ_GENERIC_PROMPT
            target_prompt = AFHQ_GENERIC_TARGET
            n_prompt = AFHQ_N_PROMPT
            
    elif name == "coco":
        ds = ZipImageDataset(coco_zip, n_images=n_images, transform=transform)
        prompt = COCO_PROMPT
        target_prompt = COCO_TARGET_PROMPT
        n_prompt = COCO_N_PROMPT
    
    # ---- Sample image dataset for single images ----
    else:
        # fallback for image list datasets
        sample_imgs = {
            "cat": ["cat1.png", "cat2.png"],
            "aircraft": ["aircraft1.png", "aircraft2.png"],
            "apple": ["apple1.png", "apple2.png"],
            "banana": ["banana1.png", "banana2.png"],
            "bedroom": ["bedroom1.png", "bedroom2.png"],
            "bee": ["bee1.png", "bee2.png"],
            "bird": ["bird1.png", "bird3.png"],
            "car": ["car1.png", "car2.png"],
            "cherry": ["cherry1.png", "cherry2.png"],
            "cup": ["cup1.png", "cup2.png"],
            "eagle": ["eagle1.png", "eagle2.png"],
            "face": ["face1.png", "face2.png"],
            "flower": ["flower1.png", "flower2.png"],
            "forest": ["forest1.png", "forest2.png"],
            "grape": ["grape1.png", "grape2.png"],
            "horse": ["horse1.png", "horse2.png"],
            "house": ["house1.png", "house2.png"],
            "lion_tiger": ["lion_tiger1.png", "lion_tiger2.png"],
            "mountain": ["moutain1.png", "moutain2.png"],
            "panda": ["panda1.png", "panda2.png"],
            "peach": ["peach1.png", "peach2.png"],
            "pumpkin": ["pumpkin1.png", "pumpkin2.png"],
            "shoes": ["shoes1.png", "shoes2.png"],
            "spider": ["spider1.png", "spider2.png"],
            "sushi": ["sushi1.png", "sushi2.png"],
            "tree": ["tree1.png", "tree2.png"],
        }

        if name not in sample_imgs:
            raise ValueError(f"Unknown dataset: {name}")
        
        # Load static image files
        file_list = sample_imgs[name]
        for f in file_list:
            img = Image.open(f'sample_imgs/{f}').resize((image_size, image_size))
            imgs.append(img)

        # Prompts
        prompt = f"A photo of {name.replace('_', ' ')}"
        target_prompt = TARGET_MAP[name]
        n_prompt = GENERIC_N_PROMPT
        
    # ---- Load dataset images ----
    if 'ds' in locals():
        for pil, _ in ds:
            imgs.append(pil.resize((image_size, image_size)))
    else:
        ds = None
    
    return imgs, prompt, target_prompt, n_prompt, ds




# ================================================================
# 9. Test run
# ================================================================
#if __name__ == "__main__":
#    ds, prompt = load_dataset("afhq-cat", n_images=5, image_size=768)
#    print("AFHQ cat:", len(ds), "Prompt:", prompt)
#
#    ds, prompt = load_dataset("ffhq", n_images=3)
#    print("FFHQ:", len(ds), "Prompt:", prompt)
#
#    ds, prompt = load_dataset("coco", n_images=3)
#    print("COCO:", len(ds), "Prompt:", prompt)

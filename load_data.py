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
class ZipImageDataset(Dataset):
    def __init__(self, zip_path, n_images=None, transform=None):
        extract_dir = os.path.splitext(zip_path)[0]
        if not os.path.exists(extract_dir):
            print(f"Extracting {zip_path} …")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            print("Extraction complete.")
        
        self.transform = transform or get_transform()
        
        # Recursively find all image files
        self.img_paths = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.img_paths.append(os.path.join(root, f))
        
        if n_images is not None and n_images < len(self.img_paths):
            self.img_paths = random.sample(self.img_paths, n_images)
        
        if len(self.img_paths) == 0:
            print(f"Warning: No images found in {extract_dir}")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img)
    
#%% Subclass loader

class AFHQClassDataset(Dataset):
    def __init__(self, root_dir, subclass, n_images=None, transform=None):
        subclass = subclass.lower()
        subclass_dir = os.path.join(root_dir, subclass)

        if not os.path.isdir(subclass_dir):
            raise ValueError(f"AFHQ subclass not found: {subclass_dir}")

        self.transform = transform or get_transform()
        
        # collect images
        self.img_paths = [
            os.path.join(subclass_dir, f)
            for f in os.listdir(subclass_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        # optional sampling
        if n_images is not None and n_images < len(self.img_paths):
            self.img_paths = random.sample(self.img_paths, n_images)

        if len(self.img_paths) == 0:
            print(f"Warning: No images found in {subclass_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        return self.transform(img)


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

    name = name.lower()
    
    AFHQ_PROMPTS = {
    "cat":  "A high-quality portrait photo of a cat, detailed fur, natural lighting, realistic and sharp.",
    "dog":  "A high-quality portrait photo of a dog, detailed fur, natural lighting, realistic and sharp.",
    "wild": "A high-quality portrait photo of a wild animal, natural fur texture, realistic and detailed.",
    }
    AFHQ_GENERIC_PROMPT = "A high-quality close-up portrait of an animal, natural fur texture, detailed and realistic."


    # --- FFHQ ---
    if name == "ffhq":
        zip_path = "/work3/fmry/Data/ffhq/00000-20251208T180936Z-3-001.zip"
        dataset = ZipImageDataset(zip_path, n_images=n_images, transform=transform)
        prompt = "A high-quality portrait photo of a human face, natural lighting, sharp details, realistic skin texture."
        return dataset, prompt

    # --- AFHQ subclasses ---
    elif name in ["afhq-cat", "cat"]:
        root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        dataset = AFHQClassDataset(root, "cat", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["cat"]
        return dataset, prompt

    elif name in ["afhq-dog", "dog"]:
        root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        dataset = AFHQClassDataset(root, "dog", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["dog"]
        return dataset, prompt

    elif name in ["afhq-wild", "wild"]:
        root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        dataset = AFHQClassDataset(root, "wild", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["wild"]
        return dataset, prompt

    # --- Generic AFHQ ---
    elif name == "afhq":
        root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
        dataset = AFHQ(root, n_images=n_images, transform=transform)
        prompt = AFHQ_GENERIC_PROMPT
        return dataset, prompt

    # --- COCO ---
    elif name == "coco":
        zip_path = "/work3/fmry/Data/coco/train2017.zip"
        dataset = ZipImageDataset(zip_path, n_images=n_images, transform=transform)
        prompt = "A detailed photograph of an everyday real-world scene, natural lighting, realistic colors, high-quality."
        return dataset, prompt

    else:
        raise ValueError(f"Unknown dataset: {name}")


# -----------------------------
# 6. Example usage
# -----------------------------
if __name__ == "__main__":
    ds, prompt = load_dataset("afhq-cat", n_images=8, image_size=256)
    print("AFHQ-cat length:", len(ds))
    print("Prompt:", prompt)

    ds, prompt = load_dataset("ffhq", n_images=4)
    print("FFHQ length:", len(ds))
    print("Prompt:", prompt)

    ds, prompt = load_dataset("coco", n_images=6)
    print("COCO length:", len(ds))
    print("Prompt:", prompt)


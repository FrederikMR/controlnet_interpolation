import os
import zipfile
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

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
        img_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        if n_images is not None and n_images < len(img_paths):
            img_paths = random.sample(img_paths, n_images)

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
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_paths.append(os.path.join(root, f))

        if n_images is not None and n_images < len(img_paths):
            img_paths = random.sample(img_paths, n_images)

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

        img_paths = [
            os.path.join(subclass_dir, f)
            for f in os.listdir(subclass_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if n_images is not None and n_images < len(img_paths):
            img_paths = random.sample(img_paths, n_images)

        if len(img_paths) == 0:
            print(f"WARNING: No images found in {subclass_dir}")

        super().__init__(img_paths, transform)


# ================================================================
# 6. AFHQ generic (all subfolders)
# ================================================================
class AFHQ(BaseImageDataset):
    def __init__(self, root_dir, n_images=None, transform=None):
        img_paths = []
        for sub in os.listdir(root_dir):
            sp = os.path.join(root_dir, sub)
            if os.path.isdir(sp):
                for f in os.listdir(sp):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_paths.append(os.path.join(sp, f))

        if n_images is not None and n_images < len(img_paths):
            img_paths = random.sample(img_paths, n_images)

        super().__init__(img_paths, transform)


# ================================================================
# 7. Prompts
# ================================================================
AFHQ_PROMPTS = {
    "cat":  "A high-quality portrait photo of a cat, detailed fur, natural lighting, realistic and sharp.",
    "dog":  "A high-quality portrait photo of a dog, detailed fur, natural lighting, realistic and sharp.",
    "wild": "A high-quality portrait photo of a wild animal, natural fur texture, realistic and detailed.",
}

AFHQ_GENERIC_PROMPT = (
    "A high-quality close-up portrait of an animal, natural fur texture, detailed and realistic."
)

FFHQ_PROMPT = (
    "A high-quality portrait photo of a human face, natural lighting, sharp details, realistic skin texture."
)

COCO_PROMPT = (
    "A detailed photograph of an everyday real-world scene, natural lighting, realistic colors, high-quality."
)


# ================================================================
# 8. Unified loader
# ================================================================
def load_dataset(name, n_images=None, image_size=768):
    transform = get_transform(image_size)
    name = name.lower()

    # ---- FFHQ ----
    if name == "ffhq":
        zip_path = "/work3/fmry/Data/ffhq/00000-20251208T180936Z-3-001.zip"
        ds = ZipImageDataset(zip_path, n_images=n_images, transform=transform)
        return ds, FFHQ_PROMPT

    # ---- AFHQ subclasses ----
    root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"

    if name in ("afhq-cat", "cat"):
        ds = AFHQClassDataset(root, "cat", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["cat"]

    if name in ("afhq-dog", "dog"):
        ds = AFHQClassDataset(root, "dog", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["dog"]

    if name in ("afhq-wild", "wild"):
        ds = AFHQClassDataset(root, "wild", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["wild"]

    # ---- Generic AFHQ ----
    if name == "afhq":
        ds = AFHQ(root, n_images=n_images, transform=transform)
        prompt = AFHQ_GENERIC_PROMPT

    # ---- COCO ----
    if name == "coco":
        zip_path = "/work3/fmry/Data/coco/train2017.zip"
        ds = ZipImageDataset(zip_path, n_images=n_images, transform=transform)
        prompt = COCO_PROMPT
        
        
    imgs = []
    for pil, _ in ds:
        imgs.append(pil)
        
        
    return imgs, prompt

    
    

    raise ValueError(f"Unknown dataset: {name}")


# ================================================================
# 9. Test run
# ================================================================
if __name__ == "__main__":
    ds, prompt = load_dataset("afhq-cat", n_images=5, image_size=768)
    print("AFHQ cat:", len(ds), "Prompt:", prompt)

    ds, prompt = load_dataset("ffhq", n_images=3)
    print("FFHQ:", len(ds), "Prompt:", prompt)

    ds, prompt = load_dataset("coco", n_images=3)
    print("COCO:", len(ds), "Prompt:", prompt)

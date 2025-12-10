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
# 7. Prompts (updated)
# ================================================================
AFHQ_PROMPTS = {
    "cat":  "A high-quality close-up portrait of a cat, realistic fur texture, natural lighting, single pair of eyes, anatomically correct, centered composition.",
    "dog":  "A high-quality close-up portrait of a dog, realistic fur texture, natural lighting, single pair of eyes, anatomically correct, centered composition.",
    "wild": "A high-quality close-up portrait of a wild animal, realistic fur or skin texture, natural lighting, single pair of eyes, anatomically correct, centered composition.",
}

AFHQ_GENERIC_PROMPT = (
    "A high-quality close-up portrait of an animal, realistic fur or skin texture, natural lighting, single pair of eyes, anatomically correct, centered composition."
)

FFHQ_PROMPT = (
    "A high-quality realistic human face portrait, close-up, perfectly symmetrical, single pair of eyes, natural lighting, anatomically correct features, photorealistic skin texture."
)

COCO_PROMPT = (
    "A high-quality photograph of a real-world scene, perspective-correct, natural lighting, realistic colors, sharp details, centered composition, no distortions."
)

AFHQ_TARGET_PROMPTS = {
    "cat":  "A majestic fluffy cat, high-quality close-up, expressive eyes, realistic fur texture, natural lighting, anatomically correct, centered composition.",
    "dog":  "A well-groomed dog, high-quality close-up, expressive face, realistic fur texture, natural lighting, anatomically correct, centered composition.",
    "wild": "A powerful wild animal, high-quality close-up, expressive eyes, realistic fur or skin texture, natural lighting, anatomically correct, centered composition.",
}

AFHQ_GENERIC_TARGET = (
    "A majestic close-up portrait of an animal with expressive eyes, realistic anatomy, natural lighting, and detailed fur or skin texture."
)

FFHQ_TARGET_PROMPT = (
    "A stunning cinematic close-up portrait of a human face, perfectly symmetrical, single pair of eyes, natural lighting, anatomically correct features, photorealistic, highly detailed and expressive."
)

COCO_TARGET_PROMPT = (
    "A highly aesthetic cinematic photograph of a real-world scene, perspective-correct, natural lighting, realistic colors, sharp details, centered composition, no distortions."
)

TARGET_MAP = {
    "cat":            "a majestic fluffy cat with bright expressive eyes",
    "aircraft":       "a futuristic high-tech aircraft flying under dramatic natural lighting",
    "apple":          "a perfectly ripe glossy apple with vibrant colors",
    "banana":         "a fresh banana with rich texture, natural lighting",
    "bedroom":        "a luxurious modern bedroom interior with warm lighting",
    "bee":            "a highly detailed macro photo of a bee on a flower, realistic textures",
    "bird":           "a majestic bird in flight with vibrant feathers, natural lighting",
    "car":            "a luxury sports car with sleek aerodynamic design, natural lighting",
    "cherry":         "a bowl of glossy ripe cherries with natural lighting",
    "cup":            "an elegant ceramic teacup under soft natural lighting",
    "eagle":          "A powerful eagle in flight, wings spread, dramatic natural lighting",
    "face":           "A cinematic close-up of a face, expressive details, natural lighting, perfectly symmetrical",
    "flower":         "A vibrant blooming flower with delicate petals, natural lighting",
    "forest":         "A mystical forest scene with soft volumetric light, realistic textures",
    "grape":          "A bunch of glossy grapes with dew drops, natural lighting",
    "horse":          "A majestic horse running through a field, natural lighting",
    "house":          "A luxurious mansion exterior, realistic architecture, natural lighting",
    "lion_tiger":     "A majestic big cat with powerful features, dramatic natural lighting",
    "mountain":       "A snow-capped mountain under dramatic sky, realistic textures, perspective-correct",
    "panda":          "A cute fluffy panda in bamboo forest, realistic fur, natural lighting",
    "peach":          "A perfectly ripe peach with velvety skin, realistic textures, natural lighting",
    "pumpkin":        "A large vibrant pumpkin, warm autumn lighting, realistic textures",
    "shoes":          "Luxury designer shoes displayed in a stylish scene, natural lighting, realistic textures",
    "spider":         "A detailed macro shot of a spider on a web, realistic textures, natural lighting",
    "sushi":          "A beautifully arranged sushi platter, soft natural lighting, realistic textures",
    "tree":           "An ancient majestic tree with sprawling branches, natural lighting, realistic textures",
}

# ================================================================
# 8. Negative prompts
# ================================================================
GENERIC_N_PROMPT = (
    "text, watermark, logo, signature, distorted, mutated, extra limbs, extra eyes, duplicate features, disfigured, broken anatomy, blurry, low-resolution, poorly drawn, low quality, warped, melted, glitch, unrealistic, ugly, oversaturated"
)

AFHQ_N_PROMPT = GENERIC_N_PROMPT + ", unnatural fur or skin texture, misaligned eyes, messy composition"
FFHQ_N_PROMPT = GENERIC_N_PROMPT + ", unnatural facial features, lopsided face, misplaced eyes, messy hair, broken anatomy"
COCO_N_PROMPT  = GENERIC_N_PROMPT + ", distorted objects, floating parts, inconsistent perspective, unnatural lighting, unnatural shadows"

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
        prompt, target_prompt, n_prompt = FFHQ_PROMPT, FFHQ_TARGET_PROMPT, FFHQ_N_PROMPT
        
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
            prompt, target_prompt, n_prompt = AFHQ_PROMPTS[cls], AFHQ_TARGET_PROMPTS[cls], AFHQ_N_PROMPT
        else:
            ds = AFHQ(af_root, n_images=n_images, transform=transform)
            prompt, target_prompt, n_prompt = AFHQ_GENERIC_PROMPT, AFHQ_GENERIC_TARGET, AFHQ_N_PROMPT
            
    elif name == "coco":
        ds = ZipImageDataset(coco_zip, n_images=n_images, transform=transform)
        prompt, target_prompt, n_prompt = COCO_PROMPT, COCO_TARGET_PROMPT, COCO_N_PROMPT
    
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
            "mountain": ["mountain1.png", "mountain2.png"],
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
        
        file_list = sample_imgs[name]
        for f in file_list:
            img = Image.open(f'sample_imgs/{f}').resize((image_size, image_size))
            imgs.append(img)
        prompt = f'a photo of {name}'
        target_prompt = TARGET_MAP[name]
        n_prompt = GENERIC_N_PROMPT
        
    # ---- Load dataset images ----
    if 'ds' in locals():
        for pil, _ in ds:
            imgs.append(pil)
    
    return imgs, prompt, target_prompt, n_prompt



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

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
    "cat":  "A high-quality portrait of a cat, realistic fur texture, natural lighting, sharp details, anatomically correct, single pair of eyes, realistic ears.",
    "dog":  "A high-quality portrait of a dog, realistic fur texture, natural lighting, sharp details, anatomically correct, single pair of eyes, realistic ears.",
    "wild": "A high-quality portrait of a wild animal, realistic fur/skin texture, natural lighting, sharp details, anatomically correct, single pair of eyes.",
}

AFHQ_GENERIC_PROMPT = (
    "A high-quality close-up portrait of an animal, realistic fur or skin texture, detailed and anatomically correct, single pair of eyes, natural lighting."
)

FFHQ_PROMPT = (
    "A high-quality realistic human face portrait, perfectly symmetrical, single pair of eyes, normal ears, natural lighting, sharp facial details, realistic skin texture, photorealistic, no extra facial features."
)

COCO_PROMPT = (
    "A detailed photograph of an everyday real-world scene, perspective-correct, natural lighting, realistic colors, high-quality, no distorted objects."
)

AFHQ_TARGET_PROMPTS = {
    "cat":  "A majestic fluffy cat with vibrant fur colors, expressive eyes, realistic ears, anatomically correct, beautiful lighting.",
    "dog":  "A well-groomed dog with elegant fur texture, expressive face, realistic anatomy, warm cinematic lighting.",
    "wild": "A powerful wild animal in dramatic natural lighting, detailed fur/skin, anatomically correct, intense presence.",
}

AFHQ_GENERIC_TARGET = (
    "A majestic detailed portrait of an animal with expressive eyes, realistic anatomy, and beautiful lighting."
)

FFHQ_TARGET_PROMPT = (
    "A stunning cinematic portrait of a human face, perfectly symmetrical, single pair of eyes, normal ears, dramatic lighting, expressive details, highly aesthetic."
)

COCO_TARGET_PROMPT = (
    "A highly aesthetic and cinematic photograph of a real-world scene, perspective-correct, with dramatic lighting and rich realistic colors."
)

TARGET_MAP = {
    "cat":            "a majestic fluffy cat with bright expressive eyes",
    "aircraft":       "a futuristic high-tech aircraft flying in dramatic lighting",
    "apple":          "a perfectly ripe glossy apple with vibrant colors",
    "banana":         "a beautifully lit fresh banana with rich texture",
    "bedroom":        "a luxurious modern bedroom interior with warm lighting",
    "bee":            "a highly detailed macro photo of a bee on a flower",
    "bird":           "a majestic bird in flight with vibrant feathers",
    "car":            "a luxury sports car with sleek aerodynamic design",
    "cherry":         "a bowl of shiny ripe cherries with beautiful reflections",
    "cup":            "an elegant ceramic teacup in soft natural lighting",
    "eagle":          "a powerful eagle with its wings spread in dramatic light",
    "face":           "a stunning cinematic portrait of a face with dramatic shadows",
    "flower":         "a vibrant blooming flower with delicate petals",
    "forest":         "a mystical enchanted forest with soft volumetric light",
    "grape":          "a bunch of glossy grapes with dew drops",
    "horse":          "a majestic horse running through a field",
    "house":          "a beautiful luxurious old-fashioned mansion",
    "lion_tiger":     "a majestic big cat with powerful features and intense lighting",
    "mountain":       "an epic snow-capped mountain under dramatic sky",
    "panda":          "a cute fluffy panda in a bamboo forest",
    "peach":          "a perfectly ripe peach with soft velvety skin",
    "pumpkin":        "a large vibrant pumpkin in warm autumn light",
    "shoes":          "luxury designer shoes displayed in a stylish scene",
    "spider":         "a detailed macro shot of a spider weaving a web",
    "sushi":          "a beautifully arranged sushi platter in soft lighting",
    "tree":           "an ancient majestic tree with sprawling branches",
}



# ================================================================
# 8. Unified loader
# ================================================================
def load_dataset(name, n_images=None, image_size=768):
    transform = get_transform(image_size)
    name = name.lower()
    root = "/work3/fmry/Data/afhq/stargan-v2/data/train/"
    # ---- FFHQ ----
    if name == "ffhq":
        zip_path = "/work3/fmry/Data/ffhq/00000-20251208T180936Z-3-001.zip"
        ds = ZipImageDataset(zip_path, n_images=n_images, transform=transform)
        prompt, target_prompt = FFHQ_PROMPT, FFHQ_TARGET_PROMPT
        
        imgs = []
        for pil, _ in ds:
            imgs.append(pil)

    elif name == "afhq-cat":
        ds = AFHQClassDataset(root, "cat", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["cat"]
        target_prompt = AFHQ_TARGET_PROMPTS["cat"]
        print(image_size)
        imgs = []
        for pil, _ in ds:
            imgs.append(pil)
            import numpy as np
            print(np.array(pil).shape)
        
    elif name  == "afhq-dog":
        ds = AFHQClassDataset(root, "dog", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["dog"]
        target_prompt = AFHQ_TARGET_PROMPTS["dog"]
        
        imgs = []
        for pil, _ in ds:
            imgs.append(pil)
        
    elif name == "afhq-wild":
        ds = AFHQClassDataset(root, "wild", n_images=n_images, transform=transform)
        prompt = AFHQ_PROMPTS["wild"]
        target_prompt = AFHQ_TARGET_PROMPTS["wild"]
        
        imgs = []
        for pil, _ in ds:
            imgs.append(pil)
        
    elif name == "afhq":
        ds = AFHQ(root, n_images=n_images, transform=transform)
        prompt, target_prompt = AFHQ_GENERIC_PROMPT, AFHQ_GENERIC_TARGET
        
        imgs = []
        for pil, _ in ds:
            imgs.append(pil)
            
    elif name == "coco":
        zip_path = "/work3/fmry/Data/coco/train2017.zip"
        ds = ZipImageDataset(zip_path, n_images=n_images, transform=transform)
        prompt, target_prompt = COCO_PROMPT, COCO_TARGET_PROMPT
        
        imgs = []
        for pil, _ in ds:
            imgs.append(pil)
        
    elif name == "cat":
        img1 = Image.open('sample_imgs/cat1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/cat2.png').resize((image_size, image_size))
        prompt='a photo of cat'
        target_prompt = TARGET_MAP[name]
        imgs = [img1, img2]
    elif name == "cat":
        img1 = Image.open('sample_imgs/cat1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/cat2.png').resize((image_size, image_size))
        prompt='a photo of cat'
        target_prompt = TARGET_MAP[name]
        imgs = [img1, img2]
    elif name == "aircraft":
        img1 = Image.open('sample_imgs/aircraft1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/aircraft2.png').resize((image_size, image_size))
        prompt='a photo of aircraft'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
    elif name == "apple":
        img1 = Image.open('sample_imgs/apple1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/apple2.png').resize((image_size, image_size))
        
        prompt='a photo of apple'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
    elif name == "banana":
        
        img1 = Image.open('sample_imgs/banana1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/banana2.png').resize((image_size, image_size))
        
        prompt='a photo of banana'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "bedroom":
        
        img1 = Image.open('sample_imgs/bedroom1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/bedroom2.png').resize((image_size, image_size))

        prompt='a photo of bed'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "bee":
        
        img1 = Image.open('sample_imgs/bee1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/bee2.png').resize((image_size, image_size))
        
        prompt='a photo,bee,wasp'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "bird":
        
        img1 = Image.open('sample_imgs/bird1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/bird3.png').resize((image_size, image_size))
        
        prompt='a photo of bird'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "car":
        
        img1 = Image.open('sample_imgs/car1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/car2.png').resize((image_size, image_size))
        
        prompt='a photo of car'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "cherry":
        
        img1 = Image.open('sample_imgs/cherry1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/cherry2.png').resize((image_size, image_size))
        
        prompt='a photo of cherry'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "cup":
        
        img1 = Image.open('sample_imgs/cup1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/cup2.png').resize((image_size, image_size))
        
        prompt='a photo of cup'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "eagle":
        
        img1 = Image.open('sample_imgs/eagle1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/eagle2.png').resize((image_size, image_size))
        
        prompt='eagle'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "face":
        
        img1 = Image.open('sample_imgs/face1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/face2.png').resize((image_size, image_size))
        
        prompt = 'a photo of face'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "flower":
        
        img1 = Image.open('sample_imgs/flower1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/flower2.png').resize((image_size, image_size))
        prompt='a photo of flower'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
    
    elif name == "forest":
        
        img1 = Image.open('sample_imgs/forest1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/forest2.png').resize((image_size, image_size))
        
        prompt='a photo of forest'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "grape":
        
        img1 = Image.open('sample_imgs/grape1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/grape2.png').resize((image_size, image_size))
        
        prompt = 'a photo of grape'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "horse":
        
        img1 = Image.open('sample_imgs/horse1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/horse2.png').resize((image_size, image_size))
        prompt='a photo of a horse'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "house":
        
        img1 = Image.open('sample_imgs/house1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/house2.png').resize((image_size, image_size))
        
        prompt='a photo of house'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "lion_tiger":
        
        img1 = Image.open('sample_imgs/lion_tiger1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/lion_tiger2.png').resize((image_size, image_size))
        
        prompt = "a photo of a lion's face,a photo of a tiger's face"
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "mountain":
        
        img1 = Image.open('sample_imgs/moutain1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/moutain2.png').resize((image_size, image_size))
        
        prompt='a photo of moutain and lake'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "panda":
        
        img1 = Image.open('sample_imgs/panda1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/panda2.png').resize((image_size, image_size))
        
        prompt='a photo of panda'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "peach":
        
        img1 = Image.open('sample_imgs/peach1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/peach2.png').resize((image_size, image_size))
        
        prompt = 'a photo of fruit,peach'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "peach":
        
        img1 = Image.open('sample_imgs/pumpkin1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/pumpkin2.png').resize((image_size, image_size))
        
        prompt='a photo of pumpkin'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "shoes":
        
        img1 = Image.open('sample_imgs/shoes1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/shoes2.png').resize((image_size, image_size))
        
        prompt='shoes'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "spider":
        
        img1 = Image.open('sample_imgs/spider1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/spider2.png').resize((image_size, image_size))
        
        prompt='a photo of spider'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "sushi":
        
        img1 = Image.open('sample_imgs/sushi1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/sushi2.png').resize((image_size, image_size))
        
        prompt='a photo of sushi'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    elif name == "tree":
        
        img1 = Image.open('sample_imgs/tree1.png').resize((image_size, image_size))
        img2 = Image.open('sample_imgs/tree2.png').resize((image_size, image_size))
        
        prompt='a photo of tree'
        target_prompt = TARGET_MAP[name]
        
        imgs = [img1, img2]
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
        
    n_prompt='text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'
        
    return imgs, prompt, target_prompt, n_prompt


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

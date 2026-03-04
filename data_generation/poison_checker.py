import os
import pickle
import numpy as np
from PIL import Image

def load_image(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    img = data["img"]

    # PIL -> numpy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # float to unit8
    if img.dtype != np.uint8:
        img = ((img + 1) * 127.5).clip(0,255).astype(np.uint8)

    return img


clean_dir = "data"
poison_dir = "output_data/dog_features"

files = [f for f in os.listdir(poison_dir) if f.endswith(".p")]

for f in files[:5]:   # check first 5
    clean_img = load_image(os.path.join(clean_dir, f))
    poison_img = load_image(os.path.join(poison_dir, f))

    print(f"\nChecking {f}")
    print("Same shape:", clean_img.shape == poison_img.shape)

    diff = np.abs(clean_img.astype(np.int32) - poison_img.astype(np.int32))
    print("Max pixel difference:", diff.max())
    print("Mean pixel difference:", diff.mean())
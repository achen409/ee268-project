import os
import pickle
import numpy as np
from PIL import Image

# path to dir to check
outdir = "output_data/poisoned_dog"
# outdir = "output_data/dog_features"
# outdir = "data"

# list all .p files
files = sorted([f for f in os.listdir(outdir) if f.endswith(".p")])
print(f"Found {len(files)} files.")

if len(files) == 0:
    print("No output files found. Something went wrong.")
    exit()

# load first file as a test
test_file = os.path.join(outdir, files[100])
with open(test_file, "rb") as f:
    data = pickle.load(f)

# check keys
print("Keys in pickle:", list(data.keys()))

# check types
print("Type of 'text':", type(data["text"]))
print("Type of 'img':", type(data["img"]))

# print text
print(f"Image Text: {data["text"]}")

# show image
img = data["img"]
if isinstance(img, Image.Image):
    img.show()
elif isinstance(img, np.ndarray):
    print("Image shape:", img.shape)
    print("Image dtype:", img.dtype)

    if img.dtype != np.uint8:
        img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    pil_img.show()
else:
    print("img is not a PIL.Image.Image. Type:", type(img))
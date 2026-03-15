import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

data_dir = "../data_generation/data"
rows = 4
cols = 5

files = sorted([f for f in os.listdir(data_dir) if f.endswith(".p")])
files = files[:rows * cols]

fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
axes = axes.flatten()

for i, f in enumerate(files):
    with open(os.path.join(data_dir, f), "rb") as file:
        data = pickle.load(file)

    img = data["img"]
    text = data["text"]

    # convert numpy to PIL
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = ((img + 1) * 127.5).clip(0,255).astype(np.uint8)
        img = Image.fromarray(img)

    img = img.convert("RGB")

    axes[i].imshow(img)
    axes[i].axis("off")

    # shorten long captions
    wrapped_text = "\n".join(textwrap.wrap(text, width=30))

    axes[i].set_title(f"{wrapped_text}", fontsize=12, pad=10)

plt.subplots_adjust(
    left=0.05,
    right=0.95,
    top=0.95,
    bottom=0.05,
    wspace=0.2,     # horizontal spacing
    hspace=0.4      # vertical spacing
)

plt.savefig("laion_dog_collage.png", dpi=300)
plt.show()

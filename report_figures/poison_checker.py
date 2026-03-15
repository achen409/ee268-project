import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(path, debug=False):
    with open(path, "rb") as f:
        data = pickle.load(f)

    if debug:
        txt = data["text"]
        print(txt)
    img = data["img"]

    # PIL -> numpy
    if isinstance(img, Image.Image):
        img = np.array(img)

    # float to unit8
    if img.dtype != np.uint8:
        img = ((img + 1) * 127.5).clip(0,255).astype(np.uint8)

    return img


clean_dir = "../data_generation/output_data/dog_features"
poison_dir = "../data_generation/output_data/poisoned_dog"
target_path = "../data_generation/target.png"

files = [f for f in os.listdir(poison_dir) if f.endswith(".p")]

# for f in files[:5]:   # check first 5
#     clean_img = load_image(os.path.join(clean_dir, f))
#     poison_img = load_image(os.path.join(poison_dir, f))

#     print(f"\nChecking {f}")
#     print("Same shape:", clean_img.shape == poison_img.shape)

#     diff = np.abs(clean_img.astype(np.int32) - poison_img.astype(np.int32))
#     print("Max pixel difference:", diff.max())
#     print("Mean pixel difference:", diff.mean())

# choose one example
f = files[0]
debug_bool = True

clean_img = load_image(os.path.join(clean_dir, f), debug=debug_bool)
poison_img = load_image(os.path.join(poison_dir, f), debug=debug_bool)

diff = np.abs(clean_img.astype(np.int32) - poison_img.astype(np.int32))

max_diff = diff.max()
mean_diff = diff.mean()

print("Max pixel difference:", max_diff)
print("Mean pixel difference:", mean_diff)

# load target perturbation image
target_img = Image.open(target_path).convert("RGB")

# ====================================================================
# visualization
fig, axes = plt.subplots(1,4, figsize=(16,4))

axes[0].imshow(clean_img)
axes[0].set_title("Clean Image", fontsize=12, pad=10)
axes[0].axis("off")

axes[1].imshow(target_img)
axes[1].set_title("Target Concept", fontsize=12, pad=10)
axes[1].axis("off")

axes[2].imshow(poison_img)
axes[2].set_title("Poisoned Image", fontsize=12, pad=10)
axes[2].axis("off")

diff_visible = (diff * 8).clip(0,255).astype(np.uint8)
axes[3].imshow(diff_visible.astype(np.uint8))
axes[3].set_title("Pixel Difference (Amplified)", fontsize=12, pad=10)
axes[3].axis("off")

plt.suptitle(
    f"Pixel Stats:\nMax Pixel Difference = {max_diff} \nMean Pixel Difference = {mean_diff:.3f}",
    fontsize=14
)

plt.tight_layout()
plt.savefig("clean_poison_target.png", dpi=300)
plt.show()
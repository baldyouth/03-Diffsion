import torch
import imageio

from datasets import load_dataset
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler

def download_dataset():
    dataset_name = "korexyz/celeba-hq-256x256"
    load_dataset(dataset_name, split="train", cache_dir="/home/swjtu/workspace_01/data/celeba-hq-256x256")

def load_dataset_test():
    dataset_name = "../data/celeba-hq-256x256"
    dataset = load_dataset(dataset_name, split="train")
    _, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:4]["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()
    plt.show()

def simulate_noise_animation(image, steps=1000):
    noise_scheduler=DDPMScheduler(num_train_timesteps=steps)
    noise = torch.randn_like(image)
    images = []

    for t in range(0, steps, 200):
        timestep = torch.tensor([t], dtype=torch.long)
        noisy_image = noise_scheduler.add_noise(image, noise, timestep)

        img = (noisy_image / 2 + 0.5).clamp(0, 1)
        img = img[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")

        images.append(img)
    
    grid_size=(2, 5)
    _, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 8))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < len(images):
            ax.imshow(images[i])
            ax.axis("off")
            ax.set_title(f"t={i * 200}")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig('simulate_noise_animation.jpg')
    print(f"JPG 已保存为：{'simulate_noise_animation.jpg'}")

    imageio.mimsave("simulate_noise_animation.gif", images, loop=0, fps=1)
    print(f"GIF 已保存为：{'simulate_noise_animation.gif'}")
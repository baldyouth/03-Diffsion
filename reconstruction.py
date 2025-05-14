from PIL import Image
import torch
from diffusers import DDPMScheduler, UNet2DModel
from datasets import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import math

def denorm(tensor):
    # [-1, 1] => [0, 1]
    return (tensor.clamp(-1, 1) + 1) / 2

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

model_name = "diffusion-test/unet"
scheduler_name = "diffusion-test/scheduler"

model = UNet2DModel.from_pretrained(model_name, use_safetensors=True).to("cuda")
scheduler = DDPMScheduler.from_pretrained(scheduler_name, use_safetensors=False)

sample_size = model.config.sample_size

dataset_name = "../data/celeba-hq-256x256"
dataset = load_dataset(dataset_name, split="validation")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}
dataset.set_transform(transform)

image_num = 2
batch = dataset[0:image_num]["images"]
sample_images = torch.stack(batch, dim=0).to("cuda")

# add_noise
T = 500
scheduler.set_timesteps(T)
noise = torch.randn(sample_images.shape).to("cuda")
timesteps = torch.LongTensor([T]).to("cuda")
noisy_images = scheduler.add_noise(sample_images, noise, timesteps)

# reconstruction
reverse_timesteps = scheduler.timesteps[scheduler.timesteps <= T]
input_images = noisy_images.clone()
for timestep in reverse_timesteps:
    with torch.no_grad():
        t_batch = torch.full((input_images.shape[0],), timestep, device="cuda", dtype=torch.long)
        noise_pred = model(input_images, t_batch).sample
        input_images = scheduler.step(noise_pred, timestep, input_images).prev_sample
reconstructed_images = input_images

# reconstructed_images = []
# for i in range(len(sample_images)):
#     input_i = noisy_images[i:i+1]
#     for timestep in reverse_timesteps:
#         with torch.no_grad():
#             noisy_residual = model(input_i, timestep).sample
#             input_i = scheduler.step(noisy_residual, timestep, input_i).prev_sample
#     reconstructed_images.append(input_i.squeeze(0).detach())

# reconstructed_images = torch.stack(reconstructed_images, dim=0)

# calc loss
mse_list = []
psnr_list = []

plt.figure(figsize=(12, 4 * len(sample_images)))
for i in range(len(sample_images)):
    orig = denorm(sample_images[i]).cpu().permute(1, 2, 0).numpy()
    noisy = denorm(noisy_images[i]).cpu().permute(1, 2, 0).numpy()
    recon = denorm(reconstructed_images[i]).cpu().permute(1, 2, 0).numpy()

    mse = F.mse_loss(torch.tensor(recon), torch.tensor(orig)).item()
    psnr = compute_psnr(recon, orig)
    mse_list.append(mse)
    psnr_list.append(psnr)

    plt.subplot(len(sample_images), 3, 3 * i + 1)
    plt.title(f"Original #{i+1}")
    plt.imshow(orig)
    plt.axis("off")

    plt.subplot(len(sample_images), 3, 3 * i + 2)
    plt.title(f"Noisy t={T}")
    plt.imshow(noisy)
    plt.axis("off")

    plt.subplot(len(sample_images), 3, 3 * i + 3)
    plt.title(f"MSE:{mse:.4f}\nPSNR: {psnr:.2f}dB")
    plt.imshow(recon)
    plt.axis("off")

plt.tight_layout()
plt.savefig("images/Reconstruction.jpg")

for i, (m, p) in enumerate(zip(mse_list, psnr_list)):
    print(f"Image {i+1}: MSE={m:.6f}, PSNR={p:.2f}dB")

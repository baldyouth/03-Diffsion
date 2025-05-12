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
    return (tensor.clamp(-1, 1) + 1) / 2

def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

# 模型路径
model_name = "diffusion-test/unet"
scheduler_name = "diffusion-test/scheduler"

# 加载模型和调度器
model = UNet2DModel.from_pretrained(model_name, use_safetensors=True).to("cuda")
scheduler = DDPMScheduler.from_pretrained(scheduler_name, use_safetensors=False)
scheduler.set_timesteps(1000)

sample_size = model.config.sample_size

# 加载并预处理数据
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

# 一次取4张图像
image_num = 4
batch = dataset[0:image_num]["images"]
sample_images = torch.stack(batch, dim=0).to("cuda")  # (4, 3, H, W)

# 加噪
t = 999
timesteps = torch.tensor([t], dtype=torch.long).repeat(len(sample_images)).to("cuda")
noise = torch.randn_like(sample_images)
noisy_images = scheduler.add_noise(sample_images, noise, timesteps)

# 反扩散恢复图像（逐张处理）
reconstructed_images = []
reverse_timesteps = scheduler.timesteps[scheduler.timesteps <= t]
for i in range(len(sample_images)):
    input_i = noisy_images[i:i+1]
    for timestep in reverse_timesteps:
        with torch.no_grad():
            noisy_residual = model(input_i, timestep).sample
            input_i = scheduler.step(noisy_residual, timestep, input_i).prev_sample
    reconstructed_images.append(input_i.squeeze(0).detach())

reconstructed_images = torch.stack(reconstructed_images, dim=0)

# 可视化+误差计算
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
    plt.title(f"Noisy t={t}")
    plt.imshow(noisy)
    plt.axis("off")

    plt.subplot(len(sample_images), 3, 3 * i + 3)
    plt.title(f"MSE:{mse:.4f}\nPSNR: {psnr:.2f}dB")
    plt.imshow(recon)
    plt.axis("off")

plt.tight_layout()
plt.savefig("images/Reconstruction_batch_"+str(t)+".jpg")

# 打印误差信息
for i, (m, p) in enumerate(zip(mse_list, psnr_list)):
    print(f"Image {i+1}: MSE={m:.6f}, PSNR={p:.2f}dB")

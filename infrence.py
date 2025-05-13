from diffusers import DDPMScheduler, UNet2DModel, DiffusionPipeline
from PIL import Image
import torch

def infrence(
            model_name = "./diffusion-test", 
            scheduler_name = "./diffusion-test", 
            image_nums = 4):
    scheduler = DDPMScheduler.from_pretrained(scheduler_name, use_safetensors=False )
    model = UNet2DModel.from_pretrained(model_name, use_safetensors=True).to("cuda")
    scheduler.set_timesteps(500)
    sample_size = model.config.sample_size
    
    for image_num in range(image_nums):
        noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")
        input = noise
        for t in scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = model(input, t).sample
                prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
                input = prev_noisy_sample

        image = (input / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = Image.fromarray((image * 255).round().astype("uint8"))
        image.save('./images/image_'+str(image_num+1)+'.png')

# def test02():
#     pipe = DiffusionPipeline.from_pretrained("./ddpm-cat-256")
#     image = pipe(num_inference_steps=50).images[0]
#     image.save("./images/generated_image.png")

if __name__ == '__main__':
    infrence("diffusion-test/unet", "diffusion-test/scheduler", image_nums=4)
    pass
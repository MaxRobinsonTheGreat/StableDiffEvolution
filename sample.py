import torch
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler as scheduler

import os

import util

num=1
seed=323 #random.randint(10)
prompt="nebula clouds, stars, and deep space. 8k digital astronomical photography unreal engine render. Dark and ominous, intricate fine focused detail, bright beautiful exquisite color, high color contrast"
negative_prompt="watermark, ugly, messy, disorganized, text, frame, border, margin, glitchy, incoherent, mutilated, deformed"
width=512
height=512
num_inference_steps=50
prompt_guidance=8
# model_id="runwayml/stable-diffusion-v1-5"
model_id="stabilityai/stable-diffusion-2-1"

sample_key = "images"

os.makedirs("./images", exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
pipe.enable_attention_slicing()

print(pipe.scheduler.compatibles)
# pipe.scheduler = scheduler.from_config(pipe.scheduler.config)

util.disableNSFWFilter(pipe)

latents = torch.stack(
    [torch.randn(
        (pipe.unet.in_channels, height // 8, width // 8),
        generator=torch.Generator(device='cuda').manual_seed(seed+i),
        device="cuda"
    ) for i in range(num)]
    ,0
)
prompt = [prompt]*num
negative_prompt = [negative_prompt]*num
# with autocast("cuda"):
ims = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=prompt_guidance, width=width, height=height, latents=latents).images
# ims[0].save("test.png")
grid = util.image_grid(ims, rows=1, cols=num)
    
grid.save("images/"+str(seed)+prompt[0][0:100]+".png")
grid.save("./sample.png")

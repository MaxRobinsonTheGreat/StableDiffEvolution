import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

import random
import os

import util

num=4
seed=570 #random.randint(10)
prompt="Ancient red and brown cave painting petraglyphs of hands. Beautiful minimalistic style. Exquisite intricate fine detail and color."
negative_prompt="deformed, disfigured, mutilated, messy, disorganized, weird"
width=768
height=512
num_inference_steps=50
# model_id="runwayml/stable-diffusion-v1-5"
model_id="stabilityai/stable-diffusion-2-1"

sample_key = "images"

os.makedirs("./images", exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
pipe.enable_attention_slicing()

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
ims = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, width=width, height=height, latents=latents).images
# ims[0].save("test.png")
grid = util.image_grid(ims, rows=1, cols=num)
    
grid.save("images/"+str(seed)+prompt[0][0:100]+".png")
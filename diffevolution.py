import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from transformers import CLIPTokenizer
import numpy as np
import torch
from torch import autocast
import util
from PIL import Image

prompt = "joe biden break dancing, 8k photograph" # prompt to dream about
seed = 7852
proj_name = "test"
num_ims = 4
num_inference_steps = 50
width = 512
height = 512
weights_path = "CompVis/stable-diffusion-v1-4"
device = "cuda"

num_steps = 10
step_size = 0.01
fill_in_steps = 10

torch.manual_seed(seed)
proj_path = "./evolution/"+proj_name+"_"+str(seed)
os.makedirs(proj_path, exist_ok=True)
os.makedirs(proj_path+'/selected', exist_ok=True)

print('Creating init image')

pipe = StableDiffusionPipeline.from_pretrained(
    weights_path,
    use_auth_token=True
).to(device)
util.disableNSFWFilter(pipe)

start = torch.randn((1, pipe.unet.in_channels, height // 8, width // 8), device=device)

with autocast("cuda"):
    init_img = pipe(prompt, num_inference_steps=50, latents=start, width=width, height=height)["sample"][0]

init_img.save(proj_path+'/_origin.png')

cur_latents = torch.cat([start] * num_ims)
prompt = [prompt] * num_ims

frame_index = 0
for i in range(num_steps):
    distant = torch.randn((num_ims, pipe.unet.in_channels, height // 8, width // 8), device=device)
    cur_latents = util.slerp(float(step_size), cur_latents, distant)

    with autocast("cuda"):
        images = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            latents=cur_latents,
        )["sample"]
    grid_img = util.image_grid(images, 1, num_ims)
    grid_img.save(proj_path+'/cur_pop.png')
    selection = int(input("Select 1-4: "))
    assert selection >= 1 and selection <= 4
    selected_img = images[selection-1]
    grid_img.save('{}/{}_{}.png'.format(proj_path, str(i), selection))
    selected_img.save('{}/selected/{}.png'.format(proj_path, str(i)))
    cur_latents = torch.stack([cur_latents[selection-1]]*num_ims, 0)

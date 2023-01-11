import torch
from torch import autocast
from tqdm.auto import tqdm
from util import toPil
import numpy as np
import os
from math import ceil

from diffusers import StableDiffusionUpscalePipeline

from PIL import Image

proj_name = "test"
image_path = "./outpaints/test/full.png"
prompt = "A boy holding a balloon animal, 8k uhd photograph"
negative_prompt = "ugly, gross, malformed, deformed, mutilated, extra fingers, missing fingers, glitchy"
chunk_size = 512
resize_to = 128

model_path = "stabilityai/stable-diffusion-x4-upscaler"
device = "cuda"

proj_dir = "./upscale/"+proj_name
os.makedirs(proj_dir, exist_ok=True)

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

full_img = Image.open(image_path).convert("RGB")
full_upscaled = np.array([])

full_arr = np.array(full_img)
width = full_arr.shape[1]
num_sub_ims = ceil(width / chunk_size)
print(width)
for i in range(num_sub_ims):
    start = i*chunk_size
    end = min((i+1)*chunk_size, width)
    print(start, end)
    img = full_arr[:,start:end]
    print(img.shape)
    img = toPil(img)
    img = img.resize((resize_to, resize_to))
    img.save(proj_dir+'/temp.png')
    upscaled = pipe(prompt=prompt, negative_prompt=negative_prompt, image=img).images[0]
    upscaled.save(proj_dir+'/'+str(i)+'.png')

    if i == 0:
        full_upscaled = np.array(upscaled)
    else:
        full_upscaled = np.append(full_upscaled, np.array(upscaled), 1)
    toPil(full_upscaled).save(proj_dir+'/full.png')

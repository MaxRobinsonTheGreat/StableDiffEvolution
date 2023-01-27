import torch
from util import toPil
import numpy as np
import os, json
from math import ceil

from diffusers import StableDiffusionUpscalePipeline

from PIL import Image

proj_name = "cyberpunk"
image_path = "./outpaints/cyberpunk/full.png"
prompt_file = "./outpaint_prompt.json"
prompt = "Cyberpunk Innovative futuristic Japanese city at night, hazy and foggy, octane render, professional ominous concept art, an intricate, elegant, highly detailed digital painting, concept art, smooth, sharp focus, illustration, volumetric lighting"
negative_prompt = "Deformed, disfigured, mutilated, gross, glitchy, messy, disorganized, watermark"
full_chunk_size = 512
lowres_chunk_size = 256
resize_ratio = full_chunk_size//lowres_chunk_size

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
downsize = (full_img.width//resize_ratio, full_img.height//resize_ratio)
full_img_downsized = full_img.resize(downsize)
full_upscaled = np.array([])

full_arr = np.array(full_img_downsized)
width = full_arr.shape[1]
num_sub_ims = ceil(width / lowres_chunk_size)
for i in range(num_sub_ims):
    start = i*lowres_chunk_size
    end = min((i+1)*lowres_chunk_size, width)
    img = full_arr[:,start:end]

    img = toPil(img)
    img.save(proj_dir+'/temp.png')

    while True:
        with open(prompt_file) as json_file:
            prompts = json.load(json_file)
            prompt = prompts['prompt']
            negative_prompt = prompts['negative_prompt']
        upscaled = pipe(prompt=prompt, negative_prompt=negative_prompt, image=img).images[0]
        upscaled.save(proj_dir+'/'+str(i)+'.png')

        if i == 0:
            temp_full_upscaled = np.array(upscaled)
        else:
            temp_full_upscaled = np.append(full_upscaled, np.array(upscaled), 1)
        toPil(temp_full_upscaled).save(proj_dir+'/_staged.png')
        user_in = input("Continue[enter] or reroll[r]?")
        if user_in == "":
            full_upscaled=temp_full_upscaled
            break

    toPil(full_upscaled).save(proj_dir+'/full.png')



import torch
from util import toPil
import numpy as np
import os, json
from math import ceil
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

proj_name = "cyberpunk"
image_path = "outpaints/down_test/full.png"
prompt_file = "./upscale_prompt.json"
full_chunk_size = 512
lowres_chunk_size = 128
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
horizontal = full_img.width > full_img.height
downsize = (full_img.width//resize_ratio, full_img.height//resize_ratio)
full_img_downsized = full_img.resize(downsize)
full_upscaled = np.array([])

full_arr = np.array(full_img_downsized)
big_dim = full_arr.shape[0]
if horizontal:
    big_dim = full_arr.shape[1]
num_sub_ims = ceil(big_dim / lowres_chunk_size)
for i in range(num_sub_ims):
    start = i*lowres_chunk_size
    end = min((i+1)*lowres_chunk_size, big_dim)
    if horizontal:
        img = full_arr[:,start:end]
    else: 
        img = full_arr[start:end,:]

    img = toPil(img)
    img.save(proj_dir+'/temp.png')

    while True:
        with open(prompt_file) as json_file:
            prompts = json.load(json_file)
            prompt = prompts['prompt']
            negative_prompt = prompts['negative_prompt']
        upscaled = pipe(prompt=prompt, negative_prompt=negative_prompt, image=img).images[0]
        upscaled.save(os.path.join(proj_dir, '%06d.png' % i))

        if i == 0:
            temp_full_upscaled = np.array(upscaled)
        else:
            if horizontal:
                temp_full_upscaled = np.append(full_upscaled, np.array(upscaled), 1)
            else:
                temp_full_upscaled = np.append(full_upscaled, np.array(upscaled), 0)
        toPil(temp_full_upscaled).save(proj_dir+'/_staged.png')
        user_in = input("Continue[enter] or reroll[r]?")
        if user_in == "":
            full_upscaled=temp_full_upscaled
            break

    toPil(full_upscaled).save(proj_dir+'/full.png')



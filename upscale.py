import torch
from torch import autocast
from tqdm.auto import tqdm

from diffusers import StableDiffusionUpscalePipeline

from PIL import Image

import util

device = "cuda"
model_path = "stabilityai/stable-diffusion-x4-upscaler"
prompt = "A cat"

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16
)
pipe = pipe.to(device)
# pipe.enable_attention_slicing()

init_img = Image.open('./yellow_cat_on_park_bench.png').convert("RGB")
init_img = init_img.resize((128, 128))
init_img.save('temp.png')

# with autocast("cuda"):
image = pipe(prompt=prompt, image=init_img).images[0]
image.save('./upscaled.png')
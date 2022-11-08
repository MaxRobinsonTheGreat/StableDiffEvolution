import torch
from torch import autocast
from tqdm.auto import tqdm

from diffusers import StableDiffusionImg2ImgPipeline

from PIL import Image

import util

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    use_auth_token=True
)
pipe = pipe.to(device)

prompt = ["an evolved ecosystem of microorganisms. 8k photograph, intricate detail"]*4

init_img = Image.open('./tle.png').convert("RGB")
init_img = init_img.resize((512, 512))

generator = torch.Generator(device=device).manual_seed(4321)
with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_img, strength=0.3, num_inference_steps=50, guidance_scale=7.5, generator=generator).images
image = util.image_grid(images, 2, 2)
image.save('./images/tle.png')
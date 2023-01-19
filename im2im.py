import torch

from diffusers import StableDiffusionImg2ImgPipeline

from PIL import Image

import util

device = "cuda"
model_path = "runwayml/stable-diffusion-v1-5"
# model_path = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, use_auth_token=True).to(
    device
)
pipe = pipe.to(device)

prompt = ["A top down perspective of hands holding all of diverse collective human culture and art. Landscape oil painting on canvas by Monet, insanely detailed, exquisite detail and color, perfect composition"]*4
negative_prompt = ["ugly, gross, mutilated, deformed, disfigured, messy, disorganized"]*4

init_img = Image.open('./0_0.png').convert("RGB")
init_img = init_img.resize((768, 512))

generator = torch.Generator(device=device).manual_seed(4321)
# with autocast("cuda"):
images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=[init_img]*4, strength=0.5, num_inference_steps=50, guidance_scale=7.5, generator=generator).images
image = util.image_grid(images, 2, 2)
image.save('./test.png')
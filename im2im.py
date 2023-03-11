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
pipe.enable_attention_slicing()

prompt = ["The inner neurological mind, intricate precise clear detail"]
negative_prompt = ["face, person, blurry"]

init_img = Image.open('./space_test.png').convert("RGB")
init_img = init_img.resize((1024, 1024))

generator = torch.Generator(device=device).manual_seed(3625)
# with autocast("cuda"):
images = pipe(prompt=prompt, negative_prompt=negative_prompt, image=[init_img], strength=0.7, num_inference_steps=50, guidance_scale=8, generator=generator).images
image = util.image_grid(images, 1, 1)
image.save('./test.png')
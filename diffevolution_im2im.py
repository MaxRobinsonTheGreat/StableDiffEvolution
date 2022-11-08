import torch
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image
import util
from os import makedirs
import numpy as np

init_img_path = "./evolution/purpleflower4_2568/selected/19.png"
prompt = "a beautiful evolved microorganism, 8k photograph"
proj_name = "microorg"
seed = 52874

pop_size = 4
evolution_steps = 20
select_every = 5
height = 512
width = 512

device = "cuda"
model_path = "CompVis/stable-diffusion-v1-4"
generator = torch.Generator(device=device).manual_seed(seed)
proj_path = "./evolution/"+proj_name+"_"+str(seed)
makedirs(proj_path, exist_ok=True)
makedirs(proj_path+'/selected', exist_ok=True)


if init_img_path is None:
    print('Creating init image')
    text2im = StableDiffusionPipeline.from_pretrained(
        model_path,
        use_auth_token=True
    ).to(device)
    util.disableNSFWFilter(text2im)
    with autocast("cuda"):
        init_img = text2im(prompt, num_inference_steps=50, width=width, height=height, generator=generator)["sample"][0]
    del text2im
    torch.cuda.empty_cache()
else:
    init_img = Image.open(init_img_path).convert("RGB")
    init_img = init_img.resize((width, height))

init_img.save(proj_path+'/_origin.png')

prompt = [prompt]*pop_size

im2im = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path,
    use_auth_token=True
).to(device)
util.disableNSFWFilter(im2im)

cur = init_img
for i in range(evolution_steps):
    with autocast("cuda"):
        images = im2im(prompt=prompt, init_image=cur, strength=0.7, num_inference_steps=100, generator=generator).images
    image = util.image_grid(images, 1, pop_size)
    image.save(proj_path+'/cur_pop.png')
    selection = int(input("Select 1-4: "))
    assert selection >= 1 and selection <= 4
    cur = images[selection-1]
    image.save('{}/{}_{}.png'.format(proj_path, str(i), selection))
    cur.save('{}/selected/{}.png'.format(proj_path, str(i)))


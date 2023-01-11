import torch
from PIL import Image
from matplotlib import cm
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import os

project_name = "test"
prompt = "A balloon animal"
neg_prompt = 'ugly, deformed, gross, mutilated, extra limbs, extra fingers'
start_image = 'kid.jpg'
num_outpaints = 2
height = 512
window_size = 512
slide_size = 512 // 2

proj_dir = "./outpaints/"+project_name
os.makedirs(proj_dir, exist_ok=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

mask = np.ones((height, window_size))
mask[:,0:slide_size] = 0
print(mask.shape)
mask_image = Image.fromarray(np.uint8(mask*255)).convert('RGB')
mask_image.save(proj_dir+'/mask.png')

cur_image = Image.open(start_image)
cur_image = cur_image.resize((height, window_size))
full_image = cur_image.copy()

for i in range(num_outpaints):
    im = np.array(cur_image)
    start = im.shape[1]-slide_size
    next_im = np.zeros((height, window_size, 3))
    next_im[:,0:slide_size] = im[:,start:]
    print(next_im.shape)
    next_im = Image.fromarray(np.uint8(next_im)).convert('RGB')
    next_im.save(proj_dir+'/'+str(i)+'.png')
    cur_image = pipe(prompt=prompt, image=next_im, mask_image=mask_image).images[0]
    cur_image.save(proj_dir+'/'+str(i)+'.png')
    
    im = np.array(cur_image)
    print(im[:,slide_size:].shape)
    full_image = np.append(full_image, im[:,slide_size:], 1)
    full_image = Image.fromarray(np.uint8(full_image)).convert('RGB')
    full_image.save(proj_dir+'/full.png')

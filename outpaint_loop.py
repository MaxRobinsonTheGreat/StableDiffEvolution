import torch, cv2, os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import util

image_path = 'outpaints/ai_museum_right/full.png'
prompt = "a set of paintings"
negative_prompt = "watermark, disorganized, messy, border, edge, deformed, ugly, mutilated, gross"
image_size = 512
edge_size = 128
num_choices = 4
model = "stabilityai/stable-diffusion-2-inpainting"
# model = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

util.disableNSFWFilter(pipe)

mask = np.zeros((image_size, image_size))
mask[:,edge_size:-edge_size] = 1
mask_image = Image.fromarray(np.uint8(mask*255)).convert('RGB')

start_image = np.array(Image.open(image_path))
in_image = np.zeros((image_size, image_size, 3))
in_image[:,0:edge_size,:] = start_image[:,-edge_size:,:]
in_image[:,-edge_size:,:] = start_image[:,0:edge_size,:]
in_image = util.toPil(in_image)

selected_image = None

while True:
    out_images = pipe(
        prompt=[prompt]*num_choices, 
        negative_prompt=[negative_prompt]*num_choices, 
        image=in_image, 
        mask_image=mask_image
    ).images
    util.image_grid(out_images, 1, num_choices).save('outpaint_loop_choices.png')

    choice = input('Choose [1-{}] or reroll[enter]:'.format(str(num_choices)))
    if choice.isdigit():
        choice = int(choice)
        choice-=1
        if choice>=0 and choice<num_choices:
            selected_image=out_images[choice]
            break
    
out_image = np.array(selected_image)
final = np.hstack((out_image[:,edge_size:-edge_size,:], start_image))
util.toPil(final).save('outpaint_loop.png')

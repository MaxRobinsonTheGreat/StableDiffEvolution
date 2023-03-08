import torch, cv2, os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import util

left_image_path = 'outpaints/human-art2/full.png'
right_image_path = 'outpaints/human-art/faces.png'

prompt = "an empty wall"
negative_prompt = "watermark, disorganized, messy, border, edge"
image_size = 512
edge_size = 200
num_choices = 4
# model = "stabilityai/stable-diffusion-2-inpainting"
model = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

util.disableNSFWFilter(pipe)

mask = np.zeros((image_size, image_size))
mask[:,edge_size:-edge_size] = 1
mask_image = Image.fromarray(np.uint8(mask*255)).convert('RGB')

left_image = np.array(Image.open(left_image_path))
right_image = np.array(Image.open(right_image_path))
in_image = np.zeros((image_size, image_size, 3))
in_image[:,0:edge_size,:] = left_image[:,-edge_size:,:]
in_image[:,-edge_size:,:] = right_image[:,0:edge_size,:]
in_image = util.toPil(in_image)

selected_image = None

while True:
    out_images = pipe(
        prompt=[prompt]*num_choices, 
        negative_prompt=[negative_prompt]*num_choices, 
        image=in_image, 
        mask_image=mask_image
    ).images
    util.image_grid(out_images, 1, num_choices).save('outpaint_combine_choices.png')

    choice = input('Choose [1-{}] or reroll[enter]:'.format(str(num_choices)))
    if choice.isdigit():
        choice = int(choice)
        choice-=1
        if choice>=0 and choice<num_choices:
            selected_image=out_images[choice]
            break
    
out_image = np.array(selected_image)
final = np.hstack((left_image, out_image[:,edge_size:-edge_size,:], right_image))
util.toPil(final).save('outpaint_combine.png')

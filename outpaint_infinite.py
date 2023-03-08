import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
import os, json, util

project_name = "ai_museum_right_start"
start_image = 'outpaints/ai_museum_right_2/ai_museum_right_start.png'
height = 512
window_size = 512
slide_size = 128
num_options = 4
right = False
model = "stabilityai/stable-diffusion-2-inpainting"
# model = "runwayml/stable-diffusion-inpainting"

used_prompts = []
prompt_file = "./outpaint_prompt.json"
proj_dir = "./outpaints/"+project_name
os.makedirs(proj_dir, exist_ok=True)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model,
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

util.disableNSFWFilter(pipe)

mask = np.ones((height, window_size))
if right:
    mask[:,0:slide_size] = 0
else:
    mask[:,-slide_size:] = 0

mask_image = Image.fromarray(np.uint8(mask*255)).convert('RGB')
mask_image.save(proj_dir+'/mask.png')

cur_image = Image.open(start_image)
resize_ratio = cur_image.height/height
resize_width = int(cur_image.width//resize_ratio)
cur_image = cur_image.resize((resize_width, height))
full_image = cur_image.copy()
print(resize_width-window_size)
if right:
    cur_image = np.array(cur_image)[:,resize_width-window_size:]
else:
    cur_image = np.array(cur_image)[:,:window_size]
print(cur_image.shape)
cur_image = util.toPil(cur_image)
i=0
while True:
    im = np.array(cur_image)
    start = im.shape[1]-slide_size
    next_im = np.zeros((height, window_size, 3))
    if right:
        next_im[:,0:slide_size] = im[:,start:]
    else:
        next_im[:,-slide_size:] = im[:,0:slide_size]
    next_im = Image.fromarray(np.uint8(next_im)).convert('RGB')
    next_im.save(proj_dir+'/'+str(i)+'.png')
    
    cur_image=None
    while cur_image is None:
        with open(prompt_file) as json_file:
            prompts = json.load(json_file)
            prompt = prompts['prompt']
            negative_prompt = prompts['negative_prompt']
        ims = pipe(
            prompt=[prompt]*num_options, 
            negative_prompt=[negative_prompt]*num_options, 
            image=next_im, 
            mask_image=mask_image
        ).images
        im_grid = util.image_grid(ims, 1, num_options)
        im_grid.save(proj_dir+'/_choice.png')
        choice = input("Frame "+str(i)+" finished. Choose [1-4] or reroll[enter]: ")
        if choice == "q":
            break
        if choice.isdigit():
            choice = int(choice)
            choice-=1
            if choice>=0 and choice<num_options:
                cur_image=ims[choice]
                used_prompts.append({
                    'prompt': prompt,
                    'neg_prompt': negative_prompt
                })
                with open(proj_dir+'/used_prompts.json', 'w') as fp:
                    json.dump(used_prompts, fp)

    if cur_image is None:
        break

    cur_image.save(proj_dir+'/'+str(i)+'.png')
    
    im = np.array(cur_image)
    if right:
        full_image = np.append(full_image, im[:,slide_size:], 1)
    else:
        full_image = np.append(im[:,:-slide_size], full_image, 1)
    full_image = Image.fromarray(np.uint8(full_image)).convert('RGB')
    full_image.save(proj_dir+'/full.png')

    i+=1



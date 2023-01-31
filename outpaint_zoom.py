import torch, cv2, os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from util import toPil, cropToCenter

project_name = "space3"
prompt = "nebula, stars, and deep space. 8k digital astronomical render. Dark and ominous yet beautiful, intricate fine focused detail, bright beautiful exquisite color, high color contrast"
negative_prompt = "watermark, text, frame, wall, room, poster, edge, picture, website, software, boundary"
zoom_speed = 64
num_outpaints = 75
num_filler_frames = 32
start_image = './sample.png'
# model = "stabilityai/stable-diffusion-2-inpainting"
model = "runwayml/stable-diffusion-inpainting"


image_size = (512, 512)
proj_dir = "./zooms/"+project_name
frames_dir = proj_dir+"/frames"
os.makedirs(frames_dir, exist_ok=True)
downsize = (512-(2*zoom_speed),512-(2*zoom_speed))

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

mask_image = np.ones(image_size)*255
mask_image[zoom_speed:-zoom_speed,zoom_speed:-zoom_speed] = 0
mask_image = toPil(mask_image)
mask_image.save(proj_dir+'/mask.png')

cur_image = Image.open(start_image)
cur_image = cur_image.resize(image_size)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model,
    revision="fp16",
    torch_dtype=torch.float16,
).to('cuda')

frame_count = 0
for outpaints in range(num_outpaints):
    cur_image.save(os.path.join(proj_dir, 'in.png'))

    downsized_image = cur_image.resize(downsize)
    in_image = add_margin(downsized_image, zoom_speed, zoom_speed, zoom_speed, zoom_speed, (0,0,0))
    in_image.save(proj_dir+'/zoomed.png')

    # torch.manual_seed(seed)
    while True:
        print("Outpaint", outpaints)
        out_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=in_image, mask_image=mask_image).images[0]
        out_image.paste(downsized_image, (zoom_speed, zoom_speed))
        out_image.save(proj_dir+'/choice.png')
        if input("Continue[enter] or reroll[r]?") == "":
            break
    out_image.save(os.path.join(proj_dir, 'out.png'))



    if num_filler_frames > 0:
        size_step = 2 * zoom_speed / (num_filler_frames)

        for filler_count in reversed(range(num_filler_frames)): #+1 to include the first frame
            step = round((filler_count * size_step))
            inner_size = (image_size[0]-step, image_size[1]-step)

            if inner_size != image_size:
                filler_frame = cropToCenter(out_image, inner_size)
                filler_frame = filler_frame.resize(image_size)
            else:
                filler_frame = out_image

            if filler_count == 0:
                filler_frame.save(os.path.join(proj_dir, 'end.png'))
            if filler_count == num_filler_frames:
                filler_frame.save(os.path.join(proj_dir, 'start.png'))


            filler_frame.save(os.path.join(frames_dir, '%06d.png' % frame_count))
            frame_count += 1
    else:
        out_image.save(os.path.join(frames_dir, '%06d.png' % frame_count))
        frame_count += 1
    cur_image = out_image
    
import torch, cv2, os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from util import toPil, cropToCenter, image_grid, disableNSFWFilter

project_name = "neurons"
prompt = "Many flashing biological neurons on a black background"
negative_prompt = "watermark, text, frame, border, edge"
zoom_speed = 64
num_outpaints = 10
num_filler_frames = 32
start_image = './sample.png'
choices = 4
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
disableNSFWFilter(pipe)

frame_count = 0
for outpaints in range(num_outpaints):
    cur_image.save(os.path.join(proj_dir, 'in.png'))

    downsized_image = cur_image.resize(downsize)
    in_image = add_margin(downsized_image, zoom_speed, zoom_speed, zoom_speed, zoom_speed, (0,0,0))
    in_image.save(proj_dir+'/zoomed.png')

    while True:
        print("Outpaint", outpaints)
        out_images = pipe(prompt=[prompt]*choices, negative_prompt=[negative_prompt]*choices, image=[in_image]*choices, mask_image=mask_image).images

        for im in out_images:
            im.paste(downsized_image, (zoom_speed, zoom_speed))

        grid = image_grid(out_images, 1, choices)
        grid.save(proj_dir+'/choice.png')

        if choices == 1:
            out_image = out_images[0]
            if input("Continue[enter] or reroll[r]?") == "":
                break
        else:
            choice = input("Choose [1-{}] or reroll[enter]".format(str(choices)))
            if choice.isdigit():
                choice = int(choice)
                choice-=1
                if choice>=0 and choice<choices:
                    out_image=out_images[choice]
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
    
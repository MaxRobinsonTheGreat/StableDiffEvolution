import os, glob, torch, json
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline

project_name = "atoms"
orig_size = 512
downscale_size = 128
zoom_speed = 128
model_path = "stabilityai/stable-diffusion-x4-upscaler"
prompt_file = "./upscale_prompt.json"

upscale_factor = 4
downscale_factor = orig_size / downscale_size
zoom_speed_down = round(zoom_speed / downscale_factor)
zoom_speed_up = round(zoom_speed_down * upscale_factor)
inner_size_down = downscale_size - (2*zoom_speed_down)
inner_size_up = inner_size_down * upscale_factor

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
pipe.enable_attention_slicing()


proj_dir = "./zooms/"+project_name
frames_dir = proj_dir+"/frames"
upscaled_dir = proj_dir+"/upscaled_frames"
os.makedirs(upscaled_dir, exist_ok=True)
im_files = sorted(filter(os.path.isfile, glob.glob(frames_dir + '/*.png')))


prev_im = None
for i, file in enumerate(im_files):
    im = Image.open(file)
    im_down = im.resize((downscale_size,downscale_size))

    # if prev_im is not None:
        # print('pasted to downscaled')
        # prev_im_down = prev_im_down.resize((downscale_size, downscale_size))
        # prev_im_down = prev_im.resize((inner_size_down,inner_size_down))
        # im_down.paste(prev_im_down, (zoom_speed_down, zoom_speed_down))

    im_down.save(os.path.join(upscaled_dir, 'downscaled.png'))

    while True:
        with open(prompt_file) as json_file:
            prompts = json.load(json_file)
            prompt = prompts['prompt']
            negative_prompt = prompts['negative_prompt']
        upscaled = pipe(prompt=prompt, negative_prompt=negative_prompt, image=im_down).images[0]
        
        if prev_im is not None:
            print('pasted to upscaled', inner_size_up)
            prev_im.save(os.path.join(upscaled_dir, 'prev.png'))
            prev_im_up_inner = prev_im.resize((inner_size_up,inner_size_up))
            prev_im_up_inner.save(os.path.join(upscaled_dir, 'prev_inner.png'))

            print(zoom_speed_up)
            upscaled.save(os.path.join(upscaled_dir, 'upscaled_before.png'))
            upscaled = upscaled.copy()
            upscaled.paste(prev_im_up_inner, (zoom_speed_up, zoom_speed_up))
            upscaled.save(os.path.join(upscaled_dir, 'upscaled_after.png'))


        upscaled.save(os.path.join(upscaled_dir, '%06d.png' % i))

        user_in = input("Continue[enter] or reroll[r]?")
        if user_in == "":
            prev_im=upscaled
            break

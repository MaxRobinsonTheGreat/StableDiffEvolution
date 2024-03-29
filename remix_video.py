import torch, cv2, os, json

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from util import disableNSFWFilter
import random

proj_name = 'superman'
video_file = './spiderman_orig.gif'
model = "runwayml/stable-diffusion-v1-5"
frame_step = 1
image_size = (512, 512)

prompt_file = './remix_prompts.json'
device = "cuda"
proj_dir = './remixes/'+proj_name
os.makedirs(proj_dir, exist_ok=True)

cap = cv2.VideoCapture(video_file)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=torch.float16).to(
    device
)
print(pipe.scheduler.compatibles, pipe.scheduler)
# pipe.scheduler = scheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)
pipe.enable_attention_slicing()
disableNSFWFilter(pipe)

config = None
with open(prompt_file) as json_file:
    config = json.load(json_file)

config_json = json.dumps(config, indent=4)
# save config to proj dir
with open(proj_dir+"/prompts.json", "w") as outfile:
    outfile.write(config_json)

prompt_count=0
read_count=0
im_count=0
cur_config = config[prompt_count]
prompt_frame_count=0
while(cap.isOpened()):
    ret,cv2_im = cap.read()
    if ret and read_count % frame_step == 0:

        converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)

        init_img = Image.fromarray(converted)
        init_img = init_img.resize(image_size)

        prompt = cur_config['prompt']
        negative_prompt = cur_config['neg_prompt']
        strength = cur_config['strength']
        seed = random.randint(0, 1000)
        if 'seed' in cur_config:
            seed = cur_config['seed']

        generator = torch.Generator(device=device).manual_seed(seed)

        print('frame', str(im_count))
        remixed_image = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_img, strength=strength, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
        remixed_image.save('./test.png')
                
        remixed_image.save(os.path.join(proj_dir, '%06d.png' % im_count))
        im_count+=1
        prompt_frame_count+=1

        num_frames = cur_config['num_frames']
        if prompt_frame_count>=num_frames:
            prompt_count+=1
            if prompt_count < len(config):
                cur_config = config[prompt_count]
                prompt_frame_count=0
    elif not ret:
        break
    read_count += 1
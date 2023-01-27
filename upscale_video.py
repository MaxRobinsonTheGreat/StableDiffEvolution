import torch, cv2, os

from diffusers import StableDiffusionUpscalePipeline

from PIL import Image

proj_name = "liminal"
video_path = "./liminal.mp4"
prompt = "dimly lit dingy office hallways, creepy and ominous, nightmarish, horror photography"
negative_prompt = "ugly, deformed, gross, mutilated, messy, disorganized, frame, border, text, watermark, signature"
frame_size = (512, 256)
lowres_size = (128, 128)
seed = 1

model_path = "stabilityai/stable-diffusion-x4-upscaler"
device = "cuda"

proj_dir = "./upscale/"+proj_name
os.makedirs(proj_dir, exist_ok=True)

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_path,
    revision="fp16", 
    torch_dtype=torch.float16
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

cap = cv2.VideoCapture(video_path) # says we capture an image from a webcam

frame_count = 0
while(cap.isOpened()):
    ret,cv2_im = cap.read()
    converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    init_img = Image.fromarray(converted)
    init_img = init_img.resize(lowres_size)

    torch.manual_seed(seed)
    upscaled_img = pipe(prompt=prompt, negative_prompt=negative_prompt, image=init_img).images[0]
    upscaled_img.save(os.path.join(proj_dir, '%06d.png' % frame_count))
    frame_count += 1


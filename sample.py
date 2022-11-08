import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPTokenizer

import os

import util

num=4
seed=234#654
prompt = "a yellow slime mold colony spreading and growing on a flat black empty surface, top-down view, 8k uhd photograph"

os.makedirs("./images", exist_ok=True)

# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
torch.manual_seed(seed)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    tokenizer=tokenizer,
    scheduler=lms,
    use_auth_token=True
).to("cuda")

util.disableNSFWFilter(pipe)

prompt = [prompt]*num
with autocast("cuda"):
    samples = pipe(prompt, num_inference_steps=100, width=512, height=512)["sample"]
    grid = util.image_grid(samples, rows=1, cols=num)
    
grid.save("images/"+str(seed)+prompt[0][0:100]+".png")
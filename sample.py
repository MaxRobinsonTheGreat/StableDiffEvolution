from tkinter import W
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from transformers import CLIPTokenizer

from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    tokenizer=tokenizer,
    scheduler=lms,
    use_auth_token=True
).to("cuda")

prompt = ["a sentient artificial intelligence"]*4
with autocast("cuda"):
    samples = pipe(prompt, num_inference_steps=75, width=512, height=512)["sample"]
    grid = image_grid(samples, rows=2, cols=2)
    
grid.save(prompt[0][0:100]+".png")
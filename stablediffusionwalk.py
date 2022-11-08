"""
Basic stable diffusion evolution

Code adapted from Andrej Karpathy https://gist.github.com/karpathy/00103b0037c5aaea32fe1da1af553355

after running this code run `python makevid.py FOLDER_PATH` in the folder where all the png are saved
"""

import os
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler
from transformers import CLIPTokenizer
import numpy as np
import torch
from torch import autocast
import util

def run(
        # --------------------------------------
        # args you probably want to change
        prompt = "a top-down view of a yellow slime mold spreading and growing on a flat black empty surface, 8k uhd photograph", # prompt to dream about
        num_ims = 1,
        gpu = 0, # id of the gpu to run on
        name = 'slimemold2', # name of this project, for the output directory
        rootdir = './walks',
        seeds = [325, 785, 621, 325],
        num_steps = 400, # number of steps between each pair of sampled points
        num_inference_steps = 100, # more (e.g. 100, 200 etc) can create slightly better images
        guidance_scale = 7.5, # can depend on the prompt. usually somewhere between 3-10 is good
        # --------------------------------------
        # args you probably don't want to change
        width = 512,
        height = 768,
        weights_path = "CompVis/stable-diffusion-v1-4",
        # --------------------------------------
    ):
    assert torch.cuda.is_available()
    assert height % 8 == 0 and width % 8 == 0
    # torch.manual_seed(seed)
    torch_device = f"cuda:{gpu}"
    assert len(seeds) > 1


    # init the output dir
    outdir = os.path.join(rootdir, name)
    os.makedirs(outdir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # init all of the models and move them to a given GPU
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
    pipe = StableDiffusionPipeline.from_pretrained(weights_path, tokenizer=tokenizer, scheduler=lms, use_auth_token=True).to(torch_device)
    util.disableNSFWFilter(pipe)

    start = torch.randn(
        (num_ims, pipe.unet.in_channels, height // 8, width // 8),
        generator=torch.Generator(device='cuda').manual_seed(seeds[0]),
        device=torch_device
    )

    # iterate the loop
    prompt = [prompt]*num_ims
    frame_index = 0
    for seed in seeds[1:]:
        end = torch.randn(
            (num_ims, pipe.unet.in_channels, height // 8, width // 8),
            generator=torch.Generator(device='cuda').manual_seed(seed),
            device=torch_device
        )
        for i, t in enumerate(np.linspace(0, 1, num_steps, endpoint=False)):
            init = util.slerp(float(t), start, end)

            print("dreaming... ", frame_index)
            with autocast("cuda"):
                images = pipe(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    latents=init,
                    guidance_scale=guidance_scale,
                    width=width, 
                    height=height
                )["sample"]
            grid_image = util.image_grid(images, 1, 1)
            outpath = os.path.join(outdir, 'frame%06d.png' % frame_index)
            grid_image.save(outpath)
            frame_index += 1

        start = end


run()

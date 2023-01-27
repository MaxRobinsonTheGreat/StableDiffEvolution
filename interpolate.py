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
        prompt = "the landscape of an alien world made of mushrooms and fungus, 4k digital art illustration, science fiction concept art, extremely beautiful, dark and ominous, creepy, exquisite color and detail", # prompt to dream about
        negative_prompt="glitchy, ugly, disorganized, messy, watermark",
        num_ims = 1,
        gpu = 0, # id of the gpu to run on
        name = 'fungus_world', # name of this project, for the output directory
        rootdir = './walks',
        seeds = [325, 785],
        num_steps = 100, # number of steps between each pair of sampled points
        num_inference_steps = 50, # more (e.g. 100, 200 etc) can create slightly better images
        guidance_scale = 7.5, # can depend on the prompt. usually somewhere between 3-10 is good
        width = 768,
        height = 512,
        weights_path = "stabilityai/stable-diffusion-2-1",
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

    pipe = StableDiffusionPipeline.from_pretrained(weights_path).to(torch_device)
    pipe.enable_attention_slicing()
    util.disableNSFWFilter(pipe)

    start = torch.randn(
        (num_ims, pipe.unet.in_channels, height // 8, width // 8),
        generator=torch.Generator(device='cuda').manual_seed(seeds[0]),
        device=torch_device
    )

    # iterate the loop
    prompt = [prompt]*num_ims
    negative_prompt = [negative_prompt]*num_ims
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
            # with autocast("cuda"):
            images = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                latents=init,
                guidance_scale=guidance_scale,
                width=width, 
                height=height
            )["images"]
            grid_image = util.image_grid(images, 1, 1)
            outpath = os.path.join(outdir, 'frame%06d.png' % frame_index)
            grid_image.save(outpath)
            frame_index += 1

        start = end


run()

"""
Interpolation between different latent vals AND prompts. copied from https://gist.github.com/nateraw/c989468b74c616ebbc6474aa8cdd9e53
"""

import os
import inspect
import fire
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from time import time
from PIL import Image
from einops import rearrange
import numpy as np
import torch
from torch import autocast
from torchvision.utils import make_grid
import util

# -----------------------------------------------------------------------------


@torch.no_grad()
def diffuse(
        pipe,
        cond_embeddings, # text conditioning, should be (1, 77, 768)
        negative_prompt,
        cond_latents,    # image conditioning, should be (1, 4, 64, 64)
        num_inference_steps,
        guidance_scale,
        eta,
    ):
    torch_device = cond_latents.get_device()

    # classifier guidance: add the unconditional embedding
    max_length = cond_embeddings.shape[1] # 77
    uncond_input = pipe.tokenizer([negative_prompt], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]

    # init the scheduler
    accepts_offset = "offset" in set(inspect.signature(pipe.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # diffuse!
    for i, t in enumerate(pipe.scheduler.timesteps):

        # expand the latents for classifier free guidance
        # TODO: gross much???
        latent_model_input = torch.cat([cond_latents] * 2)
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            sigma = pipe.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        # cfg
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        # TODO: omfg...
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            cond_latents = pipe.scheduler.step(noise_pred, i, cond_latents, **extra_step_kwargs)["prev_sample"]
        else:
            cond_latents = pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]

    # scale and decode the image latents with vae
    cond_latents = 1 / 0.18215 * cond_latents
    image = pipe.vae.decode(cond_latents)["sample"]

    # generate output numpy image as uint8
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)

    return image

def main(
        # --------------------------------------
        # args you probably want to change
        prompts = [
            "A vast ocean with distant islands and ships. Oil painting on canvas. extremely beautiful, dark and ominous, exquisite detail and form, high color contrast, ethereal",
            "An ornate medieval castle on a mountain range with sheer cliffsides and overhangs. Oil painting on canvas. Extremely beautiful with intricate detail, dark and ominous, exquisite detail and form, high color contrast, ethereal"
        ], # prompts to dream about
        seeds=[634, 587],
        negative_prompt="Deformed, disfigured, ugly, mutilated, gross, glitchy, frame, borders",
        gpu = 0, # id of the gpu to run on
        name = 'ocean-castle', # name of this project, for the output directory
        rootdir = './dreams',
        num_steps = 100,  # number of steps between each pair of sampled points
        # --------------------------------------
        # args you probably don't want to change
        num_inference_steps = 50,
        guidance_scale = 7.5,
        eta = 0.0,
        width = 1536,
        height = 1024,
        # --------------------------------------
):
    assert len(prompts) == len(seeds)
    assert torch.cuda.is_available()
    assert height % 8 == 0 and width % 8 == 0

    # init the output dir
    outdir = os.path.join(rootdir, name)
    os.makedirs(outdir, exist_ok=True)

    # # init all of the models and move them to a given GPU
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
    pipe.enable_attention_slicing()
    torch_device = f"cuda:{gpu}"
    pipe.unet.to(torch_device)
    pipe.vae.to(torch_device)
    pipe.text_encoder.to(torch_device)

    # negative prompt embeddings
    uncond_input = pipe.tokenizer(
        [negative_prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(
        uncond_input.input_ids.to("cuda"),
    )[0].detach()

    # get the conditional text embeddings based on the prompts
    prompt_embeddings = []

    for prompt in prompts:   
        text_input = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            embed = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, embed])
        prompt_embeddings.append(embed)
    
    # Take first embed and set it as starting point, leaving rest as list we'll loop over.
    prompt_embedding_a, *prompt_embeddings = prompt_embeddings

    # Take first seed and use it to generate init noise
    init_seed, *seeds = seeds
    init_a = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        device=torch_device,
        generator=torch.Generator(device='cuda').manual_seed(init_seed)
    )
    
    frame_index = 0
    for p, prompt_embedding_b in enumerate(prompt_embeddings):

        init_b = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator=torch.Generator(device='cuda').manual_seed(seeds[p]),
            device=torch_device
        )

        for i, t in enumerate(np.linspace(0, 1, num_steps)):
            print("dreaming... ", frame_index)

            cond_embedding = util.slerp(float(t), prompt_embedding_a, prompt_embedding_b)
            init = util.slerp(float(t), init_a, init_b)

            with autocast("cuda"):
                image = diffuse(pipe, cond_embedding, negative_prompt, init, num_inference_steps, guidance_scale, eta)

            im = Image.fromarray(image)
            outpath = os.path.join(outdir, 'frame%06d.jpg' % frame_index)
            im.save(outpath)
            frame_index += 1

        prompt_embedding_a = prompt_embedding_b
        init_a = init_b


if __name__ == '__main__':
    fire.Fire(main)
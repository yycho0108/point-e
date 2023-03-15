#!/usr/bin/env python3

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch


def text_to_image(text: str, num: int, device: str = 'cuda:0',
                  **kwds):
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler,
        # torch_dtype=torch.float16
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    # texts = [text] * num
    # image = pipe(texts, **kwds).images
    image = pipe(text, num_images_per_prompt=num,
                 **kwds).images
    return image

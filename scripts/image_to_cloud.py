#!/usr/bin/env python3

from typing import Optional, Dict, Any

import torch as th
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config


def load_model(device):
    print('creating base model...')
    base_name = 'base1B'  # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    return (
        base_model, upsampler_model,
        base_diffusion, upsampler_diffusion
    )


def load_sampler(device):
    # load models
    models = load_model(device)
    (base_model, upsampler_model,
     base_diffusion, upsampler_diffusion) = models

    # load sampler
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
        clip_denoised=False
    )
    return sampler


def image_to_cloud(img: th.Tensor, device,
                   cache: Optional[Dict[str, Any]] = None):
    # Try to lookup sampler from cache; otherwise create sampler.
    sampler = None
    if cache is not None:
        if 'sampler' not in cache:
            cache['sampler'] = load_sampler(device)
        sampler = cache['sampler']
    else:
        sampler = load_sampler(device)

    samples = None
    for x in tqdm(sampler.sample_batch_progressive(
            batch_size=1,
            model_kwargs=dict(images=[img]))):
        samples = x
    cloud = sampler.output_to_point_clouds(samples)[0]
    return cloud

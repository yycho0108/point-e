#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass
import gc

import torch as th
import torch.nn as nn

import k_diffusion as K  # ??
from diffusion import get_model
import clip


@dataclass
class TextToImageConfig:
    prompt: str = ''
    weight: float = 5.0
    n_images: int = 4
    steps: int = 50
    seed: int = 0
    display_every: int = 10


class CFGDenoiser(nn.Module):
    def __init__(self, model: nn.Module, scale: float):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, x: th.Tensor, sigma, clip_embed):
        x_in = th.cat([x] * 2)
        sigma_in = th.cat([sigma] * 2)
        clip_embed_in = th.cat([th.zeros_like(clip_embed), clip_embed])
        uncond, cond = self.model(
            x_in, sigma_in, clip_embed=clip_embed_in).chunk(2)
        return uncond + (cond - uncond) * self.scale


def load_models(model_path: str = 'cc12m_1_cfg.pth',
                device=None):
    inner_model = get_model('cc12m_1_cfg')()
    _, side_y, side_x = inner_model.shape
    inner_model.load_state_dict(
        th.load(model_path,
                map_location='cpu'))
    inner_model = inner_model.half().eval().requires_grad_(False)
    model = K.external.VDenoiser(inner_model)
    clip_model = clip.load(
        inner_model.clip_model,
        jit=False,
        device='cpu')[0]

    if device is not None:
        model.to(device)
        clip_model.to(device)

    return model, clip_model


def text_to_embed(clip_model, prompt: str, device):
    return clip_model.encode_text(
        clip.tokenize(prompt).to(device)
    ).float()  # .cuda()


def text_to_image(text: str,
                  num: int,
                  dim: Tuple[int, int] = (256, 256),
                  steps: int = 50,
                  device: str = 'cuda:0',
                  weight: float = 5.0
                  ):
    model, clip_model = load_models(device=device)
    gc.collect()
    th.cuda.empty_cache()

    embed = text_to_embed(clip_model, text, device=device)
    sigmas = K.sampling.get_sigmas_karras(steps,
                                          1e-2, 160, device=device)
    shape = (num, 3) + dim
    x = th.randn(shape,
                 device=device) * sigmas[0]
    model = CFGDenoiser(th.cuda.amp.autocast()(model), weight)
    extra_args = {'clip_embed': embed.repeat([num, 1])}
    outs = K.sampling.sample_lms(model, x, sigmas,
                                 extra_args=extra_args)
    outs = 0.5 * (outs.clamp(-1, +1) + 1)

    return outs

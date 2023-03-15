#!/usr/bin/env python3

from typing import Tuple, List, Optional

import torch as th
from tqdm.auto import trange
from functools import partial

import k_diffusion as K
import clip
from diffusion import get_model, get_models, sampling, utils


class Sampler:
    def __init__(self,
                 steps: int = 50,
                 weight: float = 5.0,
                 device=None,
                 ):
        self.device = device

        self.model, self.clip_model = self.load_models()

        self.steps = steps

        weights = [weight]
        self.weights = th.tensor([1 - sum(weights), *weights],
                                 device=device)
        self.device = device

    def load_models(self, model_path: str = 'cc12m_1_cfg.pth'):
        device = self.device
        inner_model = get_model('cc12m_1_cfg')()
        _, side_y, side_x = inner_model.shape
        inner_model.load_state_dict(
            th.load(model_path,
                    map_location='cpu'))
        inner_model = inner_model.half().eval().requires_grad_(False)
        model = inner_model
        # model = K.external.VDenoiser(inner_model)

        clip_model_name = (
            model.clip_model if hasattr(model, 'clip_model') else 'ViT-B/16'
        )
        clip_model = clip.load(
            clip_model_name,
            jit=False,
            device='cpu')[0]
        clip_model.eval().requires_grad_(False)

        if device is not None:
            model.to(device)
            clip_model.to(device)

        return model, clip_model

    def cfg_model_fn(self, embeds, x, t):
        n = x.shape[0]
        n_conds = len(embeds)
        x_in = x.repeat([n_conds, 1, 1, 1])
        t_in = t.repeat([n_conds])
        clip_embed_in = th.cat([*embeds]).repeat_interleave(n, 0)
        vs = self.model(
            x_in, t_in, clip_embed_in).view(
            [n_conds, n, *x.shape[1:]])
        v = vs.mul(self.weights[:, None, None, None, None]).sum(0)
        return v

    def run(self, model_fn, x: th.Tensor, steps: int,
            method: str, eta: Optional[float] = None):
        if method == 'ddpm':
            return sampling.sample(model_fn, x, steps, 1., {})
        if method == 'ddim':
            return sampling.sample(model_fn, x, steps, eta, {})
        if method == 'prk':
            return sampling.prk_sample(model_fn, x, steps, {})
        if method == 'plms':
            return sampling.plms_sample(model_fn, x, steps, {})
        if method == 'pie':
            return sampling.pie_sample(model_fn, x, steps, {})
        if method == 'plms2':
            return sampling.plms2_sample(model_fn, x, steps, {})
        if method == 'iplms':
            return sampling.iplms_sample(model_fn, x, steps, {})
        assert False

    def run_all(self, prompt: str,
                n: int,
                batch_size: int,
                dim: Tuple[int, int] = (256, 256)) -> List[str]:
        device = self.device
        shape = (n, 3) + dim
        x = th.randn(shape, device=device)
        t = th.linspace(1, 0, self.steps + 1, device=device)[:-1]
        steps = utils.get_spliced_ddpm_cosine_schedule(t)

        zero_embed = th.zeros([1, self.clip_model.visual.output_dim],
                              device=device)
        embeds = [zero_embed, self.clip_model.encode_text(
            clip.tokenize(prompt).to(device)).float().to(device)]

        model_fn = partial(self.cfg_model_fn, embeds)

        outs = []
        for i in trange(0, n, batch_size):
            cur_batch_size = min(n - i, batch_size)
            outs.extend(self.run(model_fn,
                                 x[i:i + cur_batch_size], steps,
                                 method='plms'
                                 ))
        return outs

        # filenames = []
        # for j, out in enumerate(outs):
        #     filename: str = F'{out_dir}/{j:05.png'
        #     utils.to_pil_image(out).save(filename)
        #     filenames.append(filename)
        # return filenames

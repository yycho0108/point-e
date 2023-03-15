#!/usr/bin/env python3

import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

has_cuda: bool = True


class GLIDESampler:
    def __init__(self, device,
                 batch_size=1,
                 guidance_scale=3.0,
                 upsample_temp=0.997,
                 num_step: int = 100
                 ):
        self.device = device
        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.upsample_temp = upsample_temp
        self.num_step = num_step

        self.create_model()
        self.create_upsampler()

    def create_model(self):
        device = self.device
        # Create base model.
        options = model_and_diffusion_defaults()
        options['use_fp16'] = has_cuda
        # use 100 diffusion steps for fast sampling
        options['timestep_respacing'] = F'{self.num_step}'
        model, diffusion = create_model_and_diffusion(**options)
        model.eval()
        if has_cuda:
            model.convert_to_fp16()
        model.to(device)
        model.load_state_dict(load_checkpoint('base', device))
        print('total base parameters', sum(x.numel()
              for x in model.parameters()))

        self.model = model
        self.options = options
        self.diffusion = diffusion

    def create_upsampler(self):
        device = self.device

        # Create upsampler model.
        options_up = model_and_diffusion_defaults_upsampler()
        options_up['use_fp16'] = has_cuda
        # use 27 diffusion steps for very fast sampling
        options_up['timestep_respacing'] = 'fast27'
        model_up, diffusion_up = create_model_and_diffusion(**options_up)
        model_up.eval()
        if has_cuda:
            model_up.convert_to_fp16()
        model_up.to(device)
        model_up.load_state_dict(load_checkpoint('upsample', device))
        print('total upsampler parameters', sum(x.numel()
              for x in model_up.parameters()))

        self.model_up = model_up
        self.options_up = options_up
        self.diffusion_up = diffusion_up

    def create_base_images(self, prompt: str):
        device = self.device
        batch_size = self.batch_size
        options = self.options
        model = self.model
        diffusion = self.diffusion

        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] *
                batch_size +
                [uncond_tokens] *
                batch_size,
                device=device),
            mask=th.tensor(
                [mask] *
                batch_size +
                [uncond_mask] *
                batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + self.guidance_scale * \
                (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model.del_cache()
        return samples

    def upsample_images(self,
                        prompt: str,
                        samples
                        ):
        # @title Upsample the 64x64 samples
        ##############################
        # Upsample the 64x64 samples #
        ##############################
        device = self.device
        batch_size = self.batch_size
        model_up = self.model_up
        options_up = self.options_up
        diffusion_up = self.diffusion

        tokens = model_up.tokenizer.encode(prompt)
        tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
            tokens, options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model_up.del_cache()
        up_shape = (
            batch_size,
            3,
            options_up["image_size"],
            options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * self.upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model_up.del_cache()

        return up_samples

    def text_to_image(self, prompt):
        samples = self.create_base_images(prompt)
        outputs = self.upsample_images(prompt, samples)
        scaled = ((outputs + 1) * 127.5).round().clamp(0, 255).to(th.uint8)
        return scaled / 255.0  # ??

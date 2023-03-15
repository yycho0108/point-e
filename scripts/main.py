#!/usr/bin/env python3

import torch as th
# from text_to_image import text_to_image
# from text_to_image_v2 import Sampler
# from text_to_image_v3 import GLIDESampler
from text_to_image_v4 import text_to_image
from image_to_cloud import image_to_cloud
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import cv2


def text_to_text(x: str) -> str:
    """ Potentially refine the text through models such as GPT. """
    return x


def cloud_to_mesh():
    raise NotImplementedError('`cloud_to_mesh` currently not implemented.')


def main():
    seed: int = 1
    num: int = 4
    # text: str = 'One corgi wearing a red santa hat in a white background'
    # text: str = '3d rendering of one corgi in a white background'
    # text: str = 'A household object that can turn bolts'
    # text:str = 'Household object that can be improvised to open a wine bottle.'
    # text: str = 'One flathead screwdriver in a white background'
    text: str = 'Image of a single flathead screwdriver'
    num_step: int = 100
    device: str = 'cuda:0'
    out_dir: str = '/tmp/out'
    guidance_scale: float = 7.5
    upsample_temp: float = 0.997
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    th.manual_seed(seed)

    if True:
        # text -> image
        text = text_to_text(text)

        # text = th.as_tensor(text, device=device) #??
        with th.inference_mode():
            # v1
            # image = text_to_image(text, num, steps=num_step,
            #                       device=device)

            # v2
            # sampler = Sampler(steps=num_step, weight=3.0, device=device)
            # image = sampler.run_all(text, num, num)

            # v3
            # sampler = GLIDESampler(device, num,
            #                        guidance_scale=guidance_scale,
            #                        upsample_temp=upsample_temp,
            #                        num_step=num_step)
            # image = sampler.text_to_image(text)

            # v4
            image = text_to_image(text,
                                  num=num,
                                  device=device,
                                  guidance_scale=guidance_scale,
                                  num_inference_steps=num_step
                                  )

        # [dummy]
        # image = th.zeros((num, 3, 256, 256), dtype=th.float32,
        #                  device=device)

        # tensor -> PIL (disk)
        for i, x in enumerate(image):
            filename: str = F'{out_dir}/image-{i:02d}.png'
            if isinstance(x, th.Tensor):
                save_image(x, filename)
            else:  # if isinstance(x, Image):
                x.save(filename)

        # denoise
        for i in range(num):
            src = cv2.imread(F'{out_dir}/image-{i:02d}.png')
            dst = cv2.GaussianBlur(src, (5, 5), 3.0)
            dst = cv2.fastNlMeansDenoisingColored(dst)
            cv2.imwrite(F'{out_dir}/denoise-{i:02d}.png', dst)

    # image->cloud -> disk
    cache = {}
    for i in range(num):
        image = Image.open(F'{out_dir}/image-{i:02d}.png')
        # image = Image.open(F'{out_dir}/image-{i:02d}.png')
        with th.inference_mode():
            cloud = image_to_cloud(image, device, cache=cache)
        with open(F'{out_dir}/cloud-{i:02d}.ply', 'wb') as fp:
            cloud.write_ply(fp)


if __name__ == '__main__':
    main()

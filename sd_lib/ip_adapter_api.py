# -- coding: utf-8 --
# @Time : 2023/8/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import torch
from PIL import Image
from cv2box import CVImage
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, \
    DDIMScheduler, AutoencoderKL

from sd_lib.models import IPAdapter, IPAdapterPlus

SD_PRETRAIN = './sd_models/stable-diffusion-v1-5'
VAE_PRETRAIN = './sd_models/stabilityai_sd-vae-ft-mse'
CLIP_IMAGE_PRETRAIN = './sd_models/clip_image_encoder'
IMAGE_PROJ_PRETRAIN = './sd_models/ip_adapter_image_proj/ip-adapter_sd15.bin'
IMAGE_PROJ_PLUS_PRETRAIN = './sd_models/ip_adapter_image_proj/ip-adapter-plus_sd15.bin'


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class IpAdapterAPI():
    def __init__(self, device="cuda"):
        # load SD pipeline
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(VAE_PRETRAIN).to(dtype=torch.float16)

        sd_pipe = StableDiffusionPipeline.from_pretrained(
            SD_PRETRAIN,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None
        )

        # load ip-adapter
        # self.ip_model = IPAdapter(sd_pipe, CLIP_IMAGE_PRETRAIN, IMAGE_PROJ_PRETRAIN, device)
        self.ip_model = IPAdapterPlus(sd_pipe, CLIP_IMAGE_PRETRAIN, IMAGE_PROJ_PLUS_PRETRAIN, device, num_tokens=16)

    def forward(self, image_pil):
        # generate image variations
        images = self.ip_model.generate(pil_image=image_pil, num_samples=4, num_inference_steps=50, seed=42)
        grid = image_grid(images, 1, 4)
        grid.show()
        return grid


if __name__ == '__main__':
    # read image prompt
    ia = IpAdapterAPI()
    image = Image.open('resources/for_sd/girl_reading_512_crop.png')
    image.resize((256, 256))
    out_image = ia.forward(image)

# -- coding: utf-8 --
# @Time : 2023/9/4
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import torch
from cv2box import CVImage

from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.image_processor import VaeImageProcessor

from sd_lib.prompt2prompt import ddim_inversion, null_optimization, EmptyControl

SD_PRETRAIN = './sd_models/stable-diffusion-v1-5'


class DDIMInversion:
    def __init__(self, device='cuda', num_inv_steps=50):
        self.device = device
        self.num_inv_steps = num_inv_steps
        noise_scheduler = DDIMScheduler.from_pretrained(SD_PRETRAIN, subfolder='scheduler')
        noise_scheduler.set_timesteps(self.num_inv_steps)
        # noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=1000,
        #     beta_start=0.00085,
        #     beta_end=0.012,
        #     beta_schedule="linear",
        #     clip_sample=False,
        #     set_alpha_to_one=False,
        #     steps_offset=1,
        # )
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            SD_PRETRAIN,
            torch_dtype=torch.float16,
            scheduler=noise_scheduler,
            # unet=unet,
            feature_extractor=None,
            safety_checker=None
        ).to(self.device)

        # self.sd_pipe.enable_model_cpu_offload()

        self.vae_scale_factor = 2 ** (len(self.sd_pipe.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def forward(self, image_in, prompt, null_optim=True):
        image_in_pil = CVImage(image_in).pillow()
        image_in_pt = self.image_processor.preprocess(image_in_pil)

        latents = self.sd_pipe.vae.encode(
            image_in_pt.to(self.device, dtype=self.sd_pipe.vae.dtype)).latent_dist.sample()
        latents = latents * 0.18215

        ddim_inv_latents = ddim_inversion(
            self.sd_pipe, self.sd_pipe.scheduler, latents=latents,
            num_inv_steps=self.num_inv_steps, prompt=prompt)

        if null_optim:
            num_inner_steps = 10
            uncond_embeddings = null_optimization(self.sd_pipe, self.sd_pipe.scheduler, ddim_inv_latents, self.num_inv_steps, num_inner_steps, prompt)
            # null_text_rec, _ = ptp_utils.text2image_ldm_stable(StableDiffuser, [prompt], EmptyControl(), latent=x_t,
            #                                                    uncond_embeddings=uncond_embeddings)
            # ptp_utils.view_images(null_text_rec)
            return ddim_inv_latents[-1], uncond_embeddings
        else:
            return ddim_inv_latents[-1]


if __name__ == '__main__':
    image_p = 'resources/for_sd/girl_reading_512_crop.png'
    blip_prompt = 'a woman reading a book'
    ddimi = DDIMInversion(device='cuda', num_inv_steps=25)
    latent_out, uncond_embedding = ddimi.forward(image_p, blip_prompt, null_optim=True)

    print(latent_out.shape)
    print(uncond_embedding.shape)

    regenerate_image = ddimi.sd_pipe(
        height=512,
        width=512,
        prompt=blip_prompt,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=EmptyControl(),
        negative_prompt_embeds=uncond_embedding,
        latents=latent_out,
        return_dict=False,
    )[0][0]
    print(regenerate_image.size)  # pillow
    CVImage(regenerate_image, 'pillow').show()

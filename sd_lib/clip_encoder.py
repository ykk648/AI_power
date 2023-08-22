# -- coding: utf-8 --
# @Time : 2023/8/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import torch
from PIL import Image

from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

"""
clip image encoder from:
https://github.com/tencent-ailab/IP-Adapter/blob/00cbac222600928f68103c16ed9931074fca9edd/ip_adapter/ip_adapter.py#L45
"""

CLIP_TEXT_PRETRAIN = './sd_models/stable-diffusion-v1-5'
CLIP_IMAGE_PRETRAIN = './sd_models/clip_image_encoder'
IMAGE_PROJ_PRETRAIN = './sd_models/ip_adapter_image_proj/ip-adapter_sd15.bin'


class ClipText:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_TEXT_PRETRAIN, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(CLIP_TEXT_PRETRAIN, subfolder="text_encoder").cuda()
        self.text_encoder.requires_grad_(False)

    def forward(self, prompt: list[str]):
        # (b,77)
        prompt_ids = self.tokenizer(
            prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True,
            return_tensors="pt"
        ).input_ids
        # (b,77,768)
        encoder_hidden_states = self.text_encoder(prompt_ids[0][np.newaxis, :].cuda())[0]
        return encoder_hidden_states


class ClipImage:
    def __init__(self):
        self.device = 'cuda'
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(CLIP_IMAGE_PRETRAIN).to(self.device,
                                                                                                   dtype=torch.float16)
        self.clip_image_processor = CLIPImageProcessor()

    def forward(self, pil_image):
        """
        Args:
            pil_image: RGB
        Returns: torch.Size([1, 1024])
        """
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
        return clip_image_embeds


class ImageProj:
    def __init__(self, num_tokens=4):
        from sd_lib.local_model_structure import ImageProjModel
        self.device = "cuda"
        self.image_proj_model = ImageProjModel(
            cross_attention_dim=768,  # self.pipe.unet.config.cross_attention_dim
            clip_embeddings_dim=1024,  # self.image_encoder.config.projection_dim
            clip_extra_context_tokens=num_tokens,
        ).to(self.device, dtype=torch.float16)
        state_dict = torch.load(IMAGE_PROJ_PRETRAIN, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])

    def forward(self, clip_image_embeds):
        """
        Args:
            clip_image_embeds: torch.Size([1, 1024])
        Returns: torch.Size([1, 4, 768])
        """
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds


if __name__ == '__main__':
    image_p = 'resources/for_sd/girl_reading_512_crop.png'
    clip_image = ClipImage()
    image_embedding = clip_image.forward(Image.open(image_p))
    print(image_embedding.shape)

    ip = ImageProj()
    image_proj_embedding, _ = ip.forward(image_embedding)
    print(image_proj_embedding.shape)

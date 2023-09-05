# -- coding: utf-8 --
# @Time : 2023/9/4
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import os
import imageio
import numpy as np
from typing import Union
import torch
import torchvision
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
from torch.optim import Adam

"""
Mainly from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/util.py
partial from https://github.com/open-mmlab/mmagic/blob/main/projects/prompt_to_prompt/inversions/ddim_inversion.py
"""

class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def prev_step(model_output: Union[torch.FloatTensor, np.ndarray],
              timestep: int, sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    prev_timestep = timestep - (
            ddim_scheduler.config.num_train_timesteps //
            ddim_scheduler.num_inference_steps)
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        ddim_scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0 else ddim_scheduler.final_alpha_cumprod)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (
                                   sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample \
                  + pred_sample_direction
    return prev_sample


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


def get_noise_pred(pipeline, ddim_scheduler, latents, t, is_forward=True, context=None, guidance_scale=7.5):
    latents_input = torch.cat([latents] * 2)
    if context is None:
        context = context
    guidance_scale = 1 if is_forward else guidance_scale
    noise_pred = pipeline.unet(
        latents_input, t, encoder_hidden_states=context)['sample']
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond)
    if is_forward:
        latents = next_step(noise_pred, t, latents, ddim_scheduler)
    else:
        latents = prev_step(noise_pred, t, latents, ddim_scheduler)
    return latents


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


# mod from https://github.com/open-mmlab/mmagic/blob/main/projects/prompt_to_prompt/inversions/null_text_inversion.py
def null_optimization(pipeline, ddim_scheduler, latents, num_inv_steps, num_inner_steps=10, prompt="",
                      guidance_scale=7.5, epsilon=1e-5):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    uncond_embeddings_list = []
    latent_cur = latents[-1]
    bar = tqdm(total=num_inner_steps * num_inv_steps)
    for i in range(num_inv_steps):
        if i ==24:
            print('24')
        uncond_embeddings = uncond_embeddings.clone().detach()
        uncond_embeddings.requires_grad = True
        optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
        latent_prev = latents[len(latents) - i - 2]
        t = ddim_scheduler.timesteps[i]
        with torch.no_grad():
            noise_pred_cond = get_noise_pred_single(
                latent_cur, t, cond_embeddings, pipeline.unet)
        for j in range(num_inner_steps):
            noise_pred_uncond = get_noise_pred_single(
                latent_cur, t, uncond_embeddings, pipeline.unet)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond)
            latents_prev_rec = prev_step(noise_pred, t, latent_cur, ddim_scheduler)
            loss = F.mse_loss(latents_prev_rec, latent_prev)
            # loss.requires_grad_(True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            bar.update()
            if loss_item < epsilon + i * 2e-5:
                break
        for j in range(j + 1, num_inner_steps):
            bar.update()
        uncond_embeddings_list.append(uncond_embeddings[:1].detach())
        with torch.no_grad():
            context = torch.cat([uncond_embeddings, cond_embeddings])
            latent_cur = get_noise_pred(pipeline, ddim_scheduler, latent_cur, t, False, context)
    bar.close()
    return uncond_embeddings_list


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, latents, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, latents, num_inv_steps, prompt)
    return ddim_latents

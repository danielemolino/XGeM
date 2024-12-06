from typing import Dict, List
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
import copy
from functools import partial
from contextlib import contextmanager

from .common.get_model import get_model, register
from .sd import DDPM

version = '0'
symbol = 'MedCoDi_M'


@register('MedCoDi_M', version)
class MedCoDi_M(DDPM):
    def __init__(self,
                 autokl_cfg=None,
                 optimus_cfg=None,
                 clip_cfg=None,
                 vision_scale_factor=0.1812,
                 text_scale_factor=4.3108,
                 scale_by_std=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if autokl_cfg is not None:
            self.autokl = get_model()(autokl_cfg)

        if optimus_cfg is not None:
            self.optimus = get_model()(optimus_cfg)

        if clip_cfg is not None:
            self.clip = get_model()(clip_cfg)

        if not scale_by_std:
            self.vision_scale_factor = vision_scale_factor
            self.text_scale_factor = text_scale_factor
        else:
            self.register_buffer("text_scale_factor", torch.tensor(text_scale_factor))
            self.register_buffer('vision_scale_factor', torch.tensor(vision_scale_factor))

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def autokl_encode(self, image):
        encoder_posterior = self.autokl.encode(image)
        z = encoder_posterior.sample().to(image.dtype)
        return self.vision_scale_factor * z

    @torch.no_grad()
    def autokl_decode(self, z):
        z = 1. / self.vision_scale_factor * z
        return self.autokl.decode(z)

    @torch.no_grad()
    def optimus_encode(self, text):
        if isinstance(text, List):
            tokenizer = self.optimus.tokenizer_encoder
            token = [tokenizer.tokenize(sentence.lower()) for sentence in text]
            token_id = []
            for tokeni in token:
                token_sentence = [tokenizer._convert_token_to_id(i) for i in tokeni]
                token_sentence = tokenizer.add_special_tokens_single_sentence(token_sentence)
                token_id.append(torch.LongTensor(token_sentence))
            token_id = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=0.0)[:, :512]
        else:
            token_id = text
        token_id = token_id.to(self.device)
        z = self.optimus.encoder(token_id, attention_mask=(token_id > 0))[1]
        z_mu, z_logvar = self.optimus.encoder.linear(z).chunk(2, -1)
        return z_mu.squeeze(1) * self.text_scale_factor

    @torch.no_grad()
    def optimus_decode(self, z, temperature=1.0, max_length=30):
        z = 1.0 / self.text_scale_factor * z
        z = z.to(self.device)
        return self.optimus.decode(z, temperature, max_length=max_length)

    @torch.no_grad()
    def clip_encode_text(self, text, encode_type='encode_text'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip(text, encode_type)
        self.clip.encode_type = swap_type
        return embedding

    @torch.no_grad()
    def clip_encode_vision(self, vision, encode_type='encode_vision'):
        swap_type = self.clip.encode_type
        self.clip.encode_type = encode_type
        embedding = self.clip(vision, encode_type)
        self.clip.encode_type = swap_type
        return embedding

    def forward(self, x=None, c=None, noise=None, xtype='frontal', ctype='text', u=None, return_algined_latents=False, env_enc=False):
        if isinstance(x, list):
            t = torch.randint(0, self.num_timesteps, (x[0].shape[0],), device=x[0].device).long()
        else:
            t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, noise, xtype, ctype, u, return_algined_latents, env_enc)

    def apply_model(self, x_noisy, t, cond, xtype='frontal', ctype='text', u=None, return_algined_latents=False, env_enc=False):
        return self.model.diffusion_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents, env_enc=env_enc)

    def get_pixel_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=-0.0)
        return loss

    def get_text_loss(self, pred, target):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss

    def p_losses(self, x_start, cond, t, noise=None, xtype='frontal', ctype='text', u=None,
                 return_algined_latents=False, env_enc=False):
        if isinstance(x_start, list):
            noise = [torch.randn_like(x_start_i) for x_start_i in x_start] if noise is None else noise
            x_noisy = [self.q_sample(x_start=x_start_i, t=t, noise=noise_i) for x_start_i, noise_i in
                       zip(x_start, noise)]
            if not env_enc:
                model_output = self.apply_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents, env_enc)
            else:
                model_output, h_con = self.apply_model(x_noisy, t, cond, xtype, ctype, u, return_algined_latents, env_enc)
            if return_algined_latents:
                return model_output

            loss_dict = {}

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()

            loss = 0.0
            for model_output_i, target_i, xtype_i in zip(model_output, target, xtype):
                if xtype_i == 'frontal':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                elif xtype_i == 'text':
                    loss_simple = self.get_text_loss(model_output_i, target_i).mean([1])
                elif xtype_i == 'lateral':
                    loss_simple = self.get_pixel_loss(model_output_i, target_i, mean=False).mean([1, 2, 3])
                loss += loss_simple.mean()

                # Controlliamo se il modello ha restituito anche h_con
                # In tal caso, abbiamo le rappresentazioni latenti delle due modalità
                # estratte dagli environmental encoder, essendo due tensori di dimensione batch_sizex1x1280
                # possiamo utilizzarli per calcolare anche un termine di contrastive loss (crossentropy come in CLIP)
                if h_con is not None:
                    def similarity(z_a, z_b):
                        return F.cosine_similarity(z_a, z_b)

                    z_a, z_b = h_con

                    z_a = z_a / z_a.norm(dim=-1, keepdim=True)
                    z_b = z_b / z_b.norm(dim=-1, keepdim=True)

                    logits_a = z_a.squeeze() @ z_b.squeeze().t()
                    logits_b = z_a.squeeze() @ z_b.squeeze().t()

                    labels = torch.arange(len(z_a)).to(z_a.device)

                    loss_a = F.cross_entropy(logits_a, labels)
                    loss_b = F.cross_entropy(logits_b, labels)

                    loss_con = (loss_a + loss_b) / 2
                    loss += loss_con

            return loss / len(xtype)

        else:
            noise = torch.randn_like(x_start) if noise is None else noise
            x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
            model_output = self.apply_model(x_noisy, t, cond, xtype, ctype)

            loss_dict = {}

            if self.parameterization == "x0":
                target = x_start
            elif self.parameterization == "eps":
                target = noise
            else:
                raise NotImplementedError()

            if xtype == 'frontal':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            elif xtype == 'text':
                loss_simple = self.get_text_loss(model_output, target).mean([1])
            elif xtype == 'lateral':
                loss_simple = self.get_pixel_loss(model_output, target, mean=False).mean([1, 2, 3])
            loss = loss_simple.mean()
            return loss

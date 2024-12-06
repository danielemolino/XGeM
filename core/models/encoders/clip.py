from typing import List
import os

import torch
import torch.nn as nn
import numpy as np
from functools import partial
from core.models.common.get_model import register
from einops import rearrange

from transformers import CLIPTokenizer, CLIPTextModel
from .clip_modules import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPConfig


version = '0'
symbol = 'clip'


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


@register('clip_text_frozen', version)
class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)
        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


@register('clip_frozen', version)
class FrozenCLIP(AbstractEncoder):
    def __init__(self,
                 version="openai/clip-vit-large-patch14",
                 max_length=77,
                 encode_type='encode_text',
                 fp16=False,
                 data_dir='.'):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        config = CLIPConfig.from_pretrained(version)
        self.model = CLIPModel(config, add_temporal_attention=True)
        self.max_length = max_length
        self.encode_type = encode_type
        self.fp16 = fp16

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):
        return self.model.text_projection.weight.device

    def get_device(self):
        # A trick to get device
        return self.model.text_projection.weight.device

    def freeze(self, modules):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze(self, modules):
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True

    def encode_text_pooled(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        outputs = self.model.get_text_features(input_ids=tokens)
        return outputs

    def encode_vision_pooled(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        pixels = inputs['pixel_values'].half() if self.fp16 else inputs['pixel_values']
        pixels = pixels.to(self.get_device())
        return self.model.get_image_features(pixel_values=pixels)

    def encode_text_noproj(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.get_device())
        if self.dtype == torch.half:
            tokens = tokens.short()
        outputs = self.model.text_model(input_ids=tokens)
        return outputs.last_hidden_state

    def encode_vision_noproj(self, vision_inputs):
        # vision_inputs = ((vision_inputs + 1) / 2).to('cpu').numpy()
        vision_inputs = vision_inputs.to('cpu').numpy()

        if vision_inputs.ndim == 5:
            num_frames = vision_inputs.shape[2]
            vision_inputs = rearrange(vision_inputs, 'b c f h w -> (b f) h w c')
        else:
            num_frames = 1
            vision_inputs = rearrange(vision_inputs, 'b c h w -> b h w c')

        vision_inputs = [vi for vi in vision_inputs]
        inputs = self.processor(images=vision_inputs, return_tensors="pt")

        pixels = inputs['pixel_values'].to(self.dtype).to(self.device)

        if num_frames > 1:
            pixels = rearrange(pixels, '(b f) h w c -> b f h w c', f=num_frames)
        outputs = self.model.vision_model(pixel_values=pixels)
        return outputs

    def encode_text(self, text):
        if isinstance(text, List):
            batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            tokens = batch_encoding["input_ids"].to(self.get_device())
        else:
            tokens = text
        outputs = self.model.text_model(input_ids=tokens)
        z_pooled = outputs.pooler_output
        z_pooled = self.model.text_projection(z_pooled)
        z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z_pooled.unsqueeze(1)

    def encode_vision(self, images):
        z = self.encode_vision_noproj(images)
        z_pooled = z.pooler_output
        z_pooled = self.model.visual_projection(z_pooled)
        z_pooled = z_pooled / torch.norm(z_pooled, dim=-1, keepdim=True)
        return z_pooled.unsqueeze(1)

    def encode(self, *args, **kwargs):
        return getattr(self, self.encode_type)(*args, **kwargs)

    def forward(self, input, encode_type):
        if encode_type == 'encode_text':
            return self.encode_text(input)
        elif encode_type == 'encode_vision':
            # Se il numero di canali è 1, copiamo l'immagine su 3 canali essendo un'immagine in scala di grigi
            if input.shape[1] == 1:
                input = torch.cat([input, input, input], dim=1)
            return self.encode_vision(input)


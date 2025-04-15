import os
from core.models.MedCoDi_M_wrapper import MedCoDi_M_wrapper
from PIL import Image
import torch
import torch.nn as nn
from core.common.utils import remove_duplicate_word
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from skimage import exposure
import pandas as pd
import time

def run_demo():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(torch.cuda.device_count())

    print('Starting model loading...')
    model_load_paths = ['MedCoDi-M.pt']
    inference_tester = MedCoDi_M_wrapper(model='MedCoDi_M',
                                        data_dir='./Weights',
                                        pth=model_load_paths,
                                        load_weights=True)
    print('Putting model on device...')
    inference_tester.to(device)
    print('Model instantiated correctly!')

    frontal_xray = './Examples/Frontal.tiff'
    lateral_xray = './Examples/Lateral.tiff'
    report = 'Minimal bilateral pleural effusions and bibasilar atelectasis.  No evidence for congestive heart failure.'
    
    ########################
    ######## T->F ##########
    ########################
    prompt = report
    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    h, w = [256, 256]
    shapes = []
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['frontal'],
        condition_types=['text'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()
    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/T->F.png')

    ########################
    ######## F->L ##########
    ########################
    im = tifffile.imread(frontal_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    h, w = [256, 256]
    shapes = []
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['lateral'],
        condition_types=['frontal'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/F->L.png')

    ########################
    ######## F->T ##########
    ########################
    im = tifffile.imread(frontal_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    shapes = []
    n = 768
    shape = [1, n]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['text'],
        condition_types=['frontal'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x_b = inference_tester.net.optimus_decode(z[0], max_length=100)
    x = x_b[0]
    x = [a.tolist() for a in x]
    rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
    rec_text = "".join(rec_text).replace('<BOS>', '').replace('<EOS>', '')
    print(f'Generation F->T: {rec_text}')
    
    ########################
    ######## L->T ##########
    ########################
    im = tifffile.imread(lateral_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    shapes = []
    n = 768
    shape = [1, n]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['text'],
        condition_types=['lateral'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x_b = inference_tester.net.optimus_decode(z[0], max_length=100)
    x = x_b[0]
    x = [a.tolist() for a in x]
    rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
    rec_text = "".join(rec_text).replace('<BOS>', '').replace('<EOS>', '')
    print(f'Generation L->T: {rec_text}')
    
    ########################
    ######## T->L ##########
    ########################
    prompt = report

    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []

    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    h, w = [256, 256]
    shapes = []
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['lateral'],
        condition_types=['text'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/T->L.png')

    ########################
    ######## L->F ##########
    ########################
    im = tifffile.imread(lateral_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    h, w = [256, 256]
    shapes = []
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['frontal'],
        condition_types=['lateral'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/L->F.png')

    ########################
    ######## F+L->T ########
    ########################
    im1 = tifffile.imread(frontal_xray)
    im2 = tifffile.imread(lateral_xray)
    im1 = torch.tensor(im1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    im2 = torch.tensor(im2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim1 = inference_tester.net.clip_encode_vision(im1, encode_type='encode_vision').to(device)
    cim2 = inference_tester.net.clip_encode_vision(im2, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im1).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim1]))
    conditioning.append(torch.cat([uim, cim2]))

    shapes = []
    n = 768
    shape = [1, n]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['text'],
        condition_types=['frontal', 'lateral'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x_b = inference_tester.net.optimus_decode(z[0], max_length=100)
    x = x_b[0]
    x = [a.tolist() for a in x]
    rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
    rec_text = "".join(rec_text).replace('<BOS>', '').replace('<EOS>', '')
    print(f'Generation F+L->T: {rec_text}')

    ########################
    ######## F+T->L ########
    ########################
    prompt = report
    im = tifffile.imread(frontal_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    shapes = []
    h, w = [256, 256]
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['lateral'],
        condition_types=['text', 'frontal'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/F+T->L.png')

    ########################
    ######## L+T->F ########
    ########################
    prompt = report
    im = tifffile.imread(lateral_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    shapes = []
    h, w = [256, 256]
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['frontal'],
        condition_types=['text', 'lateral'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/L+T->F.png')

    ########################
    ######## F->T+L ########
    ########################
    im = tifffile.imread(frontal_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    shapes = []
    n = 768
    shape = [1, n]
    shapes.append(shape)
    h, w = [256, 256]
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['text', 'lateral'],
        condition_types=['frontal'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    x_b = inference_tester.net.optimus_decode(z[0], max_length=100)
    x = x_b[0]
    x = [a.tolist() for a in x]
    rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
    rec_text = "".join(rec_text).replace('<BOS>', '').replace('<EOS>', '')
    print(f'Generation F->T+L: {rec_text}')

    x = inference_tester.net.autokl_decode(z[1])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/F->T+L.png')

    ########################
    ######## L->T+F ########
    ########################
    im = tifffile.imread(lateral_xray)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    conditioning = []
    cim = inference_tester.net.clip_encode_vision(im, encode_type='encode_vision').to(device)
    uim = None
    scale = 7.5
    dummy = torch.zeros_like(im).to(device)
    uim = inference_tester.net.clip_encode_vision(dummy, encode_type='encode_vision').to(device)
    conditioning.append(torch.cat([uim, cim]))

    shapes = []
    n = 768
    shape = [1, n]
    shapes.append(shape)

    h, w = [256, 256]
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['text', 'frontal'],
        condition_types=['lateral'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x_b = inference_tester.net.optimus_decode(z[0], max_length=100)
    x = x_b[0]
    x = [a.tolist() for a in x]
    rec_text = [inference_tester.net.optimus.tokenizer_decoder.decode(a) for a in x]
    rec_text = "".join(rec_text).replace('<BOS>', '').replace('<EOS>', '')
    print(f'Generation L->T+F: {rec_text}')

    x = inference_tester.net.autokl_decode(z[1])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'Examples/L->T+F.png')

    ########################
    ######## T->F+L ########
    ########################
    prompt = report

    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    shapes = []
    h, w = [256, 256]
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)
    shapes.append(shape)

    z, _ = inference_tester.sampler.sample(
        steps=50,
        shape=shapes,
        condition=conditioning,
        unconditional_guidance_scale=scale,
        xtype=['frontal', 'lateral'],
        condition_types=['text'],
        eta=1,
        verbose=False,
        mix_weight={'lateral': 1, 'text': 1, 'frontal': 1})

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    x = inference_tester.net.autokl_decode(z[1])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im2 = x[0].cpu().numpy()

    # facciamo un subplot
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im, cmap='gray')
    axs[1].imshow(im2, cmap='gray')
    plt.savefig(f'Examples/T->F+L.png')

if __name__ == "__main__":
    run_demo()

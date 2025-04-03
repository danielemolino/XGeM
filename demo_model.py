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
    def get_gpu_usage(device):
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024 ** 2
        print(f"Memory used: {mem_used_MB:.2f} MB")


    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(torch.cuda.device_count())

    get_gpu_usage(device)

    print('Starting model loading...')
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_video_diffuser_8frames.pth']
    inference_tester = MedCoDi_M_wrapper(model='MedCoDi_M',
                                        data_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/checkpoints',
                                        pth=model_load_paths,
                                        load_weights=True)  # turn on fp16=True if loading fp16 weights
    print('Putting model on device...')
    get_gpu_usage(device)
    inference_tester.to(device)
    get_gpu_usage(device)
    print('Model instantiated correctly!')

    get_gpu_usage(device)

    clip_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/Clip_Training/saved_checkpoints/checkpoint_29_epoch_Training_Clip_5e^-5.pt'
    a, b = inference_tester.net.clip.load_state_dict(torch.load(clip_weights, map_location=device), strict=False)
    print('Clip weights loaded correctly!')

    optimus_weights = '/mimer/NOBACKUP/groups/naiss2023-6-336/dmolino/CoDI-Original/Report_Training/saved_checkpoints/VAE/checkpoint_99_epoch_VAE-Training-Prova1.pt'
    optimus_weights = torch.load(optimus_weights, map_location='cpu')
    a, b = inference_tester.net.optimus.load_state_dict(optimus_weights, strict=False)
    print('Optimus weights loaded correctly!')

    frontal_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/CXR_Training/saved_checkpoints/Frontal/checkpoint_99_epoch_Training-Frontal-MultiPrompt-New.pt'
    frontal_weights = torch.load(frontal_weights, map_location='cpu')
    for key in list(frontal_weights.keys()):  # Utilizza list per creare una copia delle chiavi
        if 'unet_image' in key:
            value = frontal_weights.pop(key)
            new_key = key.replace('unet_image', 'unet_frontal')
            frontal_weights[new_key] = value

    lateral_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/CXR_Training/saved_checkpoints/Lateral/checkpoint_99_epoch_Training-Lateral-MultiPrompt-New.pt'
    lateral_weights = torch.load(lateral_weights, map_location='cpu')
    for key in list(lateral_weights.keys()):  # Utilizza list per creare una copia delle chiavi
        if 'unet_image' in key:
            value = lateral_weights.pop(key)
            new_key = key.replace('unet_image', 'unet_lateral')
            lateral_weights[new_key] = value

    a, b = inference_tester.net.model.load_state_dict(frontal_weights, strict=False)
    print('Frontal weights loaded correctly!')
    a, b = inference_tester.net.model.load_state_dict(lateral_weights, strict=False)
    print('Lateral weights loaded correctly!')

    text_weights = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/Report_Training/saved_checkpoints/checkpoint_99_epoch_Report_Diffusion_Training-MultiPrompt-New.pt'
    text_weights = torch.load(text_weights, map_location='cpu')
    a, b = inference_tester.net.model.load_state_dict(text_weights, strict=False)
    print('Text weights loaded correctly!')

    # Abbiamo caricato i diffusivi, ma adesso vanno caricati gli environmental encoders e i cross-attention layers
    blocks_to_load = [inference_tester.net.model.diffusion_model.unet_frontal.connecters_out,
                    inference_tester.net.model.diffusion_model.unet_text.connecters_out,
                    inference_tester.net.model.diffusion_model.unet_lateral.connecters_out,
                    inference_tester.net.model.diffusion_model.unet_frontal.input_block_connecters_in,
                    inference_tester.net.model.diffusion_model.unet_text.input_block_connecters_in,
                    inference_tester.net.model.diffusion_model.unet_lateral.input_block_connecters_in,
                    inference_tester.net.model.diffusion_model.unet_frontal.output_block_connecters_in,
                    inference_tester.net.model.diffusion_model.unet_text.output_block_connecters_in,
                    inference_tester.net.model.diffusion_model.unet_lateral.output_block_connecters_in]

    string_to_load = ['model.model.diffusion_model.unet_frontal.connecters_out',
                    'model.model.diffusion_model.unet_text.connecters_out',
                    'model.model.diffusion_model.unet_lateral.connecters_out',
                    'model.model.diffusion_model.unet_frontal.input_block_connecters_in',
                    'model.model.diffusion_model.unet_text.input_block_connecters_in',
                    'model.model.diffusion_model.unet_lateral.input_block_connecters_in',
                    'model.model.diffusion_model.unet_frontal.output_block_connecters_in',
                    'model.model.diffusion_model.unet_text.output_block_connecters_in',
                    'model.model.diffusion_model.unet_lateral.output_block_connecters_in']

    path1 = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/EnvEnc_Training/saved_checkpoints/Training-EnvEnc-F+L+Contrastive-Loss/50'
    path2 = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/EnvEnc_Training/saved_checkpoints/Training-EnvEnc-T+Lfreezed+Contrastive-Loss/50'

    for string, block in zip(string_to_load, blocks_to_load):
        block_weights = f'{path1}/{string}.pt'
        # controlliamo se il path1 esiste, altrimenti printiamo un messaggio di errore e passiamo al prossimo blocco
        if not os.path.exists(block_weights):
            print(f"Path {block_weights} does not exist. Skipping block...")
            continue
        block_weights = torch.load(block_weights, map_location='cpu')
        a, b = block.load_state_dict(block_weights, strict=True)

    print('Environmental encoders and cross-attention layers loaded correctly!')

    get_gpu_usage(device)

    """
    df1 = pd.read_csv('csv/test_short_frontal_clean.csv')
    df2 = pd.read_csv('csv/test_short_lateral_clean.csv')

    # campioniamo 5 righe da df1 e 5 da df2
    df1 = df1.sample(1)
    df2 = df2.sample(1)

    ########################
    ######## T->F ##########
    ########################
    # Iniziamo a contare il tempo
    start = time.time()
    get_gpu_usage(device)

    prompt = df1['report'].values
    prompt = prompt[0]
    # 1) Passiamo il prompt al CLIP
    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    print('SHAPE:', str(ctx.shape))

    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    h, w = [256, 256]
    shapes = []
    shape = [1, 4, h // 8, w // 8]
    shapes.append(shape)

    get_gpu_usage(device)

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

    get_gpu_usage(device)

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])

    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    # mettiamo come titolo il prompt
    plt.imshow(im, cmap='gray')
    plt.savefig(f'plots/T->F.png')

    get_gpu_usage(device)
    end_time = time.time() - start
    print(f'T->F time: {end_time}')

    ########################
    ######## F->L ##########
    ########################
    start = time.time()
    # carichiamo la frontale
    path1 = df1['path'].values
    path1 = path1[0]
    path1 = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/MIMIC/' + path1
    path1 = path1.replace('.dcm', '.tiff')
    im = tifffile.imread(path1)
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

    # adesso la passiamo al decoder
    x = inference_tester.net.autokl_decode(z[0])
    x = torch.clamp((x[0] + 1.0) / 2.0, min=0.0, max=1.0)
    im = x[0].cpu().numpy()

    plt.imshow(im, cmap='gray')
    plt.savefig(f'plots/F->L.png')
    end_time = time.time() - start
    print(f'F->L time: {end_time}')

    ########################
    ######## F->T ##########
    ########################
    start = time.time()
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
    end_time = time.time() - start

    ########################
    ######## L->T ##########
    ########################
    start = time.time()
    path2 = df2['path'].values
    path2 = path2[0]
    path2 = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/MIMIC/' + path2
    path2 = path2.replace('.dcm', '.tiff')
    im = tifffile.imread(path2)
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
    end_time = time.time() - start
    print(f'L->T time: {end_time}')

    ########################
    ######## T->L ##########
    ########################
    start = time.time()
    prompt = df2['report'].values
    prompt = prompt[0]
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
    plt.savefig(f'plots/T->L.png')
    end_time = time.time() - start

    ########################
    ######## L->F ##########
    ########################
    start = time.time()
    im = tifffile.imread(path2)
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
    plt.savefig(f'plots/L->F.png')
    end_time = time.time() - start

    ########################
    ######## F+L->T ########
    ########################
    start = time.time()
    im1 = tifffile.imread(path1)
    im2 = tifffile.imread(path2)

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
    end_time = time.time() - start
    print(f'F+L->T time: {end_time}')

    ########################
    ######## F+T->L ########
    ########################
    start = time.time()
    prompt = df1['report'].values
    prompt = prompt[0]
    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    im = tifffile.imread(path1)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
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
    plt.savefig(f'plots/F+T->L.png')
    end_time = time.time() - start
    print(f'F+T->L time: {end_time}')

    ########################
    ######## L+T->F ########
    ########################
    start = time.time()
    prompt = df2['report'].values
    prompt = prompt[0]
    ctx = inference_tester.net.clip_encode_text(1 * [prompt], encode_type='encode_text').to(device)
    utx = None
    scale = 7.5
    conditioning = []
    if scale != 1.0:
        utx = inference_tester.net.clip_encode_text(1 * [""], encode_type='encode_text').to(device)
    conditioning.append(torch.cat([utx, ctx]))

    im = tifffile.imread(path2)
    im = torch.tensor(im, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
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
    plt.savefig(f'plots/L+T->F.png')
    end_time = time.time() - start
    print(f'L+T->F time: {end_time}')

    ########################
    ######## T->F+L ########
    ########################
    start = time.time()
    prompt = df1['report'].values
    prompt = prompt[0]
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
    plt.savefig(f'plots/T->F+L.png')

    end_time = time.time() - start
    print(f'T->F+L time: {end_time}')

    ########################
    ######## F->T+L ########
    ########################
    start = time.time()
    im = tifffile.imread(path1)
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

    # adesso la passiamo al decoder
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
    plt.savefig(f'plots/F->T+L.png')
    end_time = time.time() - start
    print(f'F->T+L time: {end_time}')

    ########################
    ######## L->T+F ########
    ########################
    start = time.time()
    im = tifffile.imread(path2)
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
    plt.savefig(f'plots/L->T+F.png')
    end_time = time.time() - start
    print(f'L->T+F time: {end_time}')
    """

if __name__ == "__main__":
    run_demo()













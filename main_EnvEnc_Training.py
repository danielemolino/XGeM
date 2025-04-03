import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf

from Clip_Training.utils import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from Clip_Training.utils import set_seed, mkdir, setup_logger, load_config_file
from EnvEnc_Training.EnvEnc_Training import train

from core.models.MedCoDi_M_wrapper import MedCoDi_M_wrapper
from DataLoader import MIMIC_CXR_Dataset, MultiPromptGenerator

from torch.optim import Adam, AdamW  # both are same but AdamW has a default weight decay

import argparse

TRAINER_CONFIG_PATH = 'EnvEnc_Training/envenc_train_config.yaml'


def main():
    config = load_config_file(TRAINER_CONFIG_PATH)

    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    filename = f"envenc_training_logs_{config.name}.txt"
    logger = setup_logger("ENVENC TRAINING", config.logs, 0, filename=filename)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.device
    config.n_gpu = torch.cuda.device_count()  # config.n_gpu
    set_seed(seed=11, n_gpu=config.n_gpu)

    # Load the model (Questa volta usiamo il mio modello perchè ho bisogno di avere frontal lateral e text
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_video_diffuser_8frames.pth']
    inference_tester = MedCoDi_M_wrapper(model='MedCoDi_M', data_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/checkpoints', pth=model_load_paths, load_weights=True)  # turn on fp16=True if loading fp16 weights

    codi = inference_tester.net
    del inference_tester

    optimus_weights = '/mimer/NOBACKUP/groups/naiss2023-6-336/dmolino/CoDI-Original/Report_Training/saved_checkpoints/VAE/checkpoint_99_epoch_VAE-Training-Prova1.pt'
    optimus_weights = torch.load(optimus_weights, map_location='cpu')
    a, b = codi.optimus.load_state_dict(optimus_weights, strict=False)
    """
    frontal_weights = torch.load(config.frontal_weights, map_location='cpu')
    for key in list(frontal_weights.keys()):  # Utilizza list per creare una copia delle chiavi
        if 'unet_image' in key:
            value = frontal_weights.pop(key)
            new_key = key.replace('unet_image', 'unet_frontal')
            frontal_weights[new_key] = value
    """

    lateral_weights = torch.load(config.lateral_weights, map_location='cpu')
    for key in list(lateral_weights.keys()):  # Utilizza list per creare una copia delle chiavi
        if 'unet_image' in key:
            value = lateral_weights.pop(key)
            new_key = key.replace('unet_image', 'unet_lateral')
            lateral_weights[new_key] = value
    # a, b = codi.model.load_state_dict(frontal_weights, strict=False)
    a, b = codi.model.load_state_dict(lateral_weights, strict=False)

    # Non li carico solo perchè ora non vanno trainati
    text_weights = torch.load(config.text_weights, map_location='cpu')
    a, b = codi.model.load_state_dict(text_weights, strict=False)

    if config.load_envenc:
        # Vanno caricati anche i pesi dell'environmental encoder della lateral, in quanto voglio fare F->T+L
        # Abbiamo caricato i diffusivi, ma adesso vanno caricati gli environmental encoders e i cross-attention layers
        blocks_to_load = [  # inference_tester.net.model.diffusion_model.unet_frontal.connecters_out,
            # inference_tester.net.model.diffusion_model.unet_text.connecters_out,
            codi.model.diffusion_model.unet_lateral.connecters_out,
            # inference_tester.net.model.diffusion_model.unet_frontal.input_block_connecters_in,
            # inference_tester.net.model.diffusion_model.unet_text.input_block_connecters_in,
            codi.model.diffusion_model.unet_lateral.input_block_connecters_in,
            # inference_tester.net.model.diffusion_model.unet_frontal.output_block_connecters_in,
            # inference_tester.net.model.diffusion_model.unet_text.output_block_connecters_in,
            codi.model.diffusion_model.unet_lateral.output_block_connecters_in]

        string_to_load = [  # 'model.model.diffusion_model.unet_frontal.connecters_out',
            # 'model.model.diffusion_model.unet_text.connecters_out',
            'model.model.diffusion_model.unet_lateral.connecters_out',
            # 'model.model.diffusion_model.unet_frontal.input_block_connecters_in',
            # 'model.model.diffusion_model.unet_text.input_block_connecters_in',
            'model.model.diffusion_model.unet_lateral.input_block_connecters_in',
            # 'model.model.diffusion_model.unet_frontal.output_block_connecters_in',
            # 'model.model.diffusion_model.unet_text.output_block_connecters_in',
            'model.model.diffusion_model.unet_lateral.output_block_connecters_in']

        path = '/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/EnvEnc_Training/saved_checkpoints/Training-EnvEnc-F+L+Contrastive-Loss/50'

        for string, block in zip(string_to_load, blocks_to_load):
            block_weights = f'{path}/{string}.pt'
            block_weights = torch.load(block_weights, map_location='cpu')
            a, b = block.load_state_dict(block_weights, strict=False)

    logger.info(f"Training/evaluation parameters {config}")

    # Load the dataloader
    path_to_csv = config.dataset
    csv = pd.read_csv(path_to_csv)
    other_view = 'lateral' if config.view == 'frontal' else 'frontal'
    view_checker = pd.read_csv('csv/train_frontal_lateral.csv').set_index('study_id')
    # Eliminiamo da csv tutte le righe che hanno view_target.lower() 0
    csv = csv[csv['study_id'].isin(view_checker[view_checker[config.view] == 1].index)]
    # Teniamo tutte le righe che hanno other_view > 0
    csv = csv[csv['study_id'].isin(view_checker[view_checker[other_view] > 0].index)]

    if config.text_emb:
        text_embeddings = np.load('embeddings/text_embeddings.npy')
    else:
        text_embeddings = None
    if config.frontal_emb:
        frontal_embeddings = np.load('embeddings/frontal_embeddings.npy')
    else:
        frontal_embeddings = None
    if config.lateral_emb:
        lateral_embeddings = np.load('embeddings/lateral_embeddings.npy')
    else:
        lateral_embeddings = None
    dataset = MultiPromptGenerator(csv, root_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/MIMIC/', view=config.view, text_embeddings=text_embeddings, frontal_embeddings=frontal_embeddings, lateral_embeddings=lateral_embeddings, train=True)
    dataloader = DataLoader(dataset, batch_size=config.per_gpu_train_batch_size, shuffle=True)

    # Now training
    config.checkpoint_dir = os.path.join(config.saved_checkpoints, config.name)
    mkdir(config.checkpoint_dir)
    global_step, avg_loss = train(config, dataloader, codi, logger, config.text_emb)  # save model every this epochs

    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)


if __name__ == "__main__":
    main()

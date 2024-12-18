import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf

from Clip_Training.utils import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from Clip_Training.utils import set_seed, mkdir, setup_logger, load_config_file
from CXR_Training.CXR_Diffusion_Training import train

from core.models.MedCoDi_M_wrapper import MedCoDi_M_wrapper
from DataLoader import MIMIC_CXR_Dataset, MultiPrompt_MIMIC_CXR_Dataset

from torch.optim import Adam, AdamW  # both are same but AdamW has a default weight decay

import argparse

TRAINER_CONFIG_PATH = 'CXR_Training/cxr_train_config.yaml'


def main():
    config = load_config_file(TRAINER_CONFIG_PATH)

    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    filename = f"cxr_training_logs_{config.name}.txt"
    logger = setup_logger("CXR TRAINING", config.logs, 0, filename=filename)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.device
    config.n_gpu = torch.cuda.device_count()  # config.n_gpu
    set_seed(seed=11, n_gpu=config.n_gpu)

    # Load the model
    model_load_paths = ['CoDi_encoders.pth', 'CoDi_video_diffuser_8frames.pth']
    inference_tester = MedCoDi_M_wrapper(model='MedCoDi_M', load_weights=True, data_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/checkpoints', pth=model_load_paths)
    codi = inference_tester.net
    codi.optimus = None
    codi.clip.load_state_dict(torch.load(config.clip_weights, map_location=device))
    del inference_tester

    logger.info(f"Training/evaluation parameters {config}")

    # Load the dataloader
    path_to_csv = config.dataset
    csv = pd.read_csv(path_to_csv)
    if not config.multi_prompt:
        dataset = MIMIC_CXR_Dataset(csv, root_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/MIMIC/')
    else:
        other_view = 'frontal' if config.view == 'lateral' else 'lateral'
        text_embeddings = np.load('embeddings/text_embeddings.npy')
        image_embeddings = np.load(f'embeddings/{config.view}_embeddings.npy')
        dataset = MultiPrompt_MIMIC_CXR_Dataset(csv, root_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/MIMIC/', view=config.view, text_embeddings=text_embeddings, image_embeddings=image_embeddings)
    dataloader = DataLoader(dataset, batch_size=config.per_gpu_train_batch_size, shuffle=True)

    # Now training
    config.checkpoint_dir = os.path.join(config.saved_checkpoints, config.name)
    mkdir(config.checkpoint_dir)
    global_step, avg_loss = train(config, dataloader, codi, logger)  # save model every this epochs
    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)


if __name__ == "__main__":
    main()

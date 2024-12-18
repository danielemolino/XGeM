import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf

from Clip_Training.utils import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from Clip_Training.utils import set_seed, mkdir, setup_logger, load_config_file

from DataLoader import VAE_Dataset

from core.models.MedCoDi_M_wrapper import MedCoDi_M_wrapper
from torch.optim import Adam, AdamW  # both are same but AdamW has a default weight decay

from Report_Training.VAE_Training import train

import argparse

TRAINER_CONFIG_PATH = 'Report_Training/vae_train_config.yaml'


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
    inference_tester = MedCoDi_M_wrapper(model='MedCoDi_M', load_weights=True, data_dir='/mimer/NOBACKUP/groups/snic2022-5-277/dmolino/checkpoints', pth=model_load_paths, fp16=False)
    codi = inference_tester.net
    optimus = codi.optimus
    del inference_tester, codi

    logger.info(f"Training/evaluation parameters {config}")

    # Load the dataloader
    path_to_csv = config.dataset
    csv = pd.read_csv(path_to_csv)
    dataset = VAE_Dataset(csv)
    dataloader = DataLoader(dataset, batch_size=config.per_gpu_train_batch_size, shuffle=True, num_workers=config.num_workers)

    # Now training
    # creiamo la cartella per i checkpoint
    config.checkpoint_dir = os.path.join(config.saved_checkpoints, config.name)
    mkdir(config.checkpoint_dir)
    global_step, avg_loss = train(config, dataloader, optimus, logger)  # save model every this epochs

    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)


if __name__ == "__main__":
    main()

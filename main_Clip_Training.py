import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW

import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf

from Clip_Training.utils import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from Clip_Training.utils import set_seed, mkdir, setup_logger, load_config_file
from Clip_Training.Clip_Training_Script import train

from DataLoader import MIMIC_CXR_Dataset

import argparse
import json

from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model

import monai
from monai.data import DataLoader, partition_dataset
from monai.transforms import Compose
from monai.utils import first

import torch.distributed as dist


TRAINER_CONFIG_PATH = 'Clip_Training/clip_train_config.yaml'

def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    filenames_train = json_data["training"]
    return [_item["image"] for _item in filenames_train]

def prepare_data(
    train_files: list, device: torch.device, cache_rate: float, num_workers: int = 2, batch_size: int = 1
) -> DataLoader:
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.

    Returns:
        DataLoader: Data loader for training.
    """

    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    def _load_npy(path):
        return torch.tensor(np.load(path), dtype=torch.float32)

    # Carica il CSV in un dizionario per un accesso rapido
    reports = pd.read_csv('/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/CT_RATE_preprocessed_v2/train_reports.csv')
    volume_text_mapping = {
        row["VolumeName"]: f"Findings: {row['Findings_EN']} Impression: {row['Impressions_EN']}"
        for _, row in reports.iterrows()
    }   

    # Funzione per ottenere il testo corrispondente
    def lookup_text(volume_name):
        return volume_text_mapping.get(volume_name, "")

    train_transforms = Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(keys="impression", func=lookup_text),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )
    return DataLoader(train_ds, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    config = load_config_file(TRAINER_CONFIG_PATH)

    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    filename = f"clip_training_logs_{config.name}.txt"
    logger = setup_logger("CLIP TRAINING", config.logs, 0, filename=filename)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = config.device
    config.n_gpu = torch.cuda.device_count()  # config.n_gpu
    set_seed(seed=11, n_gpu=config.n_gpu)

    # Load the model
    model_load_paths = ['CoDi_encoders.pth']
    cfgm = model_cfg_bank()(config.clip_model)
    clip = get_model()(cfgm)

    # Se allungo la sequenza non posso fare transfer learning dai pesi di MedCoDi
    """
    data_dir='/mimer/NOBACKUP/groups/naiss2023-6-336/dmolino/MedCoDi-M/Weights/Starting_checkpoints'
    pth = ['CoDi_encoders.pth']
    for path in pth:
        weight = torch.load(os.path.join(data_dir, path), map_location='cpu')
        # Supponiamo che state_dict sia già caricato
        weight = {chiave.split("model.text_model.", 1)[1]: valore
            for chiave, valore in weight.items()
                if "model.text_model" in chiave
        }
        clip.model.text_model.load_state_dict(weight, strict=True)
    """
    
    clip = clip.to(config.device)

    logger.info(f"Training/evaluation parameters {config}")

    # Load the dataloader
    if config.dataset == 'MIMIC-CXR':
        path_to_csv = config.csv
        csv = pd.read_csv(path_to_csv)
        # Puliamo i report da \n e tutto ciò che non è testo
        csv['report'] = csv['report'].str.replace('\n', ' ')
        csv['report'] = csv['report'].str.replace('[^\w\s]', '')
        csv['report'] = csv['report'].str.replace('  ', ' ')
        csv['report'] = csv['report'].str.lower()
        csv['report'] = csv['report'].str.strip()

        dataset = MIMIC_CXR_Dataset(csv, '/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/MIMIC', report_type=config.report_type)
        dataloader = DataLoader(dataset, batch_size=config.per_gpu_train_batch_size, shuffle=True)

    elif config.dataset == 'CT-RATE':
        filenames_train = load_filenames("/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/CT_RATE_preprocessed_v2/data_volumes_correct.json")
        train_files = []
        for _i in range(len(filenames_train)):
            # str_img = os.path.join(args.embedding_base_dir, filenames_train[_i])
            str_img = filenames_train[_i]
            if not os.path.exists(str_img):
                continue
            str_info = filenames_train[_i].split('/')[-1]
            train_files.append(
                {"image": str_img, "impression": str_info}
            )
        dataloader = prepare_data(
            train_files, device, cache_rate=0, batch_size=config.per_gpu_train_batch_size, num_workers=config.num_workers
        )

    # Now training
    # creiamo la cartella per i checkpoint
    config.checkpoint_dir = os.path.join(config.saved_checkpoints, config.name)
    mkdir(config.checkpoint_dir)
    global_step, avg_loss = train(config, dataloader, clip, logger)  # save model every this epochs

    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)


if __name__ == "__main__":
    main()

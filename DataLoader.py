import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import tifffile


class MIMIC_CXR_Dataset(Dataset):
    def __init__(self, csv, root_dir):
        self.csv = csv
        self.root_dir = root_dir

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        path = row['path'][:-4] + '.tiff'
        image_path = os.path.join(self.root_dir, path)
        image = tifffile.imread(image_path)
        text = row['report']
        subject_id = np.array(row['subject_id'])
        study_id = np.array(row['study_id'])
        path = row['path']
        return image, text, subject_id, study_id, path


# devo crearne un altro che serve per il training di Optimus
class VAE_Dataset(Dataset):
    def __init__(self, csv):
        self.csv = csv

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        # prendiamo il report
        text = row['report']
        return text


class MultiPrompt_MIMIC_CXR_Dataset(Dataset):
    def __init__(self, csv, root_dir, view, text_embeddings=None, image_embeddings=None, weights=[1, 1], report_gen=False):
        self.csv = csv
        self.root_dir = root_dir
        self.view = view
        self.other_view = 'frontal' if view == 'lateral' else 'lateral'
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings
        self.weights = weights
        self.view_checker = pd.read_csv('csv/train_frontal_lateral.csv').set_index('study_id')
        self.report_gen = report_gen

        if self.report_gen:
            self.other_image_embeddings = self.image_embeddings
            self.image_embeddings = np.load(f'embeddings/{self.other_view}_embeddings.npy')

    def __len__(self):
        return len(self.csv)

    def random_sample(self, study_id):
        prompt_options = {1: self.get_image_embeddings, 2: self.get_text_embeddings, 3: self.interpolate_embeddings}
        prompt, prompt_type = prompt_options[np.random.choice([1, 2, 3])](study_id)
        return prompt, prompt_type

    def get_image_embeddings(self, study_id):
        other_image_embeddings = self.image_embeddings[self.image_embeddings['study_id'] == study_id]
        # prendiamo un elemento a caso
        other_image_embedding = other_image_embeddings[np.random.randint(len(other_image_embeddings))]
        prompt = other_image_embedding['embedding']
        # Va inserita ora anche la probabilità dell'Unconditional Free Guidance, mettiamola al 10%
        p = np.random.rand()
        return prompt, 'image'

    def get_text_embeddings(self, study_id):
        if self.report_gen:
            image_embeddings = self.other_image_embeddings[self.other_image_embeddings['study_id'] == study_id]
            image_embeddings = image_embeddings[np.random.randint(len(image_embeddings))]
            prompt = image_embeddings['embedding']
            return prompt, 'image'
        else:
            return self.text_embeddings[np.where(self.text_embeddings['study_id'] == study_id)][0]['embedding'], 'text'

    def interpolate_embeddings(self, study_id):
        embeddings = [self.get_image_embeddings(study_id)[0], self.get_text_embeddings(study_id)[0]]
        norm_weights = self.weights / np.sum(self.weights)
        prompt = 0.0
        for i in range(len(embeddings)):
            prompt += embeddings[i] * norm_weights[i]
        return prompt, 'both'

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        path = row['path'][:-4] + '.tiff'
        study_id = row['study_id']

        if self.report_gen:
            diffusion_variable = row['report']
        else:
            image_path = os.path.join(self.root_dir, path)
            diffusion_variable = tifffile.imread(image_path)

        # vediamo se df['study_id'] ha un other_view maggiore di zero
        other_view = int(self.view_checker.loc[study_id][self.other_view])
        sample_options = {True: self.random_sample, False: self.get_text_embeddings}
        prompt, prompt_type = sample_options[other_view > 0](study_id)
        return diffusion_variable, prompt, prompt_type


class MultiPromptGenerator(Dataset):
    def __init__(self, csv, root_dir, view, text_embeddings=None, frontal_embeddings=None, lateral_embeddings=None, train=False):
        self.csv = csv
        self.root_dir = root_dir
        self.view = view
        self.other_view = 'frontal' if view == 'lateral' else 'lateral'
        if train:
            self.other_csv = pd.read_csv(f'csv/train_short_{self.other_view}_clean.csv')
        else:
            self.other_csv = pd.read_csv(f'csv/test_short_{self.other_view}_clean.csv')

        self.text_embeddings = text_embeddings
        self.frontal_embeddings = frontal_embeddings
        self.lateral_embeddings = lateral_embeddings

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        path = row['path'][:-4] + '.tiff'
        image_path = os.path.join(self.root_dir, path)
        image = tifffile.imread(image_path)
        text = row['report']
        subject_id = np.array(row['subject_id'])
        study_id = np.array(row['study_id'])
        dicom_id = row['dicom_id']
        path = row['path']
        # Carichiamo anche un'immagine dell'altra vista
        other_row = self.other_csv.loc[self.other_csv.study_id == study_id]
        # Potrebbero esserne più di una, prendiamone una a caso
        other_row = other_row.iloc[np.random.randint(len(other_row))]
        other_path = other_row['path'][:-4] + '.tiff'
        other_image_path = os.path.join(self.root_dir, other_path)
        other_image = tifffile.imread(other_image_path)
        if self.text_embeddings is not None:
            return image, text, other_image, path, study_id, subject_id, self.text_embeddings[np.where(self.text_embeddings['study_id'] == study_id)][0]['embedding']
        elif self.frontal_embeddings is not None:
            return image, text, other_image, path, study_id, subject_id, self.frontal_embeddings[np.where(self.frontal_embeddings['dicom_id'] == dicom_id)][0]['embedding']
        elif self.lateral_embeddings is not None:
            return image, text, other_image, path, study_id, subject_id, self.lateral_embeddings[np.where(self.lateral_embeddings['dicom_id'] == dicom_id)][0]['embedding']
        return image, text, other_image, path, study_id, subject_id, other_path

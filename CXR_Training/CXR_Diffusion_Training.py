import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from omegaconf import OmegaConf
from Clip_Training.utils import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from Clip_Training.utils import set_seed, mkdir, setup_logger, load_config_file
from DataLoader import MIMIC_CXR_Dataset
from torch.optim import Adam, AdamW  # both are same but AdamW has a default weight decay
import argparse

empty_embeddings = np.load('embeddings/empty_embeddings.npy')
empty_embeddings_options = {'text': torch.Tensor(empty_embeddings[0][1]).unsqueeze(0), 'image': torch.Tensor(empty_embeddings[1][1]).unsqueeze(0), 'both': torch.Tensor(empty_embeddings[2][1]).unsqueeze(0)}


def train(config, dataloader, model, logger, xtype='frontal'):
    # Trains the model
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = dataloader

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs

    optimizer = AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps,
                      weight_decay=config.optimizer.params.weight_decay)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(torch.device(config.device))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)

    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if scheduler:
        logger.info("  warmup steps = %d", num_warmup_steps)

    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()

    for epoch in range(int(config.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            if not config.multi_prompt:
                input_images, input_texts, _, _, _= batch
                input_images = input_images.unsqueeze(1).float()
                input_images = input_images * 2 - 1
                input_texts = list(input_texts)
                # Come primo step, bisogna fare encoding dell'immagine e del testo
                if config.n_gpu == 1:
                    image_features = model.autokl_encode(input_images.to(config.device))
                    text_features = model.clip_encode_text(input_texts).to(config.device)
                    # passiamo un numero di testi vuoti pari al batch size
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat([empty_embeddings_options['text'].to(config.device) if pi < 0.1 else x for pi, x in zip(p, text_features)], dim=0).to(config.device)
                elif config.n_gpu > 1:
                    image_features = model.module.autokl_encode(input_images.to(config.device))
                    text_features = model.module.clip_encode_text(input_texts)
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat([empty_embeddings_options['text'].to(config.device) if pi < 0.1 else x for pi, x in zip(p, text_features)], dim=0).to(config.device)
                else:
                    image_features = model.autokl_encode(input_images.to(config.device))
                    text_features = model.clip_encode_text(input_texts)
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat([empty_embeddings_options['text'].to(config.device) if pi < 0.1 else x for pi, x in zip(p, text_features)], dim=0).to(config.device)
                text_features = text_features.unsqueeze(1)
                print(text_features.shape)
            else:
                input_images, text_features, prompt_type = batch
                text_features = text_features.unsqueeze(1).to(config.device)
                input_images = input_images.unsqueeze(1).float().to(config.device)
                input_images = input_images * 2 - 1
                if config.n_gpu == 1:
                    image_features = model.autokl_encode(input_images.to(config.device))
                    # Per simulare la classifier free guidance, peschiamo n_batch numeri random tra 0 e 1, per tutti i valori minori di 0.1, sostituiamo con l'empty_embedding corrispondente
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat([empty_embeddings_options[p_type].to(config.device) if pi < 0.1 else x.to(config.device) for p_type, pi, x in zip(prompt_type, p, text_features)], dim=0).to(config.device)
                elif config.n_gpu > 1:
                    image_features = model.module.autokl_encode(input_images.to(config.device))
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat([empty_embeddings_options[p_type].to(config.device) if pi < 0.1 else x.to(config.device) for p_type, pi, x in zip(prompt_type, p, text_features)], dim=0).to(config.device)
                else:
                    image_features = model.autokl_encode(input_images.to(config.device))
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat([empty_embeddings_options[p_type].to(config.device) if pi < 0.1 else x.to(config.device) for p_type, pi, x in zip(prompt_type, p, text_features)], dim=0).to(config.device)
                text_features = text_features.unsqueeze(1)

            # Per simulare una classifier free guidance, campioniamo un valore tra 0 e 1, se è minore di 0.1, allora
            # sostituiamo al testo un testo vuoto

            mix_weight = {'frontal': 1, 'lateral': 1, 'text': 1}

            # A questo punto, ho il latent space dell'immagine e l'embedding del testo
            # Ora, bisogna fare il forward del processo diffusivo, che consiste in:
            # 1. Campionare il timestep t da una distribuzione uniforme tra 0 e il numero di timesteps (solitamente 1000)
            # 2. Campionare il rumore z da una distribuzione normale e applicarlo all'immagine
            # se text_features ha 4 dimensioni, facciamo un unsqueeze
            if len(text_features.shape) == 4:
                text_features = text_features.squeeze(1)
            if config.n_gpu == 1:
                loss = model(x=[image_features], c=[text_features],  xtype=[xtype], ctype=['text'])
            elif config.n_gpu > 1:
                loss = model.module(x=[image_features], c=[text_features], xtype=[xtype], ctype=['text'])
            else:
                loss = model(x=[image_features], c=[text_features], xtype=[xtype], ctype=['text'])

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()  # PYTORCH 1.x : call optimizer.step() first then scheduler.step()

                if scheduler:
                    scheduler.step()

                model.zero_grad()

                if global_step % config.logging_steps == 0:
                    logger.info(
                        "Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step,
                                                                                               optimizer.param_groups[
                                                                                                   0]["lr"],
                                                                                               loss.item(),
                                                                                               global_loss / global_step)
                    )
        # Save checkpoint after each epoch
        if config.save_steps_epochs > 0 and epoch % config.save_steps_epochs == 0:
            # salviamo solo i pesi del modello
            if config.n_gpu > 1:
                torch.save(model.module.model.state_dict(),
                           os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
            else:
                torch.save(model.model.state_dict(),
                           os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_epoch_{config.name}.pt'))

    if config.save_steps_epochs > 0:
        if config.n_gpu > 1:
            torch.save(model.module.model.state_dict(),
                       os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
        else:
            torch.save(model.model.state_dict(),
                       os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
    return global_step, global_loss / global_step

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
empty_embeddings_options = {'text': torch.Tensor(empty_embeddings[0][1]).unsqueeze(0),
                            'image': torch.Tensor(empty_embeddings[1][1]).unsqueeze(0),
                            'both': torch.Tensor(empty_embeddings[2][1]).unsqueeze(0)}


def train(config, dataloader, model, logger, text_emb=False):
    # Trains the model
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = dataloader

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs

    blocks_to_update = [# model.model.diffusion_model.unet_frontal.connecters_out,
                        model.model.diffusion_model.unet_text.connecters_out,
                        # model.model.diffusion_model.unet_lateral.connecters_out,
                        # model.model.diffusion_model.unet_frontal.input_block_connecters_in,
                        model.model.diffusion_model.unet_text.input_block_connecters_in,
                        # model.model.diffusion_model.unet_lateral.input_block_connecters_in,
                        # model.model.diffusion_model.unet_frontal.output_block_connecters_in,
                        model.model.diffusion_model.unet_text.output_block_connecters_in]
                        # model.model.diffusion_model.unet_lateral.output_block_connecters_in]

    string_blocks = [# 'model.model.diffusion_model.unet_frontal.connecters_out',
                     'model.model.diffusion_model.unet_text.connecters_out',
                     # 'model.model.diffusion_model.unet_lateral.connecters_out',
                     # 'model.model.diffusion_model.unet_frontal.input_block_connecters_in',
                     'model.model.diffusion_model.unet_text.input_block_connecters_in',
                     # 'model.model.diffusion_model.unet_lateral.input_block_connecters_in',
                     # 'model.model.diffusion_model.unet_frontal.output_block_connecters_in',
                     'model.model.diffusion_model.unet_text.output_block_connecters_in']
                     # 'model.model.diffusion_model.unet_lateral.output_block_connecters_in']

    model.model.diffusion_model.freeze()
    model.model.diffusion_model.unfreeze(blocks_to_update)

    # Funzione per ottenere i parametri da una lista di moduli
    def get_parameters(modules):
        for module in modules:
            for param in module.parameters():
                yield param

    optimizer = AdamW(get_parameters(blocks_to_update), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps,
                      weight_decay=config.optimizer.params.weight_decay)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        blocks_to_update = [# model.module.model.diffusion_model.unet_frontal.connecters_out,
                            model.module.model.diffusion_model.unet_text.connecters_out,
                            # model.module.model.diffusion_model.unet_lateral.connecters_out,
                            # model.module.model.diffusion_model.unet_frontal.input_block_connecters_in,
                            model.module.model.diffusion_model.unet_text.input_block_connecters_in,
                            # model.module.model.diffusion_model.unet_lateral.input_block_connecters_in,
                            # model.module.model.diffusion_model.unet_frontal.output_block_connecters_in,
                            model.module.model.diffusion_model.unet_text.output_block_connecters_in]
                            # model.module.model.diffusion_model.unet_lateral.output_block_connecters_in]

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
            input_images, input_texts, input_images2, _, _, _, prompt = batch
            input_images = input_images.unsqueeze(1).float()
            input_images = input_images * 2 - 1
            input_images2 = input_images2.unsqueeze(1).float()
            input_images2 = input_images2 * 2 - 1
            input_texts = list(input_texts)
            if text_emb:
                # Come primo step, bisogna fare encoding dell'immagine e del testo
                if config.n_gpu == 1:
                    image_features = model.autokl_encode(input_images.to(config.device))
                    image_features2 = model.autokl_encode(input_images2.to(config.device))
                    text_features = prompt.to(config.device)
                    # passiamo un numero di testi vuoti pari al batch size
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat(
                        [empty_embeddings_options['text'].to(config.device) if pi < 0.1 else x.unsqueeze(0) for pi, x in
                         zip(p, text_features)], dim=0).to(config.device)
                elif config.n_gpu > 1:
                    image_features = model.module.autokl_encode(input_images.to(config.device))
                    image_features2 = model.module.autokl_encode(input_images2.to(config.device))
                    text_features = prompt.to(config.device)
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat(
                        [empty_embeddings_options['text'].to(config.device) if pi < 0.1 else x.unsqueeze(0) for pi, x in
                         zip(p, text_features)], dim=0).to(config.device)
                else:
                    image_features = model.autokl_encode(input_images.to(config.device))
                    image_features2 = model.autokl_encode(input_images2.to(config.device))
                    text_features = prompt.to(config.device)
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    text_features = torch.cat(
                        [empty_embeddings_options['text'].to(config.device) if pi < 0.1 else x.unsqueeze(0) for pi, x in
                         zip(p, text_features)], dim=0).to(config.device)
                text_features = text_features.unsqueeze(1)
            else:
                del input_images
                if config.n_gpu == 1:
                    image_features = prompt.to(config.device)
                    image_features2 = model.autokl_encode(input_images2.to(config.device))
                    text_features = model.optimus_encode(input_texts)
                    # passiamo un numero di testi vuoti pari al batch size
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    image_features = torch.cat(
                        [empty_embeddings_options['image'].to(config.device) if pi < 0.1 else x.unsqueeze(0) for pi, x in
                         zip(p, image_features)], dim=0).to(config.device)
                elif config.n_gpu > 1:
                    image_features = prompt.to(config.device)
                    image_features2 = model.module.autokl_encode(input_images2.to(config.device))
                    text_features = model.module.optimus_encode(input_texts)
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    image_features = torch.cat(
                        [empty_embeddings_options['image'].to(config.device) if pi < 0.1 else x.unsqueeze(0) for pi, x in
                         zip(p, image_features)], dim=0).to(config.device)
                else:
                    image_features = prompt.to(config.device)
                    image_features2 = model.autokl_encode(input_images2.to(config.device))
                    text_features = model.optimus_encode(input_texts)
                    p = [np.random.rand() for _ in range(config.per_gpu_train_batch_size)]
                    image_features = torch.cat(
                        [empty_embeddings_options['image'].to(config.device) if pi < 0.1 else x.unsqueeze(0) for pi, x in
                         zip(p, image_features)], dim=0).to(config.device)
                image_features = image_features.unsqueeze(1)

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
            if text_emb:
                if config.n_gpu == 1:
                    loss = model(x=[image_features, image_features2], c=[text_features], xtype=['frontal', 'lateral'],
                                 ctype=['text'], env_enc=True)
                elif config.n_gpu > 1:
                    loss = model.module(x=[image_features, image_features2], c=[text_features],
                                        xtype=['frontal', 'lateral'], ctype=['text'], env_enc=True)
                else:
                    loss = model(x=[image_features, image_features2], c=[text_features], xtype=['frontal', 'lateral'],
                                 ctype=['text'], env_enc=True)
            else:
                if config.n_gpu == 1:
                    loss = model(x=[text_features, image_features2], c=[image_features], xtype=['text', 'lateral'],
                                 ctype=['frontal'], env_enc=True)
                elif config.n_gpu > 1:
                    loss = model.module(x=[text_features, image_features2], c=[image_features],
                                        xtype=['text', 'lateral'], ctype=['frontal'], env_enc=True)
                else:
                    loss = model(x=[text_features, image_features2], c=[image_features], xtype=['text', 'lateral'],
                                 ctype=['frontal'], env_enc=True)

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
            # Dato che il modello è molto pesante, salviamo solo i pesi dei layer unfreezed che sono contenuti in blocks_to_update
            mkdir(os.path.join(config.checkpoint_dir, str(epoch)))
            # Ora, per ogni blocco da aggiornare, salviamo i pesi
            for block, string in zip(blocks_to_update, string_blocks):
                # Block è un modulo di pytorch, ma per usarlo nel path1 del file, dobbiamo convertirlo in stringa
                torch.save(block.state_dict(), os.path.join(config.checkpoint_dir, str(epoch), f'{string}.pt'))

    if config.save_steps_epochs > 0:
        # Creiamo una cartella con il numero dell'epoch
        mkdir(os.path.join(config.checkpoint_dir, str(epoch)))
        # Ora, per ogni blocco da aggiornare, salviamo i pesi
        for block, string in zip(blocks_to_update, string_blocks):
            torch.save(block.state_dict(), os.path.join(config.checkpoint_dir, str(epoch), f'{string}.pt'))
    return global_step, global_loss / global_step

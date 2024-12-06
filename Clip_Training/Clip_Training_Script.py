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
from torch.optim import Adam, AdamW
import argparse


def train(config, dataloader, model, logger):
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = dataloader

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps \
        * config.num_train_epochs

    # blocks_to_freeze = [model.model.text_model, model.model.text_projection]
    # model.freeze(blocks_to_freeze)

    optimizer = AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps,
                      weight_decay=config.optimizer.params.weight_decay)

    # Warmup iterations = 20% of total iterations
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(torch.device(config.device))
    model.train()

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
            input_images, input_texts, _, _, _ = batch
            input_images = input_images.unsqueeze(1)
            # Check the model of the device to handle correctly the tokenization
            if config.n_gpu == 1:
                input_texts = model.tokenizer(input_texts, truncation=True, max_length=model.max_length, return_length=True,
                                              return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            elif config.n_gpu > 1:
                input_texts = model.module.tokenizer(input_texts, truncation=True, max_length=model.module.max_length, return_length=True,
                                                     return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
            else:
                input_texts = model.tokenizer(input_texts, truncation=True, max_length=model.max_length, return_length=True,
                                              return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

            input_texts = input_texts["input_ids"]
            image_features = model(input_images, 'encode_vision')
            text_features = model(input_texts, 'encode_text')

            # normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            if config.n_gpu == 1:
                logit_scale = model.model.logit_scale.exp()
            elif config.n_gpu > 1:
                logit_scale = model.module.model.logit_scale.exp()
            else:
                logit_scale = model.model.logit_scale.exp()

            logits_per_image = logit_scale * image_features.squeeze() @ text_features.squeeze().t()
            logits_per_text = logit_scale * text_features.squeeze() @ image_features.squeeze().t()

            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_per_text, labels)

            loss = (image_loss + text_loss) / 2

            if config.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()
            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step()

                # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
                if config.n_gpu == 1:
                    model.model.logit_scale.data = torch.clamp(model.model.logit_scale.data, 0, 4.6052)
                elif config.n_gpu > 1:
                    model.module.model.logit_scale.data = torch.clamp(model.module.model.logit_scale.data, 0, 4.6052)
                else:
                    model.model.logit_scale.data = torch.clamp(model.model.logit_scale.data, 0, 4.6052)

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
        if config.save_steps_epochs>0 and epoch % config.save_steps_epochs == 0:
            if config.n_gpu > 1:
                torch.save(model.module.state_dict(), os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_epoch_{config.name}.pt'))

    if config.save_steps_epochs > 0:
        if config.n_gpu > 1:
            torch.save(model.module.state_dict(),
                       os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
        else:
            torch.save(model.state_dict(),
                       os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
    return global_step, global_loss / global_step

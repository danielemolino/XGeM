from pytorch_transformers import WarmupLinearSchedule
from tqdm import trange, tqdm
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


def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio_increase=0.5, ratio_zero=0.3):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio_increase)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            if i < period * ratio_zero:
                L[int(i + c * period)] = start
            else:
                L[int(i + c * period)] = v
                v += step
            i += 1
    return L


def train(config, dataloader, model, logger):
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    train_dataloader = dataloader
    t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
    """
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    """
    num_warmup_steps = int(0.20 * t_total)
    optimizer = AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps,
                      weight_decay=config.optimizer.params.weight_decay)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=t_total)

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

    n_iter = int(config.num_train_epochs) * len(train_dataloader)
    # Valori presi dall'esempio su GitHub
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0, n_cycle=1,
                                           ratio_increase=0.25, ratio_zero=0.5)

    tmp_list = []
    train_iterator = trange(int(config.num_train_epochs), desc="Epoch")

    if config.n_gpu == 1:
        tok0 = model.tokenizer_encoder
        tok1 = model.tokenizer_decoder
        pad_token_id = model.pad_token_id
    elif config.n_gpu > 1:
        tok0 = model.module.tokenizer_encoder
        tok1 = model.module.tokenizer_decoder
        pad_token_id = model.module.pad_token_id

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            text = batch
            # prendiao il testo minuscolo
            text = [t.lower() for t in text]

            token = [tok0.tokenize(t) for t in text]
            token_id = []
            for tokeni in token:
                token_sentence = [tok0._convert_token_to_id(i) for i in tokeni]
                token_sentence = tok0.add_special_tokens_single_sentence(token_sentence)
                token_id.append(torch.LongTensor(token_sentence))
            tokenized_text0 = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=0.0)[:, :512]

            token = [tok1.tokenize(t) for t in text]
            token_id = []
            for tokeni in token:
                token_sentence = [tok1._convert_token_to_id(i) for i in tokeni]
                token_sentence = tok1.add_special_tokens_single_sentence(token_sentence)
                token_id.append(torch.LongTensor(token_sentence))
            tokenized_text1 = torch._C._nn.pad_sequence(token_id, batch_first=True, padding_value=pad_token_id)[:, :512]

            inputs, labels = mask_tokens(tokenized_text0, encoder_tokenizer, args) if config.mlm else (tokenized_text0, tokenized_text1)
            labels = tokenized_text1

            tokenized_text1 = tokenized_text1.to(config.device)
            inputs = inputs.to(config.device)
            labels = labels.to(config.device)

            model.train()

            beta_t = beta_t_list[step + epoch * len(epoch_iterator)]
            if config.n_gpu == 1:
                model.args.beta = beta_t
            elif config.n_gpu > 1:
                model.module.args.beta = beta_t

            if beta_t == 0.0:
                if config.n_gpu == 1:
                    model.args.fb_mode = 0
                elif config.n_gpu > 1:
                    model.module.args.fb_mode = 0
            else:
                if config.n_gpu == 1:
                    model.args.fb_mode = 1
                elif config.n_gpu > 1:
                    model.module.args.fb_mode = 1

            if config.use_deterministic_connect:
                if config.n_gpu == 1:
                    model.args.fb_mode = 2
                elif config.n_gpu > 1:
                    model.module.args.fb_mode = 2

            if config.n_gpu == 1:
                loss_rec, loss_kl, loss = model(inputs, labels)
            elif config.n_gpu > 1:
                loss_rec, loss_kl, loss = model.module(inputs, labels)

            # Chunyuan: loss_rec size is [4], while latent_z size is [12]
            if config.n_gpu > 1:
                loss_rec = loss_rec.mean()  # mean() to average on multi-gpu parallel training
                loss_kl = loss_kl.mean()
                loss = loss.mean()

            if config.n_gpu == 1:
                epoch_iterator.set_description(
                    (
                        f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                        f'loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; '
                        f'beta: {model.args.beta:.3f}'
                    )
                )
            elif config.n_gpu > 1:
                epoch_iterator.set_description(
                    (
                        f'iter: {step + epoch * len(epoch_iterator)}; loss: {loss.item():.3f}; '
                        f'loss_rec: {loss_rec.item():.3f}; loss_kl: {loss_kl.item():.3f}; '
                        f'beta: {model.module.args.beta:.3f}'
                    )
                )

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            # printiamo la usage della GPU
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU Memory Usage: {info.used / 1024 ** 2} MB")
            nvidia_smi.nvmlShutdown()

            global_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()

                scheduler.step()  # Update learning rate schedule

                model.zero_grad()

                global_step += 1

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
                torch.save(model.module.state_dict(),
                           os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
            else:
                torch.save(model.state_dict(),
                           os.path.join(config.saved_checkpoints, f'checkpoint_{epoch}_epoch_{config.name}.pt'))

    if config.save_steps_epochs > 0:
        if config.n_gpu > 1:
            torch.save(model.module.state_dict(),
                       os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}_epoch_{config.name}.pt'))
        else:
            torch.save(model.state_dict(),
                       os.path.join(config.checkpoint_dir, f'checkpoint_{epoch}_epoch_{config.name}.pt'))

    return global_step, global_loss / global_step

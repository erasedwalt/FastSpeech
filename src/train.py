import torch
import numpy as np

import argparse
from tqdm import tqdm
import random
import math

from lib import Config, create_mask, calc_alignment_and_spec


SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

SPEC_FILL = math.log(1e-5)


def train(
    model,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    criterion,
    aligner,
    melspec,
    vocoder,
    device,
    exp_name,
    clip=10.,
    logger=None,
    log_interval=500,
    best_loss=10000.
):

    global_steps = 0
    model.train()
    aligner.eval()
    vocoder.eval()
    while True:
        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            waveform = batch[0].to(device)
            waveform_length = batch[1]
            tokens = batch[2].to(device)
            token_length = batch[3]
            transcript = batch[4]

            specs, spec_length, teacher_durations = calc_alignment_and_spec(
                melspec,
                aligner,
                waveform,
                waveform_length,
                token_length,
                transcript
            )
            specs = specs.to(device)
            spec_length = spec_length.to(device)
            teacher_durations = teacher_durations.to(device)

            spec_mask = create_mask(specs, spec_length).to(device)
            durations_mask = create_mask(tokens, token_length).to(device)

            pred_spec, pred_durations = model(tokens, teacher_durations,
                                              durations_mask[:, None, None, :],
                                              spec_mask[:, None, None, :])

            loss = criterion(pred_spec, specs, spec_mask,
                             pred_durations, teacher_durations, durations_mask)
            loss.backward()

            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if bool(torch.isnan(grad)):
                continue

            optimizer.step()
            global_steps += 1

            if scheduler is not None:
                lr = float(scheduler.get_lr()[0])
                scheduler.step()
            else:
                lr = float(optimizer.rate())

            to_log = {'Train Loss': loss.item(), 'Grad': grad.item(), 'LR': lr}
            if logger is not None:
                logger.log_metrics(to_log)
            else:
                print(to_log)
 
            if (global_steps + 1) % log_interval == 0:
                best_loss = evaluate(model, val_loader, criterion, aligner, melspec,
                                     vocoder, device, exp_name, logger, best_loss)
                model.train()


def evaluate(
    model,
    val_loader,
    criterion,
    aligner,
    melspec,
    vocoder,
    device,
    exp_name,
    logger=None,
    best_loss=10000.,
):
    model.eval()
    global_loss = 0.
    with torch.no_grad():
        for batch in tqdm(val_loader):
            optimizer.zero_grad()

            waveform = batch[0].to(device)
            waveform_length = batch[1]
            tokens = batch[2].to(device)
            token_length = batch[3]
            transcript = batch[4]

            specs, spec_length, teacher_durations = calc_alignment_and_spec(
                melspec,
                aligner,
                waveform,
                waveform_length,
                token_length,
                transcript
            )
            specs = specs.to(device)
            spec_length = spec_length.to(device)
            teacher_durations = teacher_durations.to(device)

            spec_mask = create_mask(specs, spec_length).to(device)
            durations_mask = create_mask(tokens, token_length).to(device)

            pred_spec, pred_durations = model.inference(tokens, token_length)

            # match sequence length
            pred_spec = pred_spec[:, :, :specs.shape[-1]]
            diff = specs.shape[-1] - pred_spec.shape[-1]
            diff_tensor = torch.zeros(specs.shape[0], 80, diff, device=device) * SPEC_FILL
            pred_spec = torch.cat([pred_spec, diff_tensor], dim=2)

            loss = criterion(pred_spec, specs, spec_mask,
                             pred_durations, teacher_durations, durations_mask) 
            global_loss += loss.item()

        rand_int = random.randint(0, pred_spec.shape[0] - 1)
        log_transcript = transcript[rand_int]
        log_pred_spec = pred_spec[rand_int].detach()[:, :spec_length[rand_int]]
        log_true_spec = specs[rand_int].detach()[:, :spec_length[rand_int]]
        log_pred_audio = vocoder.inference(log_pred_spec.unsqueeze(0))
        log_true_audio = waveform[rand_int] # vocoder(log_true_spec.unsqueeze(0))

    if logger is not None:
        logger.log_spec_and_audio(
            log_transcript,
            log_pred_spec.cpu(),
            log_true_spec.cpu(),
            log_pred_audio.detach().cpu(),
            log_true_audio.detach().cpu()
        )

    global_loss /= len(val_loader)

    if best_loss > global_loss:
        best_loss = global_loss
        torch.save(model.state_dict(), f'../chkpt/{exp_name}_best.pt')
    torch.save(model.state_dict(), f'../chkpt/{exp_name}_last.pt')

    to_log = {'Val Loss': global_loss, 'Best Val Loss': best_loss}
    if logger is not None:
        logger.log_metrics(to_log)
    else:
        print(to_log)

    return best_loss


def _parse_args():
    parser = argparse.ArgumentParser(description='Train argparser')
    parser.add_argument(
        '-c', '--config',
        help='Path to config',
        required=True
    )
    return parser.parse_args()    


if __name__ == '__main__':
    args = _parse_args()
    config = Config(args.config)

    train_loader, val_loader = config.get_dataloaders()
    aligner = config.get_aligner()
    vocoder = config.get_vocoder()
    model = config.get_model()
    melspec = config.get_melspec()
    criterion = config.get_criterion()
    optimizer = config.get_optimizer(model)
    scheduler = config.get_scheduler(optimizer)
    logger = config.get_logger()

    train(model, optimizer, scheduler, train_loader, val_loader, criterion,
          aligner, melspec, vocoder, config.device, config.exp_name, config.clip,
          logger, config.log_interval, config.best_loss)

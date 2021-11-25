import torch
from torch import nn
from torch.utils.data import DataLoader

import json

from .model import FastSpeech
from .aligner import GraphemeAligner
from .melspectrogram import MelSpectrogram
from .collator import LJSpeechCollator
from .scheduler import CosineAnnealingWarmup
from .dataset import LJSpeechDataset
from .loss import FastSpeechLoss 
from .logger import WandbLogger


class Config:
    def __init__(self, path):
        with open(path, 'r') as fp:
            self.config = json.load(fp)
        self.device = self.config['device']
        self.log_interval = self.config['log_interval']
        self.clip = self.config['clip']
        self.best_loss = self.config['best_loss']
        self.exp_name = self.config['exp_name']
        print(self.config)

    def get_model(self):
        model = FastSpeech(**self.config['model'])
        if len(self.config['chkpt']) > 0:
            model.load_state_dict(torch.load(self.config['chkpt'], map_location='cpu'))
        if self.device == 'cpu':
            model = model.to('cpu')
        elif 'cuda' in self.device:
            # work only on one GPU..............
            model = model.to(self.device)
        print(model)
        print('Number of parameters', sum(p.numel() for p in model.parameters()))
        return model

    def get_optimizer(self, model):
        optimizer_class = getattr(torch.optim, self.config['optim']['name'])
        optimizer = optimizer_class(model.parameters(), **self.config['optim']['args'])
        print(optimizer)
        return optimizer

    def get_scheduler(self, optimizer):
        if 'scheduler' in self.config:
            scheduler = CosineAnnealingWarmup(optimizer, **self.config['scheduler'])
        else:
            scheduler = None
        print(scheduler)
        return scheduler

    def get_dataloaders(self):
        dataset = LJSpeechDataset(self.config['data']['path'])
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [12800, 300],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            train_dataset,
            collate_fn=LJSpeechCollator(),
            **self.config['data']['train']
        )
        val_loader = DataLoader(
            val_dataset,
            collate_fn=LJSpeechCollator(),
            **self.config['data']['val']
        )
        return train_loader, val_loader

    def get_criterion(self):
        return FastSpeechLoss()

    def get_vocoder(self):
        vocoder = torch.hub.load('seungwonpark/melgan', 'melgan').to(self.device)
        return vocoder

    def get_melspec(self):
        melspec = MelSpectrogram(self.config['melspec']).to(self.device)
        return melspec

    def get_aligner(self):
        aligner = GraphemeAligner().to(self.device)
        return aligner

    def get_logger(self):
        if 'wandb' in self.config:
            logger = WandbLogger(**self.config['wandb'])
        else:
            logger = None
        return logger

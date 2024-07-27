import os.path

import torch
import logging
import shutil
from torch import nn
from model.AlexNet import AlexNet
from data_loader.imagenet import ImagenetteDataLoader
from utils.util import *
from base.base_trainer import BaseTrainer
from torchvision import transforms


class AlexNetAgent(BaseTrainer):

    def __init__(self, data_dir="../datasets/imagenette2-320", batch_size=128, num_workers=1, lr=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)

        transform = transforms.Compose(
            [transforms.CenterCrop(255), transforms.Resize((224, 224)), transforms.ToTensor()])
        self.data_loader = ImagenetteDataLoader(data_dir=data_dir, batch_size=batch_size,
                                                num_workers=num_workers, transform=transform)
        self.model = AlexNet().to(self.device)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.mnt_best = None
        self.early_stop = None if "early_stop" not in kwargs else kwargs["early_stop"]
        self.initialize_context()

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_normal_(layer.weight)

    def initialize_context(self):
        self.model.apply(AlexNetAgent.init_weights)

    @staticmethod
    def accuracy(y_hat, y):
        y_hat = y_hat.argmax(dim=1)
        cmp = y_hat.type(y.dtype) == y
        return cmp.type(y.dtype).sum()

    def run(self):
        self.train()

    def train(self):
        not_improved_count = 0

        # resume
        if self.resume:
            self.load_checkpoint()

        # train for epochs
        for epoch in range(self.current_epoch + 1, self.max_epoch + 1):
            self.current_epoch = epoch

            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            improved = self.mnt_best is None or val_loss <= self.mnt_best

            if improved:
                self.mnt_best = val_loss
                not_improved_count = 0
                is_best = True
            else:
                not_improved_count += 1
                is_best = False

            if self.early_stop and not_improved_count > self.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                 "Training stops.".format(self.early_stop))
                break

            self.save_checkpoint(is_best)

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (X, y) in enumerate(self.data_loader.train):
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            y_hat = self.model(X)
            partial_loss = self.loss(y_hat, y)
            # train loss for batch
            batch_train_loss = partial_loss.mean()
            batch_train_loss.backward()
            # record for whole epoch
            train_loss += partial_loss.sum()
            correct += self.accuracy(y_hat, y)
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(X), len(self.data_loader.train.dataset),
                    100. * batch_idx / len(self.data_loader.train), batch_train_loss))
        train_loss /= len(self.data_loader.train.dataset)
        self.logger.info('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(self.data_loader.train.dataset),
            100. * correct / len(self.data_loader.train.dataset)))
        return train_loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X, y in self.data_loader.val:
                X, y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                val_loss += self.loss(y_hat, y).sum()
                correct += self.accuracy(y_hat, y)
        val_loss /= len(self.data_loader.val.dataset)
        self.logger.info('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(self.data_loader.val.dataset),
            100. * correct / len(self.data_loader.val.dataset)))
        return val_loss

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        torch.save(state, os.path.join(self.checkpoint_dir, self.checkpoint_file))
        if is_best:
            shutil.copyfile(os.path.join(self.checkpoint_dir, self.checkpoint_file),
                            os.path.join(self.checkpoint_dir, "model_best.pth.tar"))

    def load_checkpoint(self):
        fname = os.path.join(self.checkpoint_dir, self.checkpoint_file)
        if not os.path.exists(fname):
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.checkpoint_dir))
            return
        self.logger.info("Loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)

        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.mnt_best = checkpoint['monitor_best']

    def finalize(self):
        self.save_checkpoint()

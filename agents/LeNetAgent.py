import torch
import logging
from torch import nn
from model.LeNet import LeNet
from data_loader.mnist import FashionMNISTDataLoader
from utils.util import *
from base.base_trainer import BaseTrainer


class LeNetAgent(BaseTrainer):

    def __init__(self, max_epoch=10, batch_size=128, num_workers=1, lr=0.5, log_interval=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_loader = FashionMNISTDataLoader(batch_size=batch_size, num_workers=num_workers)
        self.model = LeNet().to(self.device)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.initialize_context()

    @staticmethod
    def init_weights(layer):
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_normal_(layer.weight)

    def initialize_context(self):
        self.model.apply(LeNetAgent.init_weights)

    @staticmethod
    def accuracy(y_hat, y):
        y_hat = y_hat.argmax(dim=1)
        cmp = y_hat.type(y.dtype) == y
        return cmp.type(y.dtype).sum()

    def run(self):
        self.train()

    def train(self):
        for epoch in range(1, self.max_epoch + 1):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()

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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(X), len(self.data_loader.train.dataset),
                    100. * batch_idx / len(self.data_loader.train), batch_train_loss))
        train_loss /= len(self.data_loader.train.dataset)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, len(self.data_loader.train.dataset),
            100. * correct / len(self.data_loader.train.dataset)))

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
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, correct, len(self.data_loader.val.dataset),
            100. * correct / len(self.data_loader.val.dataset)))


    def finalize(self):
        pass
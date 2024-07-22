from base import BaseDataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils import data


class FashionMNISTDataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size):
        self.train_data = datasets.FashionMNIST(root=data_dir, train=True, transform=transforms.ToTensor(),
                                                download=True)
        self.test_data = datasets.FashionMNIST(root=data_dir, train=False, transform=transforms.ToTensor(),
                                               download=True)
        self.batch_size = batch_size

    def get_train_data_loader(self):
        return data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def get_val_data_loader(self):
        return data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
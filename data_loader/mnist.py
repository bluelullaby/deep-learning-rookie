import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils import data


class FashionMNISTDataLoader(object):

    def __init__(self, data_dir="./datasets", num_workers=1, batch_size=128, *args, **kwargs):
        trans = transforms.Compose([transforms.ToTensor()])
        self.train_data = datasets.FashionMNIST(root=data_dir, train=True, transform=trans, download=True)
        self.val_data = datasets.FashionMNIST(root=data_dir, train=False, transform=trans, download=True)
        self.batch_size = batch_size
        self.num_workers= num_workers

    def __getattr__(self, item):
        if item == "train":
            return data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        elif item == "val":
            return data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return None
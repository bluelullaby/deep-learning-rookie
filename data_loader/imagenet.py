import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class ImagenetteDataset(Dataset):

    def __init__(self, data_dir: str, split: str, transform=None):
        super().__init__()
        labels = os.listdir(os.path.join(data_dir, split))
        self.images = []
        self.transform = transform
        self.grayscale = transforms.Grayscale(3)
        for idx, label in enumerate(labels):
            for fname in os.listdir(os.path.join(data_dir, split, label)):
                self.images.append((os.path.join(data_dir, split, label, fname), idx))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        image = Image.open(image)
        if image.layers == 1:
            image = self.grayscale(image)
        if self.transform:
            image = self.transform(image)
        return image, label


class ImagenetteDataLoader(object):

    def __init__(self, data_dir, num_workers, batch_size, transform, *args, **kwargs):
        self.train_data = ImagenetteDataset(data_dir=data_dir, split="train", transform=transform)
        self.val_data = ImagenetteDataset(data_dir=data_dir, split="val", transform=transform)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __getattr__(self, item):
        if item == "train":
            return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        elif item == "val":
            return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return None

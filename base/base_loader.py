from torch.utils.data import DataLoader


class BaseDataLoader(object):

    def __getitem__(self, item) -> DataLoader:
        assert item in ("train", "val")
        if item == "train":
            return self.get_train_data_loader()
        else:
            return self.get_val_data_loader()

    def get_train_data_loader(self):
        raise NotImplemented

    def get_val_data_loader(self):
        return NotImplemented

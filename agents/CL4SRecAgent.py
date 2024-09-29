import os

from base.base_trainer import BaseTrainer
from data_loader.recommender import Sequence, load_data_file
from data_loader.amazon import next_batch_sequence


class CL4SRecAgent(BaseTrainer):

    def __init__(self,
                 data_dir="./datasets/yelp2018",
                 batch_size=2048,
                 max_len=50,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # params
        self.batch_size = batch_size
        self.max_len = max_len
        # dataset
        self.data = Sequence(load_data_file(os.path.join(data_dir, "train.txt"), data_type="sequence"),
                             load_data_file(os.path.join(data_dir, "test.txt"), data_type="sequence"))



    def train_one_epoch(self):
        for batch_idx, (seq, pos, y, neg_idx, seq_len) in next_batch_sequence(self.data, self.batch_size, self.max_len):
            pass
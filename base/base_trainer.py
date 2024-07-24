import logging


class BaseTrainer(object):

    def __init__(self):
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, file_name):
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train_one_epoch(self, epoch):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

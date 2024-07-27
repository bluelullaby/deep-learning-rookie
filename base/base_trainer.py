import logging
from utils.util import *


class BaseTrainer(object):

    def __init__(self, *args, **kwargs):
        # logger
        self.logger = logging.getLogger()
        self.log_interval: int = 10 if "log_interval" not in kwargs else kwargs["log_interval"]
        # checkpoint
        self.resume: bool = False if "resume" not in kwargs else kwargs["resume"]
        self.checkpoint_dir: str = kwargs["checkpoint_dir"]
        self.checkpoint_file: str = "checkpoint.pth.tar" if "checkpoint_file" not in kwargs else kwargs["checkpoint_file"]
        # epoch
        self.current_epoch = 0
        assert "max_epoch" in kwargs
        self.max_epoch = kwargs["max_epoch"]
        # device
        use_gpu = True if "cuda" not in kwargs else kwargs["cuda"]
        if use_gpu:
            self.device = try_gpu()
        else:
            self.device = try_cpu()

    def load_checkpoint(self):
        raise NotImplementedError

    def save_checkpoint(self, is_best=0):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

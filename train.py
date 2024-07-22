import argparse
from parse_config import ConfigParser
import data_loader as module_data


def main(config: ConfigParser):
    # load train and val dataloader
    data_loader = config.init_obj("data_loader", module_data)
    train_data_loader, val_data_loader = data_loader["train"], data_loader["val"]



if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Pytorch Train Project")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")

    config = ConfigParser.from_args(args)
    main(config)
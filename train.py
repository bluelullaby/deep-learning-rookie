import argparse
from parse_config import ConfigParser


def main(config):
    pass


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Pytorch Train Project")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")

    config = ConfigParser.from_args(args)
    main(config)
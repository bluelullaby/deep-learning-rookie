import json
import os
import logging
import sys
from logging import Formatter
from logging.handlers import RotatingFileHandler


def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            return json.load(config_file)
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s]: %(message)s"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_info.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_file_handler.setLevel(logging.INFO)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10 ** 6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def process_config(json_file):
    config = get_config_from_json(json_file)
    print(" THE Configuration of your experiment ..")
    print(json.dumps(config, indent=4))

    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config["exp_name"]))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    # create some important directories to be used for that experiment.
    config["summary_dir"] = os.path.join("experiments", config["exp_name"], "summaries/")
    config["checkpoint_dir"] = os.path.join("experiments", config["exp_name"], "checkpoints/")
    config["out_dir"] = os.path.join("experiments", config["exp_name"], "out/")
    config["log_dir"] = os.path.join("experiments", config["exp_name"], "logs/")
    create_dirs([config["summary_dir"], config["checkpoint_dir"], config["out_dir"], config["log_dir"]])

    # setup logging in the project
    setup_logging(config["log_dir"])

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config


def create_dirs(dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

import logging


def setup_logger(name: str, level: int = 10):
    logging.basicConfig(format='%(asctime)s [%(levelname)s][%(filename)s] - %(message)s', level=level)
    root_logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler("{0}.log".format(name))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

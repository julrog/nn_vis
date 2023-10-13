import os
import zipfile

import wget

from definitions import DATA_PATH


def download_and_unzip_sample() -> str:
    output_directory = DATA_PATH
    filename = wget.download(
        'https://drive.google.com/uc?export=download&id=1EpsubJhHH4shqzDhsBB0SHsBjWgWa03S', out=output_directory)
    zip_filepath = os.path.join(output_directory, filename)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)
    return os.path.join(DATA_PATH, 'sample_model.npz')


download_and_unzip_sample()

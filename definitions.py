import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = BASE_PATH + "/storage/data/"

def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it), next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return

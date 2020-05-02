import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


def pairwise(it):
    it = iter(it)
    while True:
        try:
            yield next(it), next(it), next(it), next(it)
        except StopIteration:
            # no more elements in the iterator
            return

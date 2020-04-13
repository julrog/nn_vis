import time

from file import FileHandler

track_recursive = []


def track_time(func):
    def tracked_func(*args, **kwargs):
        global track_recursive
        start_time = time.perf_counter()
        track_recursive.append(start_time)
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        time_diff = end_time - track_recursive.pop()
        track_recursive = [start_time + time_diff for start_time in track_recursive]
        stats = dict()
        stats[func.__name__] = time_diff
        FileHandler().append_statistics(stats)
        return result

    return tracked_func

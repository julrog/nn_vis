import time

from file import FileHandler

running_times = []


def track_time(_func=None, *, track_recursive: bool = True):
    def track_time_inner(func):
        def tracked_func(*args, **kwargs):
            global running_times
            start_time = time.perf_counter()
            running_times.append(start_time)
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            time_diff = (end_time - running_times.pop()) if track_recursive else end_time - start_time
            running_times = [start_time + time_diff for start_time in running_times]
            stats = dict()
            stats[func.__name__] = time_diff
            FileHandler().append_statistics(stats)
            return result

        return tracked_func

    if _func is None:
        return track_time_inner
    else:
        return track_time_inner(_func)

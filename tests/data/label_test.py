import numpy as np

from src.torchsofa.data.label import start_time_to_interval


def test_start_time_to_interval():
    start_time = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    intervals = start_time_to_interval(start_time, 10.0)
    print(intervals.shape, intervals)


if __name__ == "__main__":
    test_start_time_to_interval()

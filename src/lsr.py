import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


def compute(file_path, plot):
    (xs, ys) = load_points_from_file(file_path)


def main(argv):
    if len(argv) < 2 or len(argv) > 3:
        print("Incorrect number of arguments")
        exit(1)
    file_path = argv[1]
    if not os.path.isfile(file_path):
        print("Invalid file path")
        exit(1)
    plot = False
    if len(argv) == 3:
        if argv[3] == "--plot":
            plot = True
        else:
            print("Unrecognised argument")
            exit(1)
    compute(file_path, plot)


if __name__ == "__main__":
    main(sys.argv)
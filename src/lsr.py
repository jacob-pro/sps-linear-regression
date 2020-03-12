import os
import sys
import pandas as pd
import numpy as np
from typing import *
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


Points = Tuple[List[float], List[float]]


def group_points_into_segments(xs: List[float], ys: List[float]) -> List[Points]:
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    lines: int = len(xs) // 20
    xs_split = np.split(np.array(xs), lines)
    ys_split = np.split(np.array(ys), lines)
    return list(map(lambda line: (xs_split[line], ys_split[line]), range(lines)))


def lsr_fn(xs: List[float], ys: List[float], fn: Callable):
    x_e = np.column_stack((np.ones(len(xs)), fn(xs)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys)
    return v


def lsr_polynomial(xs: List[float], ys: List[float], degree: int):
    # x**0 = 1, creating the column of ones
    columns = list(map(lambda i: list(map(lambda x: x**i, xs)), range(degree + 1)))
    x_e = np.column_stack(columns)
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys)
    return v


def compute(file_path: str, plot: bool) -> None:
    (xs, ys) = load_points_from_file(file_path)
    segments = group_points_into_segments(xs, ys)

    (xs1, ys1) = segments[0]
    lsr_polynomial(xs1, ys1, 0)
    lsr_fn(xs1, ys1, np.tan)

    if plot:
        view_data_segments(xs, ys)


def main(argv: List) -> None:
    if len(argv) < 2 or len(argv) > 3:
        print("Incorrect number of arguments")
        exit(1)
    file_path = argv[1]
    if not os.path.isfile(file_path):
        print("Invalid file path")
        exit(1)
    plot = False
    if len(argv) == 3:
        if argv[2] == "--plot":
            plot = True
        else:
            print("Unrecognised argument")
            exit(1)
    compute(file_path, plot)


if __name__ == "__main__":
    main(sys.argv)
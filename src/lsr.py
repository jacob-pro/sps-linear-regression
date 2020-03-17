import os
import sys
import pandas as pd
import numpy as np
from typing import *
from matplotlib import pyplot as plt
from numpy import ndarray
from dataclasses import dataclass
from abc import abstractmethod, ABC


@dataclass()
class LsrResult(ABC):
    name: str
    coefficients: ndarray
    error: float
    equation: str

    @abstractmethod
    def compute_for_x(self, x: ndarray) -> ndarray:
        pass


@dataclass()
class LsrResultPoly(LsrResult):

    def compute_for_x(self, x: ndarray) -> ndarray:
        return np.polyval(np.flip(self.coefficients), x)


@dataclass()
class LsrResultFn(LsrResult):
    function: Callable

    def compute_for_x(self, x: ndarray) -> ndarray:
        a, b = self.coefficients
        return a + b * self.function(x)


def load_points_from_file(filename: str) -> Tuple[ndarray, ndarray]:
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs: ndarray, ys: ndarray, lines: List[LsrResult]) -> None:
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

    for idx, line in enumerate(lines):
        x = np.linspace(xs[20 * idx], xs[20 * (idx + 1) - 1])
        y = line.compute_for_x(x)
        plt.plot(x, y, linestyle="solid")

    plt.show()


Segment = Tuple[ndarray, ndarray]


def group_points_into_segments(xs: ndarray, ys: ndarray) -> List[Segment]:
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    lines: int = len(xs) // 20
    xs_split = np.split(np.array(xs), lines)
    ys_split = np.split(np.array(ys), lines)
    return list(map(lambda line: (xs_split[line], ys_split[line]), range(lines)))


# Float to string with prefixed sign
def fts(x: float) -> str:
    y = '{:+.2f}'.format(x)
    return "{} {}".format(y[0:1], y[1:])


# String representation of poly multiplier
def xv(x: int) -> str:
    if x == 0:
        return ''
    elif x == 1:
        return " * x"
    else:
        return " * x^{}".format(x)


def lsr_fn(xs: ndarray, ys: ndarray, fn: Callable) -> LsrResult:
    x_e = np.column_stack((np.ones(len(xs)), fn(xs)))
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys)
    a, b = v
    y_hat = a + b * fn(xs)
    e = float(np.sum((y_hat - ys) ** 2))
    formatted = "{} {} * {}(x)".format(fts(a), fts(b), fn.__name__)[2:]
    return LsrResultFn(fn.__name__, v, e, formatted, fn)


def lsr_polynomial(xs: ndarray, ys: ndarray, degree: int) -> LsrResultPoly:
    # x**0 = 1, creating the column of ones
    columns = list(map(lambda i: list(map(lambda x: x**i, xs)), range(degree + 1)))
    x_e = np.column_stack(columns)
    v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(ys)
    y_hat = np.polyval(np.flip(v), xs)
    e = float(np.sum((y_hat - ys) ** 2))
    fn_name = ["constant", "linear", "quadratic", "cubic"][degree] if degree <= 3 else "poly{}".format(degree)
    formatted = " ".join(map(lambda x: "{}{}".format(fts(v[x]), xv(x)), range(degree + 1)))[2:]
    return LsrResultPoly(fn_name, v, e, formatted)


def compute(segments: List[Segment]) -> Tuple[List[LsrResult], float]:

    max_poly = 4
    bests: Dict[int, Tuple[List[LsrResult], float]] = {}

    for degree in range(2, max_poly + 1):
        temp = []
        for (xs, ys) in segments:
            results: List[LsrResult] = [
                lsr_polynomial(xs, ys, 1),
                lsr_polynomial(xs, ys, degree),
                lsr_fn(xs, ys, np.sin),
                lsr_fn(xs, ys, np.cos),
                lsr_fn(xs, ys, np.tan),
                lsr_fn(xs, ys, np.exp),
                lsr_fn(xs, ys, np.reciprocal),
                # lsr_fn(xs, ys, np.sqrt),
            ]
            temp.append(min(results, key=lambda r: r.error))
        bests[degree] = (temp, sum(k.error for k in temp))

    best = min(bests.values(), key=lambda b: b[1])
    return best


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

    (xs, ys) = load_points_from_file(file_path)
    segments = group_points_into_segments(xs, ys)
    lines, error = compute(segments)
    print(error)
    if plot:
        view_data_segments(xs, ys, lines)


if __name__ == "__main__":
    main(sys.argv)
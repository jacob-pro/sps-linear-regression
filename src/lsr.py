from __future__ import annotations
import os
import sys
import pandas as pd
import numpy as np
from typing import *
from matplotlib import pyplot as plt
from numpy import ndarray
from dataclasses import dataclass
from abc import abstractmethod, ABC
import random

POLYNOMIAL_DEGREE = 3
UNKNOWN_FUNCTION = np.sin


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
        lines: List of lines to plot
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


def group_points_into_segments(xs: ndarray, ys: ndarray) -> List[Segment]:
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    lines: int = len(xs) // 20
    xs_split = np.split(np.array(xs), lines)
    ys_split = np.split(np.array(ys), lines)
    return list(map(lambda line: (Segment(xs_split[line], ys_split[line])), range(lines)))


@dataclass()
class LsrResult(ABC):
    ss_error: float

    # Apply this function to an array of x values
    @abstractmethod
    def compute_for_x(self, x: ndarray) -> ndarray:
        pass

    # Get the name of this function
    @abstractmethod
    def name(self) -> str:
        pass

    # Get the equation string representation of this function
    @abstractmethod
    def equation(self) -> str:
        pass

    # Float to string with prefixed sign
    @classmethod
    def fts(cls, x: float) -> str:
        y = '{:+.2f}'.format(x)
        return "{} {}".format(y[0:1], y[1:])


@dataclass()
class LsrResultPoly(LsrResult):
    coefficients: ndarray

    def name(self) -> str:
        degree = len(self.coefficients)
        return ["constant", "linear", "quadratic", "cubic"][degree] if degree <= 3 else "poly{}".format(degree)

    def compute_for_x(self, x: ndarray) -> ndarray:
        return np.polyval(np.flip(self.coefficients), x)

    # String representation of poly multiplier
    @classmethod
    def xv(cls, x: int) -> str:
        if x == 0:
            return ''
        elif x == 1:
            return " * x"
        else:
            return " * x^{}".format(x)

    def equation(self) -> str:
        return " ".join(map(lambda x: "{}{}".format(self.fts(self.coefficients[x]), self.xv(x)),
                            range(len(self.coefficients) + 1)))[2:]


@dataclass()
class LsrResultFn(LsrResult):
    a: float
    b: float
    function: Callable

    def name(self) -> str:
        return self.function.__name__

    def compute_for_x(self, x: ndarray) -> ndarray:
        return self.a + self.b * self.function(x)

    def equation(self) -> str:
        return "{} {} * {}(x)".format(self.fts(self.a), self.fts(self.b), self.function.__name__)[2:]


@dataclass()
class SplitSegment:
    training: Segment
    validation: Segment


# A segment of a line
@dataclass()
class Segment:
    xs: ndarray
    ys: ndarray

    @dataclass()
    class Point:
        x: float
        y: float

    @classmethod
    def from_points(cls, points: List[Point]):
        xs = ndarray(map(lambda p: p.x, points))
        ys = ndarray(map(lambda p: p.x, points))
        return Segment(xs, ys)

    def to_points(self) -> List[Point]:
        return list(map(lambda i: (Segment.Point(self.xs[i], self.ys[i])), range(len(self.xs))))

    def split(self, k: int) -> List[SplitSegment]:
        shuffled: List[Segment.Point] = random.sample(self.to_points())
        validation_size = len(shuffled) // k
        split_segments: List[SplitSegment] = []
        for i in range(k):
            validation = shuffled[:validation_size]
            training = shuffled[validation_size:]
            split_segments.append(SplitSegment(Segment.from_points(training), Segment.from_points(validation)))
            np.roll(shuffled, validation_size)
        return split_segments

    def lsr_fn(self, fn: Callable) -> LsrResult:
        x_e = np.column_stack((np.ones(len(self.xs)), fn(self.xs)))
        v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(self.ys)
        a, b = v
        y_hat = a + b * fn(self.xs)
        e = float(np.sum((y_hat - self.ys) ** 2))
        return LsrResultFn(e, a, b, fn)

    def lsr_polynomial(self, degree: int) -> LsrResult:
        # x**0 = 1, creating the column of ones
        columns = list(map(lambda i: list(map(lambda x: x ** i, self.xs)), range(degree + 1)))
        x_e = np.column_stack(columns)
        v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(self.ys)
        y_hat = np.polyval(np.flip(v), self.xs)
        e = float(np.sum((y_hat - self.ys) ** 2))
        return LsrResultPoly(e, v)


def compute(segments: List[Segment]) -> Tuple[List[LsrResult], float]:
    bests = []
    for s in segments:
        results: List[LsrResult] = [
            s.lsr_polynomial(1),
            s.lsr_polynomial(POLYNOMIAL_DEGREE),
            s.lsr_fn(UNKNOWN_FUNCTION),
        ]
        bests.append(min(results, key=lambda r: r.ss_error))
    return bests, sum(k.ss_error for k in bests)


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

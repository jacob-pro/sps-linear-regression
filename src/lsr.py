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
import unittest

POLYNOMIAL_DEGREE = 2
UNKNOWN_FUNCTION = np.sin
K_FOLD = 4


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
        # print(line.name(), line.equation())

    plt.show()


def group_points_into_segments(xs: ndarray, ys: ndarray) -> List[Segment]:
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    lines: int = len(xs) // 20
    xs_split = np.split(np.array(xs), lines)
    ys_split = np.split(np.array(ys), lines)
    return list(map(lambda line: (Segment(xs_split[line], ys_split[line])), range(lines)))


def ss_error(y_hat: ndarray, y: ndarray) -> float:
    return float(np.sum((y_hat - y) ** 2))


@dataclass()
class ValidatedLsrResult:
    lsr_result: LsrResult
    cv_error: float


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
        degree = len(self.coefficients) - 1
        return ["constant", "linear", "quadratic", "cubic"][degree] if degree < 4 else "poly{}".format(degree)

    def compute_for_x(self, x: ndarray) -> ndarray:
        return np.polyval(np.flip(self.coefficients), x)

    # String representation of poly multiplier
    @classmethod
    def xv(cls, x: int) -> str:
        if x == 0:
            return ''
        elif x == 1:
            return "*x"
        else:
            return "*x^{}".format(x)

    def equation(self) -> str:
        return " ".join(map(lambda x: "{}{}".format(self.fts(self.coefficients[x]), self.xv(x)),
                            range(len(self.coefficients))))[2:]


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
        return "{} {}*{}(x)".format(self.fts(self.a), self.fts(self.b), self.function.__name__)[2:]


@dataclass()
class SplitSegment:
    training: Segment
    validation: Segment


@dataclass()
class Point:
    x: float
    y: float


# A segment of a line
@dataclass()
class Segment:
    xs: ndarray
    ys: ndarray

    @classmethod
    def from_points(cls, points: ndarray):
        xs = np.asarray(list(map(lambda p: p.x, points)))
        ys = np.asarray(list(map(lambda p: p.y, points)))
        return Segment(xs, ys)

    def to_points(self) -> List[Point]:
        return list(map(lambda i: (Point(self.xs[i], self.ys[i])), range(len(self.xs))))

    def split(self, k: int) -> List[SplitSegment]:
        points: ndarray = np.asarray(self.to_points())
        validation_size = len(points) // k
        split_segments: List[SplitSegment] = []
        for i in range(k):
            validation = points[:validation_size]
            training = points[validation_size:]
            split_segments.append(SplitSegment(Segment.from_points(training), Segment.from_points(validation)))
            points = np.roll(points, -validation_size)
        return split_segments

    def lsr_fn(self, fn: Callable) -> LsrResult:
        x_e = np.column_stack((np.ones(len(self.xs)), fn(self.xs)))
        v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(self.ys)
        a, b = v
        y_hat = a + b * fn(self.xs)
        e = ss_error(y_hat, self.ys)
        return LsrResultFn(e, a, b, fn)

    def lsr_polynomial(self, degree: int) -> LsrResult:
        # x**0 = 1, creating the column of ones
        columns = list(map(lambda i: list(map(lambda x: x ** i, self.xs)), range(degree + 1)))
        x_e = np.column_stack(columns)
        v = np.linalg.inv(x_e.T.dot(x_e)).dot(x_e.T).dot(self.ys)
        y_hat = np.polyval(np.flip(v), self.xs)
        e = ss_error(y_hat, self.ys)
        return LsrResultPoly(e, v)

    def cross_validated(self, k: int, fn: Callable[[Segment], LsrResult]) -> ValidatedLsrResult:
        ks: List[SplitSegment] = self.split(k)
        errors = []
        for ss in ks:
            lsr_result = fn(ss.training)
            y_hat = lsr_result.compute_for_x(ss.validation.xs)
            errors.append(ss_error(y_hat, ss.validation.ys))
        avg_error = float(np.mean(errors))
        return ValidatedLsrResult(fn(self), avg_error)


@dataclass()
class BestFitResult:
    lines: List[LsrResult]
    total_ss_error: float
    total_cv_error: float


def compute(segments: List[Segment], poly_degree: int, unknown_fn: Callable) -> BestFitResult:
    bests: List[ValidatedLsrResult] = []
    for s in segments:
        results: List[ValidatedLsrResult] = [
            s.cross_validated(K_FOLD, lambda x: x.lsr_polynomial(1)),
            s.cross_validated(K_FOLD, lambda x: x.lsr_polynomial(poly_degree)),
            s.cross_validated(K_FOLD, lambda x: x.lsr_fn(unknown_fn))
        ]
        bests.append(min(results, key=lambda r: r.cv_error))
    total_cv_error = sum(k.cv_error for k in bests)
    lines: [LsrResult] = list(map(lambda x: x.lsr_result, bests))
    total_ss_error = sum(k.ss_error for k in lines)
    return BestFitResult(lines, total_ss_error, total_cv_error)


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
    result = compute(segments, POLYNOMIAL_DEGREE, UNKNOWN_FUNCTION)
    print(result.total_ss_error)
    if plot:
        view_data_segments(xs, ys, result.lines)


def evaluate_training_data() -> None:
    # Get all line segments from all training files
    training_files = list(filter(lambda x: x.endswith(".csv"), os.listdir("train_data")))
    all_segments: List[Segment] = []
    for f in training_files:
        (xs, ys) = load_points_from_file("train_data/{}".format(f))
        all_segments.extend(group_points_into_segments(xs, ys))

    candidate_functions = [np.sin, np.cos, np.tan, np.reciprocal, np.exp, np.square, np.sqrt, np.log]
    candidate_polynomials = range(2, 6)

    results = []
    for f in candidate_functions:
        for p in candidate_polynomials:
            result = compute(all_segments, p, f)
            results.append((result.total_cv_error, p, f))
    results.sort(key= lambda x: x[0])
    for r in results:
        print("Total CV Error: {}, Polynomial Degree: {}, Function: {}".format(r[0], r[1], r[2].__name__))


if __name__ == "__main__":
    # evaluate_training_data()
    main(sys.argv)


class TestSegment(unittest.TestCase):

    def test_from_points_to_points(self):
        points = [Point(1, -1), Point(2, -2), Point(3, -3)]
        s = Segment.from_points(np.asarray(points))
        np.testing.assert_array_equal([1, 2, 3], s.xs)
        np.testing.assert_array_equal([-1, -2, -3], s.ys)
        np.testing.assert_array_equal(s.to_points(), points)

    def test_split(self):
        points = [Point(1, -1), Point(2, -2), Point(3, -3), Point(4, -4)]
        s = Segment.from_points(np.asarray(points))
        split: List[SplitSegment] = s.split(4)
        self.assertEqual(4, len(split))
        for x in split:
            self.assertEqual(len(x.training.xs), 3)
            self.assertEqual(len(x.validation.xs), 1)
        # Check that 4 different validation points are chosen
        all_validation = list(map(lambda x: x.validation.xs[0], split))
        np.testing.assert_array_equal([1, 2, 3, 4], all_validation)
        # Check that all 4 training sets are different
        all_training_sets = list(map(lambda x: tuple(x.training.xs.tolist()), split))
        self.assertEqual(len(all_training_sets), len(set(all_training_sets)))

    def test_lsr_fn(self):
        points = [Point(1, 5), Point(2, 7), Point(4, 9), Point(10, 12)]
        s = Segment.from_points(np.asarray(points))
        ln = s.lsr_fn(np.log)
        self.assertEqual(ln.equation(), "4.93 + 3.03*log(x)")
        self.assertAlmostEquals(ln.ss_error, 0.03148, 5)

    def test_lsr_polynomial(self):
        points = [Point(1, 5), Point(2, 7), Point(4, 9), Point(10, 12)]
        s = Segment.from_points(np.asarray(points))
        linear = s.lsr_polynomial(1)
        self.assertEqual(linear.equation(), "5.22 + 0.71*x")
        self.assertAlmostEqual(linear.ss_error, 1.97949, 5)
        cubic = s.lsr_polynomial(3)
        self.assertEqual(cubic.equation(), "2.09 + 3.42*x - 0.54*x^2 + 0.03*x^3")
        self.assertAlmostEqual(cubic.ss_error, 1.24487e-24, 5)

    def test_cross_validated(self):
        pass

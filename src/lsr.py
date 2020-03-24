from __future__ import annotations

import argparse
import os
import unittest
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import *
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray

POLYNOMIAL_DEGREE = 3
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


def view_data_segments(xs: ndarray, ys: ndarray, lines: List[LsrResult], print_eq: bool) -> None:
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
        lines:
        print_eq:
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

    if print_eq:
        name_col = list(map(lambda z: z.name(), lines))
        eq_col = list(map(lambda z: z.equation(), lines))
        print("\n", pd.DataFrame(np.column_stack([name_col, eq_col]), range(1, len(lines) + 1), ['Type', 'Equation']))

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


# A segment of a line
class Segment:
    xs: ndarray
    ys: ndarray

    def __init__(self, xs: ndarray, ys: ndarray):
        assert len(xs) == len(ys)
        self.xs = xs
        self.ys = ys

    def split(self, k: int) -> List[SplitSegment]:
        xs = self.xs
        ys = self.ys
        validation_size = len(xs) // k
        split_segments: List[SplitSegment] = []
        for i in range(k):
            validation_xs = xs[:validation_size]
            validation_ys = ys[:validation_size]
            training_xs = xs[validation_size:]
            training_ys = ys[validation_size:]
            split_segments.append(SplitSegment(Segment(training_xs, training_ys), Segment(validation_xs, validation_ys)))
            xs = np.roll(xs, -validation_size)
            ys = np.roll(ys, -validation_size)
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


def compute(segments: List[Segment], k_fold: int, poly_degree: int, unknown_fn: Callable) -> BestFitResult:
    bests: List[ValidatedLsrResult] = []
    for s in segments:
        results: List[ValidatedLsrResult] = [
            s.cross_validated(k_fold, lambda x: x.lsr_polynomial(1)),
            s.cross_validated(k_fold, lambda x: x.lsr_polynomial(poly_degree)),
            s.cross_validated(k_fold, lambda x: x.lsr_fn(unknown_fn))
        ]
        bests.append(min(results, key=lambda r: r.lsr_result.ss_error))
    total_cv_error = sum(k.cv_error for k in bests)
    lines: [LsrResult] = list(map(lambda x: x.lsr_result, bests))
    total_ss_error = sum(k.ss_error for k in lines)
    return BestFitResult(lines, total_ss_error, total_cv_error)


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
            result = compute(all_segments, K_FOLD, p, f)
            results.append((result.total_cv_error, p, f))
    results.sort(key= lambda x: x[0])
    for r in results:
        print("Total CV Error: {}, Polynomial Degree: {}, Function: {}".format(r[0], r[1], r[2].__name__))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help='path to the input file')
    parser.add_argument('--plot', action='store_true', help='plot the result')
    parser.add_argument('--print', action='store_true', help='print the equations')
    parser.add_argument('--evaluate', action='store_true', help='evaluate all training data')
    args = parser.parse_args()

    if args.evaluate:
        evaluate_training_data()
    else:
        if not os.path.isfile(args.file_path):
            print("Invalid file path")
            exit(1)
        (xs, ys) = load_points_from_file(args.file_path)
        segments = group_points_into_segments(xs, ys)
        result = compute(segments, K_FOLD, POLYNOMIAL_DEGREE, UNKNOWN_FUNCTION)
        print(result.total_ss_error)
        if args.plot:
            view_data_segments(xs, ys, result.lines, args.print)


class TestSegment(unittest.TestCase):

    def test_split(self):
        s = Segment(np.asarray([1, 2, 3, 4]), np.asarray([-1, -2, -3, -4]))
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
        s = Segment(np.asarray([1, 2, 4, 10]), np.asarray([5, 7, 9, 12]))
        ln = s.lsr_fn(np.log)
        self.assertEqual(ln.equation(), "4.93 + 3.03*log(x)")
        self.assertAlmostEqual(ln.ss_error, 0.03148, 5)

    def test_lsr_polynomial(self):
        s = Segment(np.asarray([1, 2, 4, 10]), np.asarray([5, 7, 9, 12]))
        linear = s.lsr_polynomial(1)
        self.assertEqual(linear.equation(), "5.22 + 0.71*x")
        self.assertAlmostEqual(linear.ss_error, 1.97949, 5)
        cubic = s.lsr_polynomial(3)
        self.assertEqual(cubic.equation(), "2.09 + 3.42*x - 0.54*x^2 + 0.03*x^3")
        self.assertAlmostEqual(cubic.ss_error, 1.24487e-24, 5)

    def test_cross_validated(self):
        s = Segment(np.asarray([1, 2, 4, 10]), np.asarray([5, 7, 9, 12]))

        split = s.split(2)

        fold1 = split[0]
        y_hat_1 = fold1.training.lsr_polynomial(1).compute_for_x(fold1.validation.xs)
        error_1 = ss_error(y_hat_1, fold1.validation.ys)

        fold2 = split[1]
        y_hat_2 = fold2.training.lsr_polynomial(1).compute_for_x(fold2.validation.xs)
        error_2 = ss_error(y_hat_2, fold2.validation.ys)

        total_cv_error = np.mean([error_1, error_2])
        cvr = s.cross_validated(2, lambda x: x.lsr_polynomial(1))
        lsr = s.lsr_polynomial(1)
        self.assertEqual(total_cv_error, cvr.cv_error)
        self.assertEqual(lsr.equation(), cvr.lsr_result.equation())


class TestCompute(unittest.TestCase):

    @dataclass()
    class LsrResultEmpty(LsrResult):

        def compute_for_x(self, x: ndarray) -> ndarray:
            pass

        def name(self) -> str:
            pass

        def equation(self) -> str:
            pass

    def test_compute(self):

        segment1 = Segment(np.asarray([]), np.asarray([]))
        segment1lsr = self.LsrResultEmpty(30)
        segment1.cross_validated = MagicMock(side_effect=[ValidatedLsrResult(self.LsrResultEmpty(554), 554),
                                                          ValidatedLsrResult(self.LsrResultEmpty(443), 443),
                                                          ValidatedLsrResult(segment1lsr, 30)])

        segment2 = Segment(np.asarray([]), np.asarray([]))
        segment2lsr = self.LsrResultEmpty(35)
        segment2.cross_validated = MagicMock(side_effect=[ValidatedLsrResult(self.LsrResultEmpty(899), 899),
                                                          ValidatedLsrResult(segment2lsr, 10),
                                                          ValidatedLsrResult(self.LsrResultEmpty(991), 991)])

        r = compute([segment1, segment2], K_FOLD, POLYNOMIAL_DEGREE, UNKNOWN_FUNCTION)
        self.assertEqual(r.total_ss_error, 30 + 35)
        self.assertEqual(r.total_cv_error, 30 + 10)
        np.testing.assert_array_equal([segment1lsr, segment2lsr], r.lines)

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import results_match  # noqa: E402


def ok(gold, cand):
    return results_match(gold, cand)[0]


def test_identical():
    assert ok([(2024, 74266)], [(2024, 74266)])


def test_row_order_does_not_matter():
    assert ok([("N", 8.11), ("S", 6.91)], [("S", 6.91), ("N", 8.11)])


def test_column_order_does_not_matter():
    assert ok([(2024, 74266)], [(74266, 2024)])


def test_extra_columns_are_fine():
    assert ok([(74266,)], [(2024, 74266, "label")])


def test_small_rounding_differences_are_fine():
    assert ok([(189.1,)], [(189.1427,)])
    assert ok([(63.2,)], [(63.17,)])


def test_whole_numbers_must_be_exact():
    assert not ok([(164933,)], [(164932,)])
    assert ok([(74266,)], [(74266.0,)])


def test_wrong_value():
    assert not ok([(2024, 74266)], [(2022, 73094)])


def test_wrong_row_count():
    assert not ok([(1,), (2,)], [(1,)])
    assert not ok([(1,)], [(1,), (2,)])


def test_missing_column():
    assert not ok([(2024, 74266)], [(74266,)])


def test_strings_ignore_case_and_spaces():
    assert ok([("KENNEDY", 5)], [(" Kennedy ", 5)])


def test_nulls():
    assert ok([(None, 1)], [(None, 1)])
    assert not ok([(None, 1)], [(0, 1)])


def test_empty_results():
    assert ok([], [])
    assert not ok([], [(1,)])


def test_same_numbers_in_the_wrong_rows_do_not_match():
    # the right values attached to the wrong labels is a wrong answer
    assert not ok([("N", 8.11), ("S", 6.91)], [("N", 6.91), ("S", 8.11)])

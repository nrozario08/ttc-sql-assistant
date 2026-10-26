"""Decide whether a model's query result matches the reference result.

Two different queries can both be right, so the SQL text is never compared, only what the
queries return ("execution accuracy"). The comparison is forgiving about things that don't
change the answer:
  * row order
  * column order and column names
  * extra columns (a model may add a helpful label next to the number that was asked for)
  * small rounding differences in decimals
and strict about everything else (the values, and the number of rows).
"""
import itertools

MAX_CANDIDATE_COLUMNS = 8
FLOAT_TOLERANCE = 0.051      # allows for a different ROUND() than the reference used
INT_TOLERANCE = 0.0005       # counts and whole-number sums must match exactly


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _norm(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, str):
        return v.strip().casefold()
    return v


def _sort_key(row):
    key = []
    for v in row:
        v = _norm(v)
        if v is None:
            key.append((2, ""))
        elif _is_number(v):
            key.append((0, round(float(v))))
        else:
            key.append((1, str(v)))
    return tuple(key)


def _values_equal(gold, cand):
    gold, cand = _norm(gold), _norm(cand)
    if gold is None or cand is None:
        return gold is None and cand is None
    if _is_number(gold) and _is_number(cand):
        tol = INT_TOLERANCE if isinstance(gold, int) else FLOAT_TOLERANCE
        return abs(float(gold) - float(cand)) <= tol
    return gold == cand


def _rows_equal(gold_rows, cand_rows):
    g = sorted(gold_rows, key=_sort_key)
    c = sorted(cand_rows, key=_sort_key)
    return all(
        all(_values_equal(a, b) for a, b in zip(gr, cr))
        for gr, cr in zip(g, c)
    )


def results_match(gold_rows, cand_rows):
    """Return (matched, reason)."""
    if len(gold_rows) != len(cand_rows):
        return False, f"row count differs (expected {len(gold_rows)}, got {len(cand_rows)})"
    if not gold_rows:
        return True, "both empty"

    gold_width = len(gold_rows[0])
    cand_width = len(cand_rows[0])
    if cand_width < gold_width:
        return False, f"too few columns (expected {gold_width}, got {cand_width})"
    if cand_width > MAX_CANDIDATE_COLUMNS:
        return False, "too many columns"

    # try every way of lining up the reference columns with the candidate's columns
    for picked in itertools.permutations(range(cand_width), gold_width):
        projected = [tuple(row[i] for i in picked) for row in cand_rows]
        if _rows_equal(gold_rows, projected):
            return True, "match"
    return False, "values differ"

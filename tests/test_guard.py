import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guard import UnsafeSQL, check_sql, connect_readonly, run_query  # noqa: E402

DB = os.path.join(os.path.dirname(__file__), "..", "data", "ttc.db")


@pytest.fixture(scope="module")
def con():
    return connect_readonly(DB)


# ---- layer 1: the text checks ----

@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM delays",
    "select count(*) from delays;",
    "WITH t AS (SELECT year FROM delays) SELECT COUNT(*) FROM t",
    "SELECT 1 /* a comment ; with a semicolon */",
    "SELECT COUNT(*) FROM delays WHERE station = 'DROP ZONE'",   # keyword inside a string is fine
    "-- how many\nSELECT COUNT(*) FROM delays",
])
def test_allows_read_queries(sql):
    check_sql(sql)


@pytest.mark.parametrize("sql", [
    "",
    "   ",
    "DROP TABLE delays",
    "SELECT 1; DROP TABLE delays",
    "SELECT 1; SELECT 2",
    "DELETE FROM delays",
    "UPDATE delays SET min_delay = 0",
    "INSERT INTO delays VALUES (1)",
    "PRAGMA table_info(delays)",
    "ATTACH DATABASE 'x.db' AS x",
    "CREATE TABLE t (a)",
    "WITH t AS (SELECT 1) DELETE FROM delays",
    "EXPLAIN SELECT 1",
])
def test_rejects_unsafe_text(sql):
    with pytest.raises(UnsafeSQL):
        check_sql(sql)


# ---- layers 2 and 3: the connection itself ----

def test_runs_a_normal_query(con):
    cols, rows, truncated = run_query(con, "SELECT COUNT(*) AS n FROM delays")
    assert cols == ["n"] and rows[0][0] > 100_000 and not truncated


def test_blocks_tables_outside_the_allow_list(con):
    with pytest.raises(UnsafeSQL):
        run_query(con, "SELECT COUNT(*) FROM delays_raw")
    with pytest.raises(UnsafeSQL):
        run_query(con, "SELECT name FROM sqlite_master")


def test_connection_cannot_write(con):
    # bypass layer 1 on purpose: layers 2 and 3 must still stop it
    with pytest.raises(Exception):
        con.execute("DELETE FROM delays")


def test_row_cap(con):
    cols, rows, truncated = run_query(con, "SELECT * FROM delays", max_rows=5)
    assert len(rows) == 5 and truncated


def test_timeout(con):
    runaway = ("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) "
               "SELECT COUNT(*) FROM r")
    with pytest.raises(UnsafeSQL):
        run_query(con, runaway, timeout=1)


def test_blocked_function(con):
    with pytest.raises(UnsafeSQL):
        run_query(con, "SELECT load_extension('x')")

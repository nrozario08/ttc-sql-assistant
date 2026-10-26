import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from evaluate import OracleClient, load_questions, run_variant, summarize  # noqa: E402
from fakes import FakeClient, sql_reply  # noqa: E402
from guard import connect_readonly  # noqa: E402

DB = os.path.join(HERE, "..", "data", "ttc.db")


@pytest.fixture(scope="module")
def con():
    return connect_readonly(DB)


def test_reference_sql_scores_100_percent(con):
    """If answering with the reference queries doesn't score 100%, the harness is broken."""
    questions = load_questions()
    results = run_variant(questions, lambda: OracleClient(questions), con, "oracle", "full")
    assert len(results) == 30
    assert all(r["correct_final"] and r["correct_first_try"] for r in results)


def test_wrong_answers_score_zero(con):
    questions = load_questions({"q01", "q02", "q03"})
    wrong = FakeClient([sql_reply("SELECT 12345")] * 3)
    results = run_variant(questions, lambda: wrong, con, "fake", "full")
    assert not any(r["correct_final"] for r in results)


def test_first_try_vs_final(con):
    questions = load_questions({"q02"})
    client = FakeClient([sql_reply("SELECT COUNT(*) FROM delays WHERE yr = 2023"),
                         sql_reply("SELECT COUNT(*) FROM delays WHERE year = 2023")])
    (r,) = run_variant(questions, lambda: client, con, "fake", "full")
    assert not r["correct_first_try"] and r["correct_final"] and r["attempts"] == 2


def test_cache_is_used_on_the_second_run(con, tmp_path):
    questions = load_questions({"q01", "q02"})
    first = run_variant(questions, lambda: OracleClient(questions), con, "oracle", "full",
                        cache_dir=str(tmp_path))
    assert len(list(tmp_path.iterdir())) == 2

    def no_client():
        raise AssertionError("the model should not be called when everything is cached")

    second = run_variant(questions, no_client, con, "oracle", "full", cache_dir=str(tmp_path))
    assert [r["correct_final"] for r in first] == [r["correct_final"] for r in second]


def test_summary_renders(con):
    questions = load_questions({"q01", "q02"})
    wrong = FakeClient([sql_reply("SELECT 1")] * 2)
    results = run_variant(questions, lambda: wrong, con, "claude-opus-5-5", "full")
    report = summarize({("claude-opus-5-5", "full"): results})
    assert "| claude-opus-5-5 | full |" in report and "answered wrong" in report

import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from assistant import ask, extract_sql  # noqa: E402
from fakes import FakeClient, sql_reply  # noqa: E402
from guard import connect_readonly  # noqa: E402
from prompts import EXAMPLES, build_system_prompt  # noqa: E402

DB = os.path.join(HERE, "..", "data", "ttc.db")


@pytest.fixture(scope="module")
def con():
    return connect_readonly(DB)


def test_extract_sql_takes_the_last_block():
    text = "first\n```sql\nSELECT 1\n```\nactually\n```sql\nSELECT 2\n```"
    assert extract_sql(text) == "SELECT 2"
    assert extract_sql("no code here") == ""


def test_good_answer_on_the_first_try(con):
    client = FakeClient([sql_reply("SELECT COUNT(*) FROM delays WHERE year = 2023")])
    a = ask("How many incidents in 2023?", client, con)
    assert a.error == "" and a.attempts == 1
    assert a.rows == [(22943,)]
    assert a.interpretation == "Here is how I read it."
    assert (a.input_tokens, a.output_tokens) == (100, 20)


def test_bad_column_triggers_one_retry_that_fixes_it(con):
    client = FakeClient([
        sql_reply("SELECT COUNT(*) FROM delays WHERE yr = 2023"),
        sql_reply("SELECT COUNT(*) FROM delays WHERE year = 2023"),
    ])
    a = ask("How many incidents in 2023?", client, con)
    assert a.attempts == 2 and a.error == ""
    assert a.first_try_error.startswith("SQLite error")
    assert a.first_try_sql != a.sql
    # the second request carries the error back to the model
    assert "did not work" in client.calls[1]["messages"][-1]["content"]


def test_unsafe_sql_is_rejected_and_never_run(con):
    client = FakeClient([sql_reply("DROP TABLE delays"), sql_reply("DELETE FROM delays")])
    a = ask("Please clear the table", client, con)
    assert a.attempts == 2
    assert a.error.startswith("query rejected")
    assert con.execute("SELECT COUNT(*) FROM delays").fetchone()[0] > 100_000


def test_reply_without_a_code_block(con):
    client = FakeClient(["I think the answer is 42.", "Still no code."])
    a = ask("anything", client, con)
    assert "code block" in a.error


def test_no_retry_option(con):
    client = FakeClient([sql_reply("SELECT nope FROM delays")])
    a = ask("x", client, con, retry=False)
    assert a.attempts == 1 and a.error and len(client.calls) == 1


def test_prompt_variants(con):
    base = build_system_prompt(con, "baseline")
    full = build_system_prompt(con, "full")
    assert "Line 1 Yonge-University" in base and "Rules for this dataset" not in base
    assert "Rules for this dataset" in full and "Examples:" in full
    assert len(full) > len(base)
    with pytest.raises(ValueError):
        build_system_prompt(con, "other")


def test_examples_do_not_leak_benchmark_questions():
    with open(os.path.join(HERE, "..", "eval", "questions.json")) as f:
        benchmark = {q["question"].strip().lower() for q in json.load(f)}
    for question, _sql in EXAMPLES:
        assert question.strip().lower() not in benchmark


def test_example_sql_actually_runs(con):
    from guard import run_query
    for _question, sql in EXAMPLES:
        run_query(con, sql)

# Plain-English Questions on TTC Delay Data

An assistant that answers questions about Toronto subway delays written in plain English. A language model turns the question into a SQL query, a safety layer checks and runs it against the database, and a 30-question benchmark measures how often the answers are right. Built on the same City of Toronto open data as [ttc-subway-delays](https://github.com/nrozario08/ttc-subway-delays) and [ttc-delay-prediction](https://github.com/nrozario08/ttc-delay-prediction).

## Why this project

The first two projects in this series needed SQL to get at the answers: which station loses the most time, how a line compares with last year, what a typical Monday looks like. Someone who rides the subway every day has the same questions but not the SQL. Letting them ask in their own words is the obvious next step.

The harder question is whether the answers can be trusted. A query can run without errors and still answer a different question than the one asked, for example by counting 2026 as a full year when it only runs to August, or by averaging in the zero-minute incidents. So most of this project is about measuring correctness and keeping the database safe, not about the model call itself.

## How it works

```
question --> model writes SQL --> guard checks and runs it (read-only) --> result
                   ^                                   |
                   +---- error message, one retry -----+
```

1. **Prompt.** The model sees the table schema, what each column means, the valid line and station names, rules for this dataset, and (in the `full` prompt) five worked examples.
2. **Guard** ([src/guard.py](src/guard.py)). Three independent layers:
   - a text check that allows a single `SELECT` or `WITH` statement and nothing else;
   - a read-only database connection;
   - a SQLite authorizer that only lets queries read the `delays`, `station_dim` and `delay_codes` tables, so the raw data and the schema tables are off limits.

   Queries also have a 10-second time limit and a 200-row cap.
3. **Retry.** If the query fails or is rejected, the error goes back to the model for one more attempt.

## Benchmark

[eval/questions.json](eval/questions.json) holds 30 questions (8 easy, 13 medium, 9 hard), each with a hand-written reference query. Several are built around mistakes the data invites:

- 2026 only covers January to August, so year-over-year comparisons need care.
- Averages over delay minutes should skip incidents with zero delay.
- Some station rows are not real stations (the yard, the system as a whole) and have to be excluded when ranking stations.
- Rankings are written so that there are no ties at the cutoff.

**Scoring compares results, not SQL text.** A candidate query is correct when it returns the same result as the reference query. Row order, column order, extra columns and rounding are tolerated; wrong values are not. Two numbers are reported for each run: accuracy on the first attempt and accuracy after one retry.

**Checking the checker.** Answering every question with its own reference query must score 30/30, and wrong answers must score 0. Both are covered by tests, and `python src/evaluate.py --oracle` runs the first check.

## Status

The pipeline, guard, benchmark and scorer are built and tested (56 tests). Results from running language models against the benchmark have not been recorded yet. They will be added here, with the failures grouped by type, once the runs are complete.

## Run it

```bash
pip install -r requirements.txt
python src/download_data.py        # fetch the open data files
python src/build_db.py             # build data/ttc.db
python -m pytest tests -q          # no API key needed
python src/evaluate.py --oracle    # harness check, should print 30/30
```

Ask a question (set `ANTHROPIC_API_KEY` for Claude models or `GEMINI_API_KEY` for Gemini models):

```bash
python src/assistant.py "Which station had the most delay hours in 2024?" --model gemini-3.8-flash
```

Run the benchmark:

```bash
python src/evaluate.py --models gemini-3.8-flash gemini-3.5-flash-lite --variants baseline full
```

Every model reply is cached in `eval/cache/`, so re-running reproduces the same scores without an API key. Use `--refresh` to ask the models again. Summaries are written to `results/summary.md`.

## Project layout

```
src/guard.py          query safety checks and read-only execution
src/prompts.py        system prompt (baseline and full variants)
src/assistant.py      question -> SQL -> result, with one retry
src/gemini_client.py  lets the same code call Gemini models
src/scoring.py        compares query results against the reference
src/evaluate.py       runs the benchmark and writes the summary
eval/questions.json   the 30 benchmark questions and reference SQL
sql/01_clean.sql      cleaning rules that build the delays table
tests/                tests for the guard, scoring, assistant and harness
```

## Limitations

- One dataset and one database engine (SQLite), so conclusions do not automatically carry over to other schemas.
- 30 questions is enough to see clear differences between prompts and models but not small ones.
- The delay data records what was reported, so the answers inherit its gaps and the 2026 partial year.

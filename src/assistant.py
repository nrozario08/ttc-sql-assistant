"""Ask questions about the TTC delay data in plain English.

    python src/assistant.py "Which station had the most delay hours in 2024?"

The model writes the SQL, guard.py checks it and runs it read-only, and if the query
fails the error goes back to the model for one more try.
"""
import argparse
import os
import re
import sqlite3
from dataclasses import dataclass, field

from guard import UnsafeSQL, connect_readonly, run_query
from prompts import build_system_prompt

DB_PATH = os.path.join("data", "ttc.db")
DEFAULT_MODEL = "claude-opus-5-5"
SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


@dataclass
class Answer:
    question: str
    model: str
    variant: str
    interpretation: str = ""
    sql: str = ""
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    truncated: bool = False
    error: str = ""                 # empty when the final query ran fine
    attempts: int = 0
    first_try_sql: str = ""
    first_try_error: str = ""       # empty when the first query ran fine
    input_tokens: int = 0
    output_tokens: int = 0


def extract_sql(text):
    """Pull the SQL out of the model's reply (the last code block wins)."""
    blocks = SQL_BLOCK.findall(text)
    return blocks[-1].strip() if blocks else ""


def reply_text(response):
    return "".join(b.text for b in response.content if b.type == "text")


def make_client():
    import anthropic
    return anthropic.Anthropic()


def ask(question, client, con, model=DEFAULT_MODEL, variant="full", retry=True, effort="medium"):
    system = build_system_prompt(con, variant)
    messages = [{"role": "user", "content": question}]
    answer = Answer(question=question, model=model, variant=variant)

    for attempt in range(1, (2 if retry else 1) + 1):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=system,
            messages=messages,
            output_config={"effort": effort},
        )
        answer.attempts = attempt
        answer.input_tokens += response.usage.input_tokens
        answer.output_tokens += response.usage.output_tokens

        text = reply_text(response)
        answer.interpretation = SQL_BLOCK.split(text)[0].strip()
        answer.sql = extract_sql(text)
        error = _try_query(con, answer)
        if attempt == 1:
            answer.first_try_sql, answer.first_try_error = answer.sql, error
        answer.error = error
        if not error:
            break

        # hand the error back and let the model fix it (pass its reply back unchanged)
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"That query did not work: {error}\nFix it and reply in the same format."},
        ]
    return answer


def _try_query(con, answer):
    """Run answer.sql, fill in the result, and return an error message ('' if fine)."""
    if not answer.sql:
        return "the reply did not contain a ```sql code block"
    try:
        answer.columns, answer.rows, answer.truncated = run_query(con, answer.sql)
        return ""
    except UnsafeSQL as e:
        return f"query rejected: {e}"
    except sqlite3.Error as e:
        return f"SQLite error: {e}"


def print_answer(a, max_rows=20):
    if a.interpretation:
        print(a.interpretation, "\n")
    print(a.sql, "\n")
    if a.error:
        print("Error:", a.error)
        return
    print(" | ".join(a.columns))
    for row in a.rows[:max_rows]:
        print(" | ".join("" if v is None else str(v) for v in row))
    if len(a.rows) > max_rows or a.truncated:
        print("... (more rows not shown)")


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the TTC delay data.")
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--variant", choices=["baseline", "full"], default="full")
    parser.add_argument("--no-retry", action="store_true")
    args = parser.parse_args()

    con = connect_readonly(DB_PATH)
    answer = ask(args.question, make_client(), con, model=args.model,
                 variant=args.variant, retry=not args.no_retry)
    print_answer(answer)


if __name__ == "__main__":
    main()

"""Run the benchmark and score the assistant.

    python src/evaluate.py --models gemini-3.8-flash gemini-3.5-flash-lite --variants baseline full

Model names starting with "gemini" use GEMINI_API_KEY; "claude-..." models use ANTHROPIC_API_KEY.

Every model reply is saved in eval/cache/, so a second run (or anyone else cloning the repo)
reproduces the same results without an API key or any cost. Use --refresh to ask the model again.

    python src/evaluate.py --oracle        sanity-check the harness: answers every question
                                           with the reference SQL, so it must score 100%
"""
import argparse
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

from assistant import ask, make_client
from guard import connect_readonly, run_query
from scoring import results_match

DB_PATH = os.path.join("data", "ttc.db")
QUESTIONS_PATH = os.path.join("eval", "questions.json")
CACHE_DIR = os.path.join("eval", "cache")
RESULTS_DIR = "results"

# US dollars per million tokens (input, output), from the published paid-tier price lists.
# Gemini's free tier costs nothing, so these show what the same run would cost on the paid tier.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
}


def load_questions(only=None):
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    if only:
        questions = [q for q in questions if q["id"] in only]
    return questions


def grade(con, gold_rows, sql):
    """Run a query and compare it with the reference result. Returns (correct, reason)."""
    if not sql:
        return False, "no SQL"
    try:
        _cols, rows, _truncated = run_query(con, sql)
    except Exception as e:  # noqa: BLE001 - any failure just means "wrong"
        return False, f"error: {e}"
    return results_match(gold_rows, rows)


class OracleClient:
    """Answers each question with its reference SQL. Only used to check the harness itself."""

    def __init__(self, questions):
        self._by_question = {q["question"]: q["gold_sql"] for q in questions}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        question = kwargs["messages"][0]["content"]
        text = f"Reference answer.\n\n```sql\n{self._by_question[question]}\n```"
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        )


def run_variant(questions, client_factory, con, model, variant, cache_dir=None, refresh=False):
    """Ask (or load from the cache) every question for one model + prompt variant."""
    results = []
    client = None
    for q in questions:
        cache_file = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{q['id']}.json")

        if cache_file and os.path.exists(cache_file) and not refresh:
            with open(cache_file) as f:
                answer = json.load(f)
        else:
            client = client or client_factory()
            answer = asdict(ask(q["question"], client, con, model=model, variant=variant))
            answer["rows"], answer["columns"] = [], []      # results are re-run when scoring
            if cache_file:
                with open(cache_file, "w") as f:
                    json.dump(answer, f, indent=1)

        _gold_cols, gold_rows, _ = run_query(con, q["gold_sql"])
        first_ok, first_reason = grade(con, gold_rows, answer["first_try_sql"])
        final_ok, final_reason = grade(con, gold_rows, answer["sql"])
        results.append({
            "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
            "model": model, "variant": variant,
            "correct_first_try": first_ok, "correct_final": final_ok,
            "attempts": answer["attempts"],
            "first_try_reason": first_reason, "final_reason": final_reason,
            "first_try_sql": answer["first_try_sql"], "final_sql": answer["sql"],
            "gold_sql": q["gold_sql"], "trap": q.get("trap", ""),
            "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"],
        })
    return results


def cost(model, results):
    price_in, price_out = PRICES.get(model, (None, None))
    if price_in is None:
        return None
    tokens_in = sum(r["input_tokens"] for r in results)
    tokens_out = sum(r["output_tokens"] for r in results)
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def summarize(all_results):
    """Markdown report for {(model, variant): [result, ...]}."""
    lines = ["# Benchmark results", ""]
    n_questions = len(next(iter(all_results.values())))
    lines += [f"{n_questions} questions per run. A query counts as correct when it returns the same "
              "result as the reference query.", ""]

    lines += ["## Accuracy", "",
              "| Model | Prompt | First try | After one retry | Easy | Medium | Hard | Avg tokens (in/out) | Cost |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (model, variant), res in all_results.items():
        by = {d: [r for r in res if r["difficulty"] == d] for d in ("easy", "medium", "hard")}
        n = len(res)
        c = cost(model, res)
        lines.append(
            f"| {model} | {variant} | {pct(sum(r['correct_first_try'] for r in res), n)} "
            f"| **{pct(sum(r['correct_final'] for r in res), n)}** "
            + "".join(f"| {pct(sum(r['correct_final'] for r in by[d]), len(by[d]))} " for d in by)
            + f"| {sum(r['input_tokens'] for r in res) // n:,} / {sum(r['output_tokens'] for r in res) // n:,} "
            f"| {'$%.2f' % c if c is not None else '-'} |"
        )

    lines += ["", "## Questions that were answered wrong", ""]
    any_failure = False
    for (model, variant), res in all_results.items():
        failed = [r for r in res if not r["correct_final"]]
        if not failed:
            continue
        any_failure = True
        lines += [f"### {model}, {variant} prompt ({len(failed)} wrong)", ""]
        for r in failed:
            lines += [f"**{r['id']} ({r['difficulty']})**: {r['question']}", "",
                      f"- Result: {r['final_reason']}",
                      f"- Model SQL: `{' '.join(r['final_sql'].split())}`",
                      f"- Reference SQL: `{' '.join(r['gold_sql'].split())}`", ""]
    if not any_failure:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["claude-opus-5-5"])
    parser.add_argument("--variants", nargs="+", default=["baseline", "full"], choices=["baseline", "full"])
    parser.add_argument("--only", help="comma-separated question ids, e.g. q01,q07")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and ask the model again")
    parser.add_argument("--oracle", action="store_true", help="harness check using the reference SQL")
    args = parser.parse_args()

    con = connect_readonly(DB_PATH)
    questions = load_questions(set(args.only.split(",")) if args.only else None)

    if args.oracle:
        res = run_variant(questions, lambda: OracleClient(questions), con, "oracle", "full")
        ok = sum(r["correct_final"] for r in res)
        print(f"oracle: {ok}/{len(res)} correct (should be {len(res)}/{len(res)})")
        return

    all_results = {}
    for model in args.models:
        for variant in args.variants:
            cache = os.path.join(CACHE_DIR, f"{model}__{variant}")
            print(f"running {model} / {variant} ...")
            all_results[(model, variant)] = run_variant(
                questions, lambda: make_client(model), con, model, variant,
                cache_dir=cache, refresh=args.refresh)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump([r for res in all_results.values() for r in res], f, indent=1)
    report = summarize(all_results)
    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()

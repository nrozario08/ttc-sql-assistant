"""Builds the system prompt for the SQL assistant.

Two variants, so the benchmark can measure what the extra guidance is actually worth:

  "baseline"  the table and column descriptions only
  "full"      the same, plus business rules for this dataset and worked examples
"""

INSTRUCTIONS = """You answer questions about Toronto subway (TTC) delays by writing SQLite queries.

Reply in this format and nothing else:
1. One short sentence saying how you interpret the question.
2. Exactly one ```sql code block holding a single SELECT query (it may start with WITH).

Only use the tables described below. Write plain SQLite (no schema prefixes, no stored procedures)."""

COLUMN_DOCS = """Table delays: one row per logged incident (cleaned data)
  delay_date     TEXT     date as 'YYYY-MM-DD'
  year           INTEGER
  month          INTEGER  1 to 12
  hour           INTEGER  0 to 23, hour the incident was logged
  weekday        TEXT     'Monday' ... 'Sunday'
  is_weekend     INTEGER  1 for Saturday/Sunday, else 0
  time_of_day    TEXT     'HH:MM'
  station        TEXT     cleaned station name, upper case, without the word STATION (for example 'KENNEDY', 'ST GEORGE')
  line           TEXT     one of the values listed below
  direction      TEXT     'N', 'S', 'E', 'W', or NULL when none was recorded
  code           TEXT     delay code, for example 'SUDP'
  code_description TEXT   upper-case description of the code ('UNKNOWN CODE' if the code has no description)
  category       TEXT     one of the values listed below
  min_delay      INTEGER  minutes of delay caused by the incident (0 when the incident was logged but no delay was recorded)
  min_gap        INTEGER  minutes between trains
  caused_delay   INTEGER  1 when min_delay > 0, else 0
  vehicle        INTEGER  vehicle number

Table station_dim: the list of real stations
  station          TEXT     same naming as delays.station
  total_incidents  INTEGER  incidents logged at that station across all years

Table delay_codes: the delay code lookup
  code         TEXT
  description  TEXT"""

NOTES = """Rules for this dataset:
- The data covers 2019-01-01 to 2026-08-31. 2026 is a partial year (January to August only), so do not compare 2026 with a full year unless the question is about matching months.
- "Delay minutes" or "delay hours" means SUM(min_delay); hours = minutes / 60.0.
- About 62% of incidents have min_delay = 0. When asked for the average delay (or how long delays last), average only incidents that caused a delay (min_delay > 0), unless the question says to include all incidents.
- The station column also contains things that are not stations (yards, whole-line entries, track segments). When a question is about stations, restrict to real stations with: station IN (SELECT station FROM station_dim) or a join to station_dim.
- Use the exact line and category strings listed below.
- When a question asks for a rate, share or percentage, compute it from the right denominator and round as the question asks."""

EXAMPLES = [
    ("How many incidents were logged on Line 1 in 2022?",
     "SELECT COUNT(*) FROM delays WHERE line = 'Line 1 Yonge-University' AND year = 2022"),
    ("What is the average delay in minutes for each weekday, counting only incidents that caused a delay?",
     "SELECT weekday, ROUND(AVG(min_delay), 2) FROM delays WHERE min_delay > 0 GROUP BY weekday"),
    ("Which 3 real stations had the most delay hours in 2022?",
     "SELECT d.station, ROUND(SUM(d.min_delay) / 60.0, 1) AS hours FROM delays d "
     "JOIN station_dim s ON s.station = d.station WHERE d.year = 2022 "
     "GROUP BY d.station ORDER BY hours DESC LIMIT 3"),
    ("What share of 2021 incidents were logged between 6am and 9am?",
     "SELECT ROUND(100.0 * SUM(hour BETWEEN 6 AND 9) / COUNT(*), 1) FROM delays WHERE year = 2021"),
    ("For each year from 2019 to 2025, how many incidents were there compared with the year before?",
     "WITH y AS (SELECT year, COUNT(*) AS n FROM delays WHERE year BETWEEN 2019 AND 2025 GROUP BY year) "
     "SELECT year, n, n - LAG(n) OVER (ORDER BY year) AS change FROM y"),
]


def example_block():
    parts = []
    for question, sql in EXAMPLES:
        parts.append(f"Question: {question}\n```sql\n{sql}\n```")
    return "Examples:\n\n" + "\n\n".join(parts)


def enumerations(con):
    """Allowed values for the low-cardinality columns, read from the database so they never drift."""
    lines = [r[0] for r in con.execute("SELECT DISTINCT line FROM delays ORDER BY line")]
    cats = [r[0] for r in con.execute("SELECT DISTINCT category FROM delays ORDER BY category")]
    lo, hi = con.execute("SELECT MIN(delay_date), MAX(delay_date) FROM delays").fetchone()
    return (
        "Values:\n"
        f"  line: {', '.join(repr(v) for v in lines)}\n"
        f"  category: {', '.join(repr(v) for v in cats)}\n"
        f"  dates: {lo} to {hi}"
    )


def build_system_prompt(con, variant):
    if variant not in ("baseline", "full"):
        raise ValueError(f"unknown prompt variant: {variant}")
    parts = [INSTRUCTIONS, COLUMN_DOCS, enumerations(con)]
    if variant == "full":
        parts += [NOTES, example_block()]
    return "\n\n".join(parts)

"""Safety checks for SQL written by the model.

The assistant only ever needs to read three tables, so it gets three layers of protection:

1. check_sql():  static checks on the text (one SELECT/WITH statement, no write keywords)
2. read-only connection (mode=ro), so even a bug in layer 1 can't change the database
3. an authorizer that only lets SQLite read the allow-listed tables and call harmless functions

A query timeout and a row cap keep a bad query from hanging the app.
"""
import re
import sqlite3
import time

ALLOWED_TABLES = {"delays", "station_dim", "delay_codes"}

FORBIDDEN_WORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE", "TRIGGER", "SAVEPOINT",
    "BEGIN", "COMMIT", "ROLLBACK",
}

BLOCKED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

MAX_ROWS = 200
TIMEOUT_SECONDS = 10


class UnsafeSQL(ValueError):
    """The query was rejected before (or while) running."""


def _strip_comments_and_strings(sql):
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    # string literals, with '' as an escaped quote inside them
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    sql = re.sub(r'"(?:[^"]|"")*"', '""', sql)
    return sql


def check_sql(sql):
    """Raise UnsafeSQL if the text is not a single read-only SELECT. Returns the cleaned SQL."""
    if not sql or not sql.strip():
        raise UnsafeSQL("empty query")

    cleaned = sql.strip()
    body = _strip_comments_and_strings(cleaned).strip()
    if body.endswith(";"):
        body = body[:-1].strip()
        cleaned = cleaned.rstrip().rstrip(";").rstrip()
    if ";" in body:
        raise UnsafeSQL("only one statement is allowed")

    words = re.findall(r"[A-Za-z_]+", body)
    if not words or words[0].upper() not in ("SELECT", "WITH"):
        raise UnsafeSQL("only SELECT queries are allowed")

    bad = sorted({w.upper() for w in words} & FORBIDDEN_WORDS)
    if bad:
        raise UnsafeSQL(f"not allowed: {', '.join(bad)}")
    return cleaned


def _authorizer(action, arg1, arg2, db_name, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        # arg1 is the table name, arg2 the column
        return sqlite3.SQLITE_OK if arg1 in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or "").lower() in BLOCKED_FUNCTIONS else sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def connect_readonly(db_path):
    """Open the database read-only with the authorizer installed."""
    path = db_path.replace("\\", "/")
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.set_authorizer(_authorizer)
    return con


def run_query(con, sql, max_rows=MAX_ROWS, timeout=TIMEOUT_SECONDS):
    """Check and run a query. Returns (columns, rows, truncated)."""
    sql = check_sql(sql)

    deadline = time.monotonic() + timeout
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as e:
        msg = str(e).lower()
        if "interrupted" in msg:
            raise UnsafeSQL(f"query took longer than {timeout} seconds") from e
        # the authorizer reports denials as "not authorized" / "prohibited"
        if "not authorized" in msg or "prohibited" in msg:
            raise UnsafeSQL(str(e)) from e
        raise    # ordinary SQL mistakes (bad column name etc.) stay as they are
    finally:
        con.set_progress_handler(None, 0)

    truncated = len(rows) > max_rows
    return columns, rows[:max_rows], truncated

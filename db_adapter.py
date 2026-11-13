"""
LearnSphere Cloud Database Adapter
Supports PostgreSQL (Supabase, Neon, Railway Postgres, AWS RDS)
and local SQLite fallback for development.
"""
import os
import re
import ssl
import sqlite3
from urllib.parse import urlparse, unquote

try:
    import pg8000.dbapi
    HAS_PG = True
except ImportError:
    HAS_PG = False

# Default SQLite path (use DB_PATH env var if provided)
DB_PATH = os.environ.get('DB_PATH', os.path.join(os.path.dirname(__file__), 'learnsphere.db'))

class PgRow(dict):
    """
    Dict-compatible and index-compatible row object matching sqlite3.Row behavior.
    Supports row['col'], row[0], row.get('col'), and dict(row).
    """
    def __init__(self, col_names, values):
        super().__init__(zip(col_names, values))
        self._values = tuple(values)
        self._col_names = col_names

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        return super().__getitem__(key)


def translate_sql(sql):
    """
    Translates SQLite syntax to PostgreSQL:
    1. datetime('now') -> CURRENT_TIMESTAMP
    2. DEFAULT "balanced" -> DEFAULT 'balanced'
    3. ? placeholders -> %s (ignoring ? inside string literals)
    4. INSERT OR IGNORE -> INSERT INTO ... ON CONFLICT DO NOTHING
    5. INSERT OR REPLACE INTO course_ratings -> INSERT INTO ... ON CONFLICT (course_id, student_id) DO UPDATE ...
    6. ALTER TABLE ... ADD COLUMN ... -> ALTER TABLE ... ADD COLUMN IF NOT EXISTS ...
    """
    if not sql:
        return sql

    # 1. Replace datetime('now')
    sql = re.sub(r"\bdatetime\('now'\)", "CURRENT_TIMESTAMP", sql, flags=re.IGNORECASE)

    # 2. Fix double-quoted string defaults in CREATE TABLE
    sql = re.sub(r'DEFAULT\s+"([^"]+)"', r"DEFAULT '\1'", sql)

    # 3. Add IF NOT EXISTS to ALTER TABLE ADD COLUMN
    if re.search(r"\bALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(?!IF\s+NOT\s+EXISTS\b)", sql, re.IGNORECASE):
        sql = re.sub(
            r"\bALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+",
            r"ALTER TABLE \1 ADD COLUMN IF NOT EXISTS ",
            sql,
            flags=re.IGNORECASE
        )

    # 4. Replace ? placeholders with %s outside single-quoted strings
    pattern = re.compile(r"'[^']*'")
    parts = []
    last = 0
    for m in pattern.finditer(sql):
        chunk = sql[last:m.start()].replace('?', '%s')
        parts.append(chunk)
        parts.append(m.group(0))
        last = m.end()
    chunk = sql[last:].replace('?', '%s')
    parts.append(chunk)
    sql = "".join(parts)

    # 5. Handle SQLite-specific INSERT variants
    if re.search(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", sql, re.IGNORECASE):
        sql = re.sub(r"\bINSERT\s+OR\s+IGNORE\s+INTO\b", "INSERT INTO", sql, flags=re.IGNORECASE)
        if "ON CONFLICT" not in sql.upper():
            sql += " ON CONFLICT DO NOTHING"

    if re.search(r"\bINSERT\s+OR\s+REPLACE\s+INTO\s+course_ratings\b", sql, re.IGNORECASE):
        sql = re.sub(r"\bINSERT\s+OR\s+REPLACE\s+INTO\b", "INSERT INTO", sql, flags=re.IGNORECASE)
        if "ON CONFLICT" not in sql.upper():
            sql += " ON CONFLICT (course_id, student_id) DO UPDATE SET rating = EXCLUDED.rating, review = EXCLUDED.review"

    return sql


class PgCursor:
    def __init__(self, raw_cursor, conn):
        self._cur = raw_cursor
        self._conn = conn

    @property
    def description(self):
        return self._cur.description

    def execute(self, sql, params=None):
        pg_sql = translate_sql(sql)
        try:
            if params is None:
                self._cur.execute(pg_sql)
            else:
                self._cur.execute(pg_sql, tuple(params))
            return self
        except Exception as e:
            try:
                self._conn.rollback()
            except Exception:
                pass
            err_msg = str(e).lower()
            if "unique" in err_msg or "duplicate key" in err_msg or "integrity" in err_msg:
                raise sqlite3.IntegrityError(str(e))
            raise

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        col_names = [col[0].lower() for col in self._cur.description]
        return PgRow(col_names, row)

    def fetchall(self):
        rows = self._cur.fetchall()
        if not rows:
            return []
        col_names = [col[0].lower() for col in self._cur.description]
        return [PgRow(col_names, r) for r in rows]

    def close(self):
        try:
            self._cur.close()
        except Exception:
            pass


class PgConnection:
    def __init__(self, raw_conn):
        self._conn = raw_conn
        self.row_factory = None

    def cursor(self):
        return PgCursor(self._conn.cursor(), self._conn)

    def execute(self, sql, params=None):
        cur = self.cursor()
        return cur.execute(sql, params)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        try:
            self._conn.rollback()
        except Exception:
            pass

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def connect_postgres(url):
    """
    Connect to PostgreSQL using pg8000 with SSL enabled for cloud providers.
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    parsed = urlparse(url)

    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    database = parsed.path.lstrip("/")

    # Enable SSL for cloud databases (Supabase, Neon, Render, etc.)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    raw_conn = pg8000.dbapi.connect(
        user=user,
        password=password,
        host=host,
        port=port,
        database=database,
        ssl_context=ssl_context
    )
    raw_conn.autocommit = False
    return PgConnection(raw_conn)


def get_db():
    """
    Returns active database connection:
    - If DATABASE_URL / POSTGRES_URL / SUPABASE_DB_URL is set, connects to PostgreSQL.
    - Otherwise, falls back to local SQLite (learnsphere.db).
    """
    url = (
        os.environ.get('DATABASE_URL') or 
        os.environ.get('POSTGRES_URL') or 
        os.environ.get('SUPABASE_DB_URL') or 
        ''
    ).strip()

    if url and HAS_PG:
        return connect_postgres(url)

    # SQLite fallback
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

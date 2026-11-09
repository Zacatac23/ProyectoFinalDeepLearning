"""Usage log for the API: Postgres when DATABASE_URL is set, a local CSV otherwise.

Render's free plan erases local files when the service sleeps, so production
stores each request in Postgres (Supabase).
"""
import csv
import io
import threading
from datetime import timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import psycopg
from psycopg.rows import dict_row

LOG_FIELDS = ['timestamp_utc', 'model', 'score', 'decision', 'latency_ms', 'history_len', 'warnings']

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS public.api_requests (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    model TEXT NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    decision TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    history_len INTEGER NOT NULL,
    warnings INTEGER NOT NULL
)
"""
# Supabase publishes the public schema through its REST API; RLS without policies closes that door.
ENABLE_RLS = 'ALTER TABLE public.api_requests ENABLE ROW LEVEL SECURITY'
INSERT_ROW = """
INSERT INTO public.api_requests (timestamp_utc, model, score, decision, latency_ms, history_len, warnings)
VALUES (%(timestamp_utc)s, %(model)s, %(score)s, %(decision)s, %(latency_ms)s, %(history_len)s, %(warnings)s)
"""
SELECT_ROWS = """
SELECT timestamp_utc, model, score, decision, latency_ms, history_len, warnings
FROM public.api_requests ORDER BY id
"""


class RequestLog(Protocol):
    def write(self, row: Dict[str, Any]) -> None: ...

    def read(self) -> List[Dict[str, Any]]: ...


class CsvRequestLog:
    def __init__(self, log_dir: Path):
        self.path = Path(log_dir) / 'requests.csv'
        self.lock = threading.Lock()

    def write(self, row: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            new = not self.path.exists()
            with self.path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
                if new:
                    writer.writeheader()
                writer.writerow(row)

    def read(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))


class PostgresRequestLog:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.table_ready = False

    def _connect(self) -> psycopg.Connection:
        connection = psycopg.connect(self.database_url, connect_timeout=5)
        if not self.table_ready:  # done here so the API can start while the database is unavailable
            connection.execute(CREATE_TABLE)
            connection.execute(ENABLE_RLS)
            connection.commit()
            self.table_ready = True
        return connection

    def write(self, row: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(INSERT_ROW, row)

    def read(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                rows = cursor.execute(SELECT_ROWS).fetchall()
        for row in rows:
            row['timestamp_utc'] = row['timestamp_utc'].astimezone(timezone.utc).isoformat(timespec='seconds')
        return rows


def build_request_log(database_url: Optional[str], log_dir: Path) -> RequestLog:
    return PostgresRequestLog(database_url) if database_url else CsvRequestLog(log_dir)


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=LOG_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()

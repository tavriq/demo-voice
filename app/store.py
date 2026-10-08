"""SQLite: conversations per client per hour, the daily budget in rubles, conversations for 30 days.

Budget flow, as in demo-arena: ``reserve`` checks that today's spend + the step's worst case fit
the daily budget and records the worst case; ``settle`` replaces it with the real cost.

Client keys (IPv4 or IPv6 /64) are stored only as HMAC hashes with a salt that lives in memory
and changes on every start, and only for the hourly limit. Conversations keep the card and the
client's words with contacts masked (app.pii), no client key, and are deleted after
``retention_days``. Audio is never written here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

HOUR = 3600
MSK = timezone(timedelta(hours=3))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS starts (ip_hash TEXT NOT NULL, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS starts_ip_ts ON starts (ip_hash, ts);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    day TEXT NOT NULL,
    ts REAL NOT NULL,
    rub REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spend_day ON spend (day);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    updated REAL NOT NULL,
    state TEXT NOT NULL,
    triage TEXT,
    cost_rub REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS conversations_ts ON conversations (ts);
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str | None = None
    message: str | None = None
    reservation_id: int | None = None
    retry_after_s: int | None = None


def _rub(value: float) -> str:
    return f"{value:.0f} ₽" if value == int(value) else f"{value:.2f} ₽".replace(".", ",")


class Store:
    def __init__(self, db_path: str | Path, daily_budget_rub: float, conversations_per_hour: int,
                 retention_days: int = 30, clock: Callable[[], float] = time.time):
        self.db_path = Path(db_path)
        self.daily_budget_rub = daily_budget_rub
        self.conversations_per_hour = conversations_per_hour
        self.retention_s = retention_days * 86400
        self.clock = clock
        self._salt = secrets.token_bytes(16)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.execute("DELETE FROM starts")  # hashes with an old salt only keep personal data around
        self.purge()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def day(self, ts: float | None = None) -> str:
        """Budget day in Moscow time: the page says "до 00:00 МСК"."""
        return datetime.fromtimestamp(self.clock() if ts is None else ts, tz=MSK).date().isoformat()

    def _hash(self, key: str) -> str:
        return hmac.new(self._salt, key.encode(), hashlib.sha256).hexdigest()

    # ---------- limits ----------

    def start(self, client_key: str) -> Decision:
        """A new conversation: at most ``conversations_per_hour`` per client."""
        now = self.clock()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM starts WHERE ts <= ?", (now - HOUR,))
            key = self._hash(client_key)
            count, oldest = conn.execute("SELECT COUNT(*), MIN(ts) FROM starts WHERE ip_hash = ?",
                                         (key,)).fetchone()
            if count >= self.conversations_per_hour:
                conn.execute("COMMIT")
                retry = max(1, math.ceil(oldest + HOUR - now))
                return Decision(False, "rate_limited",
                                f"Не больше {self.conversations_per_hour} разговоров в час с одного адреса. "
                                f"Попробуйте через {max(1, (retry + 59) // 60)} мин. Записанный разговор "
                                "работает и сейчас.", retry_after_s=retry)
            conn.execute("INSERT INTO starts (ip_hash, ts) VALUES (?, ?)", (key, now))
            conn.execute("COMMIT")
            return Decision(True)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def reserve(self, worst_rub: float) -> Decision:
        now = self.clock()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            spent = conn.execute("SELECT COALESCE(SUM(rub), 0) FROM spend WHERE day = ?",
                                 (self.day(now),)).fetchone()[0]
            if spent + worst_rub > self.daily_budget_rub:
                conn.execute("COMMIT")
                return Decision(False, "budget_exhausted",
                                f"Дневной бюджет демо ({_rub(self.daily_budget_rub)}) на сегодня закончился, "
                                "живой разговор — после 00:00 по Москве. Записанный разговор работает и сейчас.")
            cur = conn.execute("INSERT INTO spend (day, ts, rub, status) VALUES (?, ?, ?, 'reserved')",
                               (self.day(now), now, float(worst_rub)))
            conn.execute("COMMIT")
            return Decision(True, reservation_id=cur.lastrowid)
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def settle(self, reservation_id: int, rub: float) -> None:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE spend SET rub = ?, status = 'settled' WHERE id = ?", (float(rub), reservation_id))

    def spent_today(self) -> float:
        with closing(self._connect()) as conn:
            return float(conn.execute("SELECT COALESCE(SUM(rub), 0) FROM spend WHERE day = ?",
                                      (self.day(),)).fetchone()[0])

    # ---------- conversations ----------

    def create(self, conv_id: str, state: dict) -> None:
        now = self.clock()
        with closing(self._connect()) as conn:
            conn.execute("INSERT INTO conversations (id, ts, updated, state) VALUES (?, ?, ?, ?)",
                         (conv_id, now, now, json.dumps(state, ensure_ascii=False)))
        self.purge()

    def load(self, conv_id: str) -> tuple[dict, dict | None] | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT state, triage FROM conversations WHERE id = ? AND ts > ?",
                               (conv_id, self.clock() - self.retention_s)).fetchone()
        if not row:
            return None
        return json.loads(row[0]), (json.loads(row[1]) if row[1] else None)

    def save(self, conv_id: str, state: dict, cost_rub: float, triage: dict | None = None) -> None:
        """``state`` must already hold masked text only (app.dialog masks before it stores)."""
        with closing(self._connect()) as conn:
            conn.execute("UPDATE conversations SET state = ?, updated = ?, cost_rub = cost_rub + ?, "
                         "triage = COALESCE(?, triage) WHERE id = ?",
                         (json.dumps(state, ensure_ascii=False), self.clock(), float(cost_rub),
                          json.dumps(triage, ensure_ascii=False) if triage else None, conv_id))

    def counters(self) -> dict:
        since = self.clock() - 86400
        with closing(self._connect()) as conn:
            total, sent = conn.execute("SELECT COUNT(*), COUNT(triage) FROM conversations WHERE ts > ?",
                                       (since,)).fetchone()
        return {"conversations_24h": int(total), "sent_to_n8n_24h": int(sent)}

    def purge(self) -> None:
        now = self.clock()
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM conversations WHERE ts <= ?", (now - self.retention_s,))
            conn.execute("DELETE FROM starts WHERE ts <= ?", (now - HOUR,))
            conn.execute("DELETE FROM spend WHERE ts <= ?", (now - 3 * 86400,))

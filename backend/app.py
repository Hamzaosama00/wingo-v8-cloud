import math
import os
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import requests
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


HISTORY_URL = os.getenv(
    "WINGO_HISTORY_URL",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
)

ORIGIN_HEADER = os.getenv("WINGO_ORIGIN", "https://www.92pak8.com")
REFERER_HEADER = os.getenv("WINGO_REFERER", "https://www.92pak8.com/")
DB_PATH = os.getenv("DB_PATH", "wingo_master.db")

HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "500"))
BACKFILL_TARGET = int(os.getenv("BACKFILL_TARGET", "500"))
BACKFILL_PAGE_SIZE = int(os.getenv("BACKFILL_PAGE_SIZE", "100"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "10"))
NUMBER_STATES = list(range(10))

COLOR_STATES = ["red", "green", "violet"]
SIZE_STATES = ["small", "big"]
PARITY_STATES = ["even", "odd"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": ORIGIN_HEADER,
    "Referer": REFERER_HEADER,
}



FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
FIREBASE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

def get_firestore():
    if not firebase_admin._apps:
        opts = {"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None
        if FIREBASE_CREDENTIALS:
            firebase_admin.initialize_app(credentials.Certificate(FIREBASE_CREDENTIALS), opts)
        else:
            firebase_admin.initialize_app(options=opts)
    return firestore.client()


class PredictorEngine:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.lock = threading.RLock()

        self.last_error: Optional[str] = None
        self.last_poll_at: Optional[int] = None
        self.last_api_ok_at: Optional[int] = None
        self.last_seen_issue: Optional[str] = None

        self.ema_decay = float(os.getenv("EMA_DECAY", "0.82"))
        self._setup_db()

    # ---------------------------------------------------------
    # Database helpers
    # ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_db(self) -> None:
        db_parent = os.path.dirname(os.path.abspath(DB_PATH))
        os.makedirs(db_parent, exist_ok=True)

        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rounds (
                    issue TEXT PRIMARY KEY,
                    number INTEGER NOT NULL,
                    color TEXT NOT NULL,
                    size TEXT NOT NULL,
                    parity TEXT NOT NULL,
                    seen_at INTEGER NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    issue TEXT PRIMARY KEY,
                    based_on_issue TEXT NOT NULL,

                    joint_state TEXT NOT NULL,
                    predicted_color TEXT NOT NULL,
                    predicted_size TEXT NOT NULL,
                    predicted_parity TEXT NOT NULL,

                    joint_score REAL NOT NULL,
                    joint_margin REAL NOT NULL,
                    joint_entropy REAL NOT NULL,
                    joint_support INTEGER NOT NULL,
                    agreement_count INTEGER NOT NULL,
                    regime TEXT NOT NULL,
                    volatility REAL NOT NULL,
                    signal TEXT NOT NULL,

                    color_score REAL NOT NULL,
                    size_score REAL NOT NULL,
                    parity_score REAL NOT NULL,

                    created_at INTEGER NOT NULL,
                    verified INTEGER NOT NULL DEFAULT 0,

                    actual_color TEXT,
                    actual_size TEXT,
                    actual_parity TEXT,
                    joint_win INTEGER,
                    color_win INTEGER,
                    size_win INTEGER,
                    parity_win INTEGER
                )
            """)

            # V7 migration: keep the predicted digit as well as derived features.
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(predictions)").fetchall()
            }
            if "predicted_number" not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN predicted_number INTEGER")

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rounds_seen_at
                ON rounds(seen_at)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_created
                ON predictions(created_at)
            """)

            conn.commit()

    def save_round(self, game: Dict[str, Any]) -> bool:
        with self._connect() as conn:
            cur = conn.execute("""
                INSERT OR IGNORE INTO rounds
                (issue, number, color, size, parity, seen_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                game["issue"],
                game["number"],
                game["color"],
                game["size"],
                game["parity"],
                int(time.time()),
            ))
            conn.commit()
            return cur.rowcount > 0

    def count_rounds(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM rounds").fetchone()
            return int(row["c"])

    def persist_round_firestore(self, game: Dict[str, Any]) -> None:
        payload = dict(game)
        payload["issue"] = str(game["issue"])
        payload["issue_num"] = int(game["issue"]) if str(game["issue"]).isdigit() else 0
        payload["seen_at"] = int(time.time())
        get_firestore().collection("rounds").document(str(game["issue"])).set(payload, merge=True)

    def hydrate_rounds_firestore(self, limit: int = HISTORY_LIMIT) -> int:
        """Restore Render's ephemeral SQLite cache from permanent Firestore."""
        db = get_firestore()
        docs = list(db.collection("rounds").order_by(
            "issue_num", direction=firestore.Query.DESCENDING
        ).limit(limit).stream())
        restored = 0
        for doc in reversed(docs):
            d = doc.to_dict() or {}
            try:
                n = int(d["number"])
                game = {
                    "issue": str(d.get("issue") or doc.id),
                    "number": n,
                    "color": str(d.get("color", "")),
                    "size": str(d.get("size") or ("big" if n >= 5 else "small")),
                    "parity": str(d.get("parity") or ("even" if n % 2 == 0 else "odd")),
                }
                if self.save_round(game):
                    restored += 1
            except Exception:
                continue
        return restored

    def load_history(self, limit: int = HISTORY_LIMIT) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT issue, number, color, size, parity, seen_at
                FROM rounds
                ORDER BY
                    CASE
                        WHEN issue GLOB '[0-9]*' THEN CAST(issue AS INTEGER)
                        ELSE seen_at
                    END DESC
                LIMIT ?
            """, (limit,)).fetchall()

        rows = list(reversed(rows))
        return [dict(row) for row in rows]

    def prediction_exists(self, issue: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM predictions WHERE issue = ?",
                (issue,),
            ).fetchone()
        return row is not None

    # ---------------------------------------------------------
    # API and parsing
    # ---------------------------------------------------------

    def fetch_history(self, page_no: int = 1, page_size: int = 100) -> Optional[Dict[str, Any]]:
        try:
            response = self.session.get(
                HISTORY_URL,
                params={
                    "pageNo": page_no,
                    "pageSize": page_size,
                    "ts": int(time.time() * 1000),
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            self.last_api_ok_at = int(time.time())
            self.last_error = None
            return data

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    @staticmethod
    def parse_color(raw: Any) -> Optional[str]:
        parts = [x.strip().lower() for x in str(raw).split(",") if x.strip()]

        if "violet" in parts:
            return "violet"
        if "red" in parts:
            return "red"
        if "green" in parts:
            return "green"
        return None

    def parse_api_history(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            items = data["data"]["list"]
        except (KeyError, TypeError):
            return []

        parsed: List[Dict[str, Any]] = []

        for item in items:
            try:
                issue = str(item["issueNumber"])
                number = int(item["number"])
                color = self.parse_color(item.get("color", ""))

                if color is None:
                    continue

                parsed.append({
                    "issue": issue,
                    "number": number,
                    "color": color,
                    "size": "big" if number >= 5 else "small",
                    "parity": "even" if number % 2 == 0 else "odd",
                })

            except (KeyError, TypeError, ValueError):
                continue

        # API commonly returns newest -> oldest.
        parsed.reverse()
        return parsed

    # ---------------------------------------------------------
    # Generic statistical models
    # ---------------------------------------------------------

    @staticmethod
    def normalize(scores: Dict[Hashable, float], states: Sequence[Hashable]) -> Dict[Hashable, float]:
        total = sum(max(float(scores.get(state, 0.0)), 0.0) for state in states)

        if total <= 0:
            uniform = 1.0 / len(states)
            return {state: uniform for state in states}

        return {
            state: max(float(scores.get(state, 0.0)), 0.0) / total
            for state in states
        }

    def ema_distribution(
        self,
        sequence: Sequence[Hashable],
        states: Sequence[Hashable],
    ) -> Dict[Hashable, float]:
        scores = {state: 1.0 for state in states}

        if not sequence:
            return self.normalize(scores, states)

        newest_index = len(sequence) - 1

        for index, value in enumerate(sequence):
            if value not in scores:
                continue

            distance = newest_index - index
            scores[value] += self.ema_decay ** distance

        return self.normalize(scores, states)

    def markov_distribution(
        self,
        sequence: Sequence[Hashable],
        states: Sequence[Hashable],
    ) -> Tuple[Dict[Hashable, float], int]:
        scores = {state: 1.0 for state in states}

        if len(sequence) < 2:
            return self.normalize(scores, states), 0

        current = sequence[-1]
        support = 0

        for i in range(len(sequence) - 1):
            if sequence[i] != current:
                continue

            nxt = sequence[i + 1]
            if nxt in scores:
                scores[nxt] += 1.0
                support += 1

        return self.normalize(scores, states), support

    def ngram_distribution(
        self,
        sequence: Sequence[Hashable],
        states: Sequence[Hashable],
        max_order: int = 5,
    ) -> Tuple[Dict[Hashable, float], int, int]:
        if len(sequence) < 4:
            return self.normalize({s: 1.0 for s in states}, states), 0, 0

        max_order = min(max_order, len(sequence) - 1)

        for order in range(max_order, 0, -1):
            context = tuple(sequence[-order:])
            scores = {state: 1.0 for state in states}
            support = 0

            # The current trailing context itself is not allowed to have a future.
            for i in range(0, len(sequence) - order):
                historical_context = tuple(sequence[i:i + order])

                if historical_context != context:
                    continue

                next_index = i + order
                if next_index >= len(sequence):
                    continue

                nxt = sequence[next_index]
                if nxt in scores:
                    scores[nxt] += 1.0
                    support += 1

            if support >= 2:
                return self.normalize(scores, states), support, order

        return self.normalize({s: 1.0 for s in states}, states), 0, 0

    def streak_distribution(
        self,
        sequence: Sequence[Hashable],
        states: Sequence[Hashable],
    ) -> Tuple[Dict[Hashable, float], int]:
        scores = {state: 1.0 for state in states}

        if not sequence:
            return self.normalize(scores, states), 0

        latest = sequence[-1]
        streak = 1

        for i in range(len(sequence) - 2, -1, -1):
            if sequence[i] != latest:
                break
            streak += 1

        if streak >= 4:
            scores[latest] += 2.0
        elif streak == 3:
            scores[latest] += 1.25
        elif streak == 2:
            scores[latest] += 0.5

        return self.normalize(scores, states), streak

    @staticmethod
    def normalized_entropy(distribution: Dict[Hashable, float]) -> float:
        if len(distribution) <= 1:
            return 0.0

        entropy = 0.0

        for probability in distribution.values():
            p = float(probability)
            if p > 0:
                entropy -= p * math.log2(p)

        maximum = math.log2(len(distribution))
        return entropy / maximum if maximum > 0 else 0.0

    @staticmethod
    def gap_since_previous_same(sequence: Sequence[Hashable]) -> int:
        if len(sequence) < 2:
            return len(sequence)

        target = sequence[-1]

        for distance in range(1, len(sequence)):
            if sequence[-1 - distance] == target:
                return distance

        return len(sequence)

    @staticmethod
    def volatility(sequence: Sequence[Hashable], window: int = 20) -> float:
        seq = list(sequence[-window:])

        if len(seq) < 2:
            return 0.5

        changes = sum(
            1
            for i in range(1, len(seq))
            if seq[i] != seq[i - 1]
        )

        return changes / (len(seq) - 1)

    # ---------------------------------------------------------
    # Feature and joint state helpers
    # ---------------------------------------------------------

    @staticmethod
    def joint_state(game: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            game["color"],
            game["size"],
            game["parity"],
        )

    @staticmethod
    def joint_key(state: Tuple[str, str, str]) -> str:
        return "|".join(state)

    def combined_volatility(self, games: List[Dict[str, Any]], window: int = 20) -> float:
        recent = games[-window:]

        if len(recent) < 2:
            return 0.5

        values = []

        for feature in ("color", "size", "parity"):
            seq = [game[feature] for game in recent]
            values.append(self.volatility(seq, window=len(seq)))

        return sum(values) / len(values)

    @staticmethod
    def regime_from_volatility(volatility: float) -> str:
        if volatility < 0.30:
            return "stable"
        if volatility > 0.70:
            return "choppy"
        return "normal"

    def regime_weights(
        self,
        volatility: float,
        markov_support: int,
        pattern_support: int,
        pattern_order: int,
        context_support: int = 0,
        include_context: bool = False,
    ) -> Dict[str, float]:
        regime = self.regime_from_volatility(volatility)

        if regime == "stable":
            raw = {
                "ema": 0.10,
                "markov": 0.45,
                "pattern": 0.10,
                "streak": 0.25,
            }
        elif regime == "choppy":
            raw = {
                "ema": 0.40,
                "markov": 0.10,
                "pattern": 0.30,
                "streak": 0.05,
            }
        else:
            raw = {
                "ema": 0.25,
                "markov": 0.25,
                "pattern": 0.25,
                "streak": 0.10,
            }

        # Evidence-sensitive adjustment. Weak models get less say.
        markov_strength = min(markov_support / 20.0, 1.0)
        pattern_strength = min(pattern_support / 10.0, 1.0)
        order_strength = min(pattern_order / 5.0, 1.0) if pattern_order else 0.0

        raw["markov"] *= 0.30 + 0.70 * markov_strength
        raw["pattern"] *= 0.20 + 0.80 * pattern_strength * max(order_strength, 0.2)

        if include_context:
            context_strength = min(context_support / 15.0, 1.0)
            raw["context"] = 0.15 * (0.25 + 0.75 * context_strength)

        total = sum(raw.values())
        return {name: value / total for name, value in raw.items()}

    def context_distribution(
        self,
        games: List[Dict[str, Any]],
        feature: str,
        states: Sequence[Hashable],
    ) -> Tuple[Dict[Hashable, float], int]:
        scores = {state: 1.0 for state in states}

        if len(games) < 2:
            return self.normalize(scores, states), 0

        current_context = self.joint_state(games[-1])
        support = 0

        for i in range(len(games) - 1):
            if self.joint_state(games[i]) != current_context:
                continue

            nxt = games[i + 1][feature]
            if nxt in scores:
                scores[nxt] += 1.0
                support += 1

        return self.normalize(scores, states), support

    # ---------------------------------------------------------
    # Individual feature model
    # ---------------------------------------------------------

    def predict_feature(
        self,
        games: List[Dict[str, Any]],
        feature: str,
        states: Sequence[Hashable],
    ) -> Dict[str, Any]:
        sequence = [game[feature] for game in games]
        vol = self.combined_volatility(games)

        ema = self.ema_distribution(sequence, states)
        markov, markov_support = self.markov_distribution(sequence, states)
        pattern, pattern_support, pattern_order = self.ngram_distribution(sequence, states)
        streak, streak_length = self.streak_distribution(sequence, states)
        context, context_support = self.context_distribution(games, feature, states)

        weights = self.regime_weights(
    volatility=vol,
    markov_support=markov_support,
    pattern_support=pattern_support,
    pattern_order=pattern_order,
    context_support=context_support,
    include_context=True,
)
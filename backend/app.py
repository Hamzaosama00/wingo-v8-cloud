import math
import os
import sqlite3
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

import requests
import psutil
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


HISTORY_URL = os.getenv(
    "WINGO_HISTORY_URL",
    "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
)

ORIGIN_HEADER = os.getenv("WINGO_ORIGIN", "https://www.92pak8.com")
REFERER_HEADER = os.getenv("WINGO_REFERER", "https://www.92pak8.com/")
DB_PATH = os.getenv("DB_PATH", "wingo_master.db")

HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "300"))
BACKFILL_TARGET = int(os.getenv("BACKFILL_TARGET", "300"))
BACKFILL_PAGE_SIZE = int(os.getenv("BACKFILL_PAGE_SIZE", "100"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "50"))
NUMBER_STATES = list(range(10))
ENABLE_SOURCE_WORKER = os.getenv("ENABLE_SOURCE_WORKER", "0").lower() in ("1", "true", "yes")

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
        if FIREBASE_CREDENTIALS:
            firebase_admin.initialize_app(
                credentials.Certificate(FIREBASE_CREDENTIALS),
                {"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None,
            )
        else:
            firebase_admin.initialize_app(
                options={"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None
            )
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
            if "actual_number" not in columns:
                conn.execute("ALTER TABLE predictions ADD COLUMN actual_number INTEGER")

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

    def save_round_cloud(self, game: Dict[str, Any]) -> None:
        payload = dict(game)
        payload["seen_at"] = int(payload.get("seen_at") or time.time())
        get_firestore().collection("rounds").document(str(game["issue"])).set(payload, merge=True)

    def hydrate_from_firestore(self, limit: int = HISTORY_LIMIT) -> int:
        """Rebuild ephemeral SQLite from permanent Firestore without buffering docs."""
        restored = 0
        query = (
            get_firestore().collection("rounds")
            .order_by("issue", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        for doc in query.stream():
            data = doc.to_dict() or {}
            try:
                number = int(data["number"])
                game = {
                    "issue": str(data.get("issue") or doc.id),
                    "number": number,
                    "color": str(data["color"]),
                    "size": str(data.get("size") or ("big" if number >= 5 else "small")),
                    "parity": str(data.get("parity") or ("even" if number % 2 == 0 else "odd")),
                }
                if self.save_round(game):
                    restored += 1
            except Exception:
                continue
        return restored

    def save_prediction_cloud(self, prediction: Optional[Dict[str, Any]]) -> None:
        if not prediction:
            return
        payload = {k: v for k, v in prediction.items() if v is not None}
        get_firestore().collection("predictions").document(str(prediction["issue"])).set(payload, merge=True)

    def hydrate_predictions_from_firestore(self, limit: int = 400) -> int:
        """Restore prediction ledger so a restart cannot silently replace an old forecast."""
        restored = 0
        query = (
            get_firestore().collection("predictions")
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        with self._connect() as conn:
            for doc in query.stream():
                d = doc.to_dict() or {}
                try:
                    vals = (
                        str(d.get("issue") or doc.id),
                        str(d["based_on_issue"]),
                        str(d["joint_state"]),
                        str(d["predicted_color"]),
                        str(d["predicted_size"]),
                        str(d["predicted_parity"]),
                        int(d["predicted_number"]) if d.get("predicted_number") is not None else None,
                        float(d["joint_score"]), float(d["joint_margin"]), float(d["joint_entropy"]),
                        int(d["joint_support"]), int(d["agreement_count"]), str(d["regime"]),
                        float(d["volatility"]), str(d["signal"]), float(d["color_score"]),
                        float(d["size_score"]), float(d["parity_score"]), int(d["created_at"]),
                        int(d.get("verified") or 0),
                        d.get("actual_number"), d.get("actual_color"), d.get("actual_size"), d.get("actual_parity"),
                        d.get("joint_win"), d.get("color_win"), d.get("size_win"), d.get("parity_win"),
                    )
                    cur = conn.execute("""
                        INSERT OR IGNORE INTO predictions (
                            issue,based_on_issue,joint_state,predicted_color,predicted_size,predicted_parity,
                            predicted_number,joint_score,joint_margin,joint_entropy,joint_support,agreement_count,
                            regime,volatility,signal,color_score,size_score,parity_score,created_at,verified,
                            actual_number,actual_color,actual_size,actual_parity,joint_win,color_win,size_win,parity_win
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, vals)
                    restored += int(cur.rowcount > 0)
                except Exception:
                    continue
            conn.commit()
        return restored

    def sync_verified_predictions_to_cloud(self, limit: int = 50) -> None:
        """Persist verification results too, so accuracy survives Render restarts."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM predictions
                WHERE verified = 1
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        db = get_firestore()
        for row in rows:
            payload = {k: v for k, v in dict(row).items() if v is not None}
            db.collection("predictions").document(str(row["issue"])).set(payload, merge=True)

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

        scores: Dict[Hashable, float] = {}

        for state in states:
            scores[state] = (
                ema[state] * weights["ema"]
                + markov[state] * weights["markov"]
                + pattern[state] * weights["pattern"]
                + streak[state] * weights["streak"]
                + context[state] * weights["context"]
            )

        scores = self.normalize(scores, states)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        prediction = ordered[0][0]
        score = float(ordered[0][1])
        margin = score - float(ordered[1][1])

        return {
            "prediction": prediction,
            "score": score,
            "margin": margin,
            "entropy": self.normalized_entropy(scores),
            "distribution": scores,
            "weights": weights,
            "markov_support": markov_support,
            "pattern_support": pattern_support,
            "pattern_order": pattern_order,
            "context_support": context_support,
            "streak_length": streak_length,
            "gap_current": self.gap_since_previous_same(sequence),
            "total_support": markov_support + pattern_support + context_support,
        }

    # ---------------------------------------------------------
    # Joint-state model
    # ---------------------------------------------------------

    def predict_joint(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        sequence = [self.joint_state(game) for game in games]

        # Learn valid joint states from observed history.
        states = sorted(set(sequence))

        if len(states) < 2:
            states = list(set(states) | {
                ("red", "small", "even"),
                ("green", "small", "odd"),
            })

        vol = self.combined_volatility(games)

        ema = self.ema_distribution(sequence, states)
        markov, markov_support = self.markov_distribution(sequence, states)
        pattern, pattern_support, pattern_order = self.ngram_distribution(sequence, states)
        streak, streak_length = self.streak_distribution(sequence, states)

        weights = self.regime_weights(
            volatility=vol,
            markov_support=markov_support,
            pattern_support=pattern_support,
            pattern_order=pattern_order,
            include_context=False,
        )

        scores: Dict[Hashable, float] = {}

        for state in states:
            scores[state] = (
                ema[state] * weights["ema"]
                + markov[state] * weights["markov"]
                + pattern[state] * weights["pattern"]
                + streak[state] * weights["streak"]
            )

        scores = self.normalize(scores, states)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        prediction = ordered[0][0]
        score = float(ordered[0][1])
        second = float(ordered[1][1]) if len(ordered) > 1 else 0.0

        return {
            "prediction": prediction,
            "score": score,
            "margin": score - second,
            "entropy": self.normalized_entropy(scores),
            "distribution": scores,
            "weights": weights,
            "markov_support": markov_support,
            "pattern_support": pattern_support,
            "pattern_order": pattern_order,
            "streak_length": streak_length,
            "gap_current": self.gap_since_previous_same(sequence),
            "total_support": markov_support + pattern_support,
            "volatility": vol,
            "regime": self.regime_from_volatility(vol),
        }

    # ---------------------------------------------------------
    # V7 number-first adaptive model
    # ---------------------------------------------------------

    @staticmethod
    def features_from_number(number: int) -> Tuple[str, str, str]:
        # Matches the API parsing convention used by this project:
        # violet takes precedence for 0/5; otherwise even=red, odd=green.
        if number in (0, 5):
            color = "violet"
        else:
            color = "red" if number % 2 == 0 else "green"
        size = "big" if number >= 5 else "small"
        parity = "even" if number % 2 == 0 else "odd"
        return color, size, parity

    def frequency_number_distribution(self, sequence: Sequence[int], window: int = 100) -> Dict[int, float]:
        scores = {n: 1.0 for n in NUMBER_STATES}
        for value in list(sequence)[-window:]:
            if value in scores:
                scores[value] += 1.0
        return self.normalize(scores, NUMBER_STATES)

    @staticmethod
    def dynamic_window_for_sequence(sequence: Sequence[int]) -> Tuple[int, float, str]:
        """Shorter lookback in choppy sequences; longer in stable sequences."""
        seq = list(sequence)
        if len(seq) < 3:
            return min(len(seq), 50), 0.5, "normal"
        recent = seq[-30:]
        changes = sum(1 for a, b in zip(recent, recent[1:]) if a != b)
        volatility = changes / max(len(recent) - 1, 1)
        if volatility < 0.60:
            return min(len(seq), 200), volatility, "stable"
        if volatility > 0.88:
            return min(len(seq), 40), volatility, "choppy"
        return min(len(seq), 100), volatility, "normal"

    def higher_order_markov_distribution(
        self, sequence: Sequence[int], max_order: int = 3
    ) -> Tuple[Dict[int, float], int, int]:
        seq = list(sequence)
        if len(seq) < 3:
            return self.normalize({n: 1.0 for n in NUMBER_STATES}, NUMBER_STATES), 0, 0
        for order in range(min(max_order, len(seq)-1), 0, -1):
            ctx = tuple(seq[-order:])
            scores = {n: 1.0 for n in NUMBER_STATES}
            support = 0
            for i in range(len(seq)-order):
                if tuple(seq[i:i+order]) == ctx:
                    nxt = seq[i+order]
                    if nxt in scores:
                        scores[nxt] += 1.0
                        support += 1
            if support >= (2 if order > 1 else 1):
                return self.normalize(scores, NUMBER_STATES), support, order
        return self.normalize({n: 1.0 for n in NUMBER_STATES}, NUMBER_STATES), 0, 0

    def lag_distribution(self, sequence: Sequence[int], lags: Sequence[int] = (5, 10)) -> Tuple[Dict[int, float], float]:
        """Empirical lag repeat/transition evidence. Does not assume cycles exist."""
        seq = list(sequence)
        scores = {n: 1.0 for n in NUMBER_STATES}
        strengths = []
        for lag in lags:
            if len(seq) <= lag:
                continue
            matches = sum(int(seq[i] == seq[i-lag]) for i in range(lag, len(seq)))
            strengths.append(matches / max(len(seq)-lag, 1))
            anchor = seq[-lag]
            for i in range(lag, len(seq)-1):
                if seq[i-lag] == anchor:
                    scores[seq[i]] += 0.5
        return self.normalize(scores, NUMBER_STATES), (sum(strengths)/len(strengths) if strengths else 0.0)

    def zscore_reversion_distribution(self, sequence: Sequence[int], window: int = 60) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Frequency deviation diagnostic with a deliberately weak reversion prior."""
        seq = list(sequence)[-window:]
        n = max(len(seq), 1)
        expected = n / 10.0
        sd = math.sqrt(max(n * 0.1 * 0.9, 1e-9))
        counts = {x: seq.count(x) for x in NUMBER_STATES}
        z = {x: (counts[x]-expected)/sd for x in NUMBER_STATES}
        # Underrepresented digits receive only a mild boost; this is not a gambler's-fallacy claim.
        scores = {x: math.exp(max(-2.0, min(2.0, -0.20*z[x]))) for x in NUMBER_STATES}
        return self.normalize(scores, NUMBER_STATES), z

    def momentum_distribution(self, sequence: Sequence[int]) -> Tuple[Dict[int, float], float]:
        """Measures recent persistence in derived size/parity; kept weak."""
        seq=list(sequence)
        scores={n:1.0 for n in NUMBER_STATES}
        if len(seq)<6:
            return self.normalize(scores, NUMBER_STATES), 0.0
        sizes=[1 if n>=5 else 0 for n in seq[-20:]]
        recent=sum(1 for a,b in zip(sizes[-6:],sizes[-5:]) if a==b)/5.0
        prior_pairs=list(zip(sizes[:-6],sizes[1:-5]))
        prior=(sum(1 for a,b in prior_pairs if a==b)/len(prior_pairs)) if prior_pairs else 0.5
        momentum=recent-prior
        desired=sizes[-1] if momentum>0 else 1-sizes[-1]
        for n in NUMBER_STATES:
            if (1 if n>=5 else 0)==desired:
                scores[n]+=min(abs(momentum),0.5)
        return self.normalize(scores, NUMBER_STATES), momentum

    def candidate_number_distributions(self, sequence: Sequence[int]) -> Dict[str, Dict[int, float]]:
        seq=list(sequence)
        window, _, _ = self.dynamic_window_for_sequence(seq)
        seq=seq[-window:] if window else seq
        ema = self.ema_distribution(seq, NUMBER_STATES)
        markov, _, _ = self.higher_order_markov_distribution(seq, max_order=3)
        pattern, _, _ = self.ngram_distribution(seq, NUMBER_STATES, max_order=3)
        freq = self.frequency_number_distribution(seq, window=window or 100)
        lag, _ = self.lag_distribution(seq)
        zscore, _ = self.zscore_reversion_distribution(seq)
        momentum, _ = self.momentum_distribution(seq)
        return {
            "ema": ema, "markov": markov, "pattern": pattern, "frequency": freq,
            "lag": lag, "zscore": zscore, "momentum": momentum,
        }

    def walk_forward_model_scores(
        self,
        sequence: Sequence[int],
        lookback: int = 100,
        min_train: int = 30,
    ) -> Dict[str, Dict[str, float]]:
        names = ("ema","markov","pattern","frequency","lag","zscore","momentum")
        stats = {
            name: {
                "correct":0.0, "tested":0.0, "logloss":0.0,
                "weighted_logloss":0.0,
                "stable_correct":0.0, "stable_tested":0.0,
                "normal_correct":0.0, "normal_tested":0.0,
                "choppy_correct":0.0, "choppy_tested":0.0,
            } for name in names
        }
        start=max(min_train,len(sequence)-lookback)
        for target_index in range(start,len(sequence)):
            train=list(sequence[:target_index])
            actual=int(sequence[target_index])
            _, _, regime=self.dynamic_window_for_sequence(train)
            candidates=self.candidate_number_distributions(train)
            for name,dist in candidates.items():
                predicted=max(dist,key=dist.get)
                p=max(float(dist.get(actual,0.0)),1e-9)
                peak=max(float(v) for v in dist.values())
                loss=-math.log(p)
                # Confident distributions are penalized more when wrong.
                penalty=1.0 + (2.0*max(0.0,peak-0.10) if predicted!=actual else 0.0)
                s=stats[name]
                s["tested"]+=1.0
                s["correct"]+=float(predicted==actual)
                s["logloss"]+=loss
                s["weighted_logloss"]+=loss*penalty
                s[f"{regime}_tested"]+=1.0
                s[f"{regime}_correct"]+=float(predicted==actual)

        for name in names:
            s=stats[name]; tested=s["tested"]
            if tested:
                s["accuracy"]=s["correct"]/tested
                s["logloss"]/=tested
                s["weighted_logloss"]/=tested
            else:
                s["accuracy"]=0.10
                s["logloss"]=s["weighted_logloss"]=math.log(10.0)
            for regime in ("stable","normal","choppy"):
                t=s[f"{regime}_tested"]
                s[f"{regime}_accuracy"]=(s[f"{regime}_correct"]/t) if t else None
        return stats

    def adaptive_number_weights(self, sequence: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        stats=self.walk_forward_model_scores(sequence)
        _,_,current_regime=self.dynamic_window_for_sequence(sequence)
        raw={}
        for name,row in stats.items():
            tested=max(row["tested"],1.0)
            reliability=min(tested/60.0,1.0)
            quality=math.exp(-float(row["weighted_logloss"]))
            rt=row.get(f"{current_regime}_tested",0.0)
            ra=row.get(f"{current_regime}_accuracy")
            regime_factor=1.0
            if rt>=10 and ra is not None:
                regime_factor=0.75+min(max(float(ra)/0.10,0.5),1.5)*0.25
            raw[name]=0.02+reliability*quality*regime_factor
        total=sum(raw.values())
        return ({n:v/total for n,v in raw.items()},stats)

    def predict_number(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        full_sequence=[int(game["number"]) for game in games]
        window,adaptive_volatility,adaptive_regime=self.dynamic_window_for_sequence(full_sequence)
        sequence=full_sequence[-window:] if window else full_sequence
        candidates=self.candidate_number_distributions(sequence)
        weights,backtest=self.adaptive_number_weights(sequence)
        distribution={n:sum(weights[name]*candidates[name][n] for name in candidates) for n in NUMBER_STATES}
        distribution=self.normalize(distribution,NUMBER_STATES)
        ordered=sorted(distribution.items(),key=lambda item:item[1],reverse=True)
        prediction=int(ordered[0][0]); score=float(ordered[0][1]); margin=score-float(ordered[1][1])
        entropy=self.normalized_entropy(distribution)
        _,z=self.zscore_reversion_distribution(sequence)
        _,lag_strength=self.lag_distribution(sequence)
        _,momentum=self.momentum_distribution(sequence)
        # Calibrated evidence score, not a claimed win probability.
        evidence=max(0.0,min(1.0,
            0.35*(1.0-entropy) +
            0.30*min(margin/0.12,1.0) +
            0.20*min(max((score-0.10)/0.15,0.0),1.0) +
            0.15*min(max(lag_strength-0.10,0.0)/0.20,1.0)
        ))
        return {
            "prediction":prediction,"score":score,"margin":margin,"entropy":entropy,
            "distribution":distribution,"weights":weights,"backtest":backtest,
            "adaptive_window":window,"adaptive_volatility":adaptive_volatility,
            "adaptive_regime":adaptive_regime,"z_scores":z,
            "lag_strength":lag_strength,"momentum":momentum,
            "confidence":evidence,
        }

    # ---------------------------------------------------------
    # V8.3 Flash selective-prediction helpers
    # ---------------------------------------------------------

    def entropy_trend(self, sequence: Sequence[int], window: int = 30) -> float:
        """Positive = entropy is rising (recent sequence is becoming less structured)."""
        seq = list(sequence)
        if len(seq) < 20:
            return 0.0

        def window_entropy(values: Sequence[int]) -> float:
            counts = {n: 1e-9 for n in NUMBER_STATES}
            for value in values:
                counts[int(value)] += 1.0
            return self.normalized_entropy(self.normalize(counts, NUMBER_STATES))

        w = min(window, len(seq) // 2)
        if w < 10:
            return 0.0
        previous = window_entropy(seq[-2*w:-w])
        recent = window_entropy(seq[-w:])
        return recent - previous

    def regime_priors(self, regime: str) -> Dict[str, float]:
        """Explicit model families per regime; normalized before use."""
        if regime == "stable":
            raw = {
                "ema": 0.08, "markov": 0.34, "pattern": 0.34,
                "frequency": 0.08, "lag": 0.08, "zscore": 0.03, "momentum": 0.05,
            }
        elif regime == "choppy":
            raw = {
                "ema": 0.30, "markov": 0.06, "pattern": 0.06,
                "frequency": 0.12, "lag": 0.08, "zscore": 0.32, "momentum": 0.06,
            }
        else:
            raw = {
                "ema": 0.12, "markov": 0.12, "pattern": 0.10,
                "frequency": 0.28, "lag": 0.08, "zscore": 0.08, "momentum": 0.22,
            }
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    def flash_number_weights(self, sequence: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], str]:
        """Blend walk-forward quality with regime-specific priors."""
        adaptive, stats = self.adaptive_number_weights(sequence)
        _, _, regime = self.dynamic_window_for_sequence(sequence)
        priors = self.regime_priors(regime)
        raw = {}
        for name in adaptive:
            # Keep performance evidence, but force a meaningful regime specialization.
            raw[name] = max(1e-9, adaptive[name]) * (0.35 + 0.65 * priors.get(name, 0.0) * 7.0)
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}, stats, regime

    def triadic_cooccurrence(self, games: List[Dict[str, Any]], window: int = 120) -> Dict[str, Any]:
        """
        Tracks observed Color|Size|Parity joint states.
        These features are derived from the same number, so this is a joint-state
        context model, not independent causal correlation.
        """
        recent = games[-window:]
        counts: Dict[str, float] = {}
        for game in recent:
            key = self.joint_key(self.joint_state(game))
            counts[key] = counts.get(key, 0.0) + 1.0

        total = sum(counts.values()) or 1.0
        distribution = {k: v / total for k, v in counts.items()}
        strongest = max(distribution.items(), key=lambda kv: kv[1]) if distribution else ("", 0.0)
        return {
            "distribution": distribution,
            "strongest_state": strongest[0],
            "strength": float(strongest[1]),
            "support": len(recent),
        }

    def flash_predict_number(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        full = [int(game["number"]) for game in games]
        window, adaptive_volatility, regime = self.dynamic_window_for_sequence(full)
        sequence = full[-window:] if window else full
        candidates = self.candidate_number_distributions(sequence)
        weights, backtest, regime = self.flash_number_weights(sequence)

        distribution = {
            n: sum(weights[name] * candidates[name][n] for name in candidates)
            for n in NUMBER_STATES
        }
        distribution = self.normalize(distribution, NUMBER_STATES)
        ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        prediction = int(ordered[0][0])
        score = float(ordered[0][1])
        margin = score - float(ordered[1][1])
        entropy = self.normalized_entropy(distribution)
        trend = self.entropy_trend(sequence)

        # Model agreement: count candidate models whose top digit maps to the
        # same Big/Small class as the ensemble prediction.
        predicted_size = self.features_from_number(prediction)[1]
        size_votes = 0
        for dist in candidates.values():
            top = int(max(dist, key=dist.get))
            if self.features_from_number(top)[1] == predicted_size:
                size_votes += 1
        model_agreement = size_votes / max(len(candidates), 1)

        # Evidence score is deliberately conservative. It is NOT a win probability.
        regime_stability = 1.0 if regime == "stable" else (0.60 if regime == "normal" else 0.25)
        pattern_strength = min(1.0, max(0.0, margin / 0.10))
        entropy_health = min(1.0, max(0.0, 1.0 - entropy))
        confidence = max(0.0, min(1.0,
            0.30 * model_agreement +
            0.25 * regime_stability +
            0.25 * pattern_strength +
            0.20 * entropy_health
        ))

        return {
            "prediction": prediction,
            "score": score,
            "margin": margin,
            "entropy": entropy,
            "entropy_trend": trend,
            "distribution": distribution,
            "weights": weights,
            "backtest": backtest,
            "adaptive_window": window,
            "adaptive_volatility": adaptive_volatility,
            "adaptive_regime": regime,
            "confidence": confidence,
            "model_agreement": model_agreement,
        }

    # ---------------------------------------------------------
    # Prediction decision
    # ---------------------------------------------------------

    @staticmethod
    def agreement_count(
        joint_prediction: Tuple[str, str, str],
        color_prediction: str,
        size_prediction: str,
        parity_prediction: str,
    ) -> int:
        expected = (
            color_prediction,
            size_prediction,
            parity_prediction,
        )

        return sum(
            1
            for joint_value, individual_value in zip(joint_prediction, expected)
            if joint_value == individual_value
        )

    @staticmethod
    def signal_quality(
        joint: Dict[str, Any],
        agreement: int,
    ) -> str:
        entropy = float(joint["entropy"])
        score = float(joint["score"])
        margin = float(joint["margin"])
        support = int(joint["total_support"])

        if entropy >= 0.90:
            return "SKIP"

        if support < 5:
            return "SKIP"

        if agreement <= 1:
            return "SKIP"

        if (
            entropy <= 0.62
            and score >= 0.45
            and margin >= 0.10
            and agreement == 3
        ):
            return "HIGH"

        return "MEDIUM"

    def build_prediction(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        number = self.flash_predict_number(games)
        color_value, size_value, parity_value = self.features_from_number(number["prediction"])

        color = self.predict_feature(games, "color", COLOR_STATES)
        size = self.predict_feature(games, "size", SIZE_STATES)
        parity = self.predict_feature(games, "parity", PARITY_STATES)
        joint_state = (color_value, size_value, parity_value)
        agreement = self.agreement_count(
            joint_state, color["prediction"], size["prediction"], parity["prediction"]
        )

        confidence = float(number["confidence"])
        entropy = float(number["entropy"])
        entropy_trend = float(number["entropy_trend"])
        margin = float(number["margin"])
        regime = str(number["adaptive_regime"])
        model_agreement = float(number["model_agreement"])
        triadic = self.triadic_cooccurrence(games)

        signal = "SKIP"
        reason = "confidence_below_0.85"

        # Strict V8.3 Flash gate: abstention is the default.
        if regime == "choppy":
            reason = "choppy_regime"
        elif entropy_trend > 0.015:
            reason = "entropy_rising"
        elif agreement < 3:
            reason = "feature_disagreement"
        elif model_agreement < 0.70:
            reason = "model_disagreement"
        elif entropy > 0.82:
            reason = "high_entropy"
        elif margin < 0.045:
            reason = "weak_margin"
        elif confidence < 0.85:
            reason = "confidence_below_0.85"
        else:
            signal = "HIGH"
            reason = "strict_gate_passed"

        volatility = self.combined_volatility(games)
        joint = {
            "prediction": joint_state,
            "score": number["score"],
            "margin": margin,
            "entropy": entropy,
            "entropy_trend": entropy_trend,
            "distribution": {},
            "weights": number["weights"],
            "total_support": len(games),
            "volatility": volatility,
            "regime": regime,
            "number_prediction": number["prediction"],
            "number_distribution": number["distribution"],
            "backtest": number["backtest"],
            "confidence": confidence,
            "model_agreement": model_agreement,
            "adaptive_window": number["adaptive_window"],
            "triadic_cooccurrence": triadic,
            "gate_reason": reason,
        }
        return {
            "joint": joint, "number": number, "color": color, "size": size,
            "parity": parity, "agreement": agreement, "signal": signal,
            "gate_reason": reason,
        }


    # ---------------------------------------------------------
    # Save / verify predictions
    # ---------------------------------------------------------

    @staticmethod
    def get_next_issue(current_issue: str) -> str:
        try:
            return str(int(current_issue) + 1)
        except ValueError:
            return f"{current_issue}_NEXT"

    def save_next_prediction(self, games: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if len(games) < MIN_HISTORY:
            return None

        latest = games[-1]
        target_issue = self.get_next_issue(latest["issue"])

        if self.prediction_exists(target_issue):
            return None

        result = self.build_prediction(games)

        joint_state = result["joint"]["prediction"]
        number_prediction = int(result["number"]["prediction"])
        color_prediction = str(joint_state[0])
        size_prediction = str(joint_state[1])
        parity_prediction = str(joint_state[2])

        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO predictions (
                    issue,
                    based_on_issue,

                    joint_state,
                    predicted_color,
                    predicted_size,
                    predicted_parity,
                    predicted_number,

                    joint_score,
                    joint_margin,
                    joint_entropy,
                    joint_support,
                    agreement_count,
                    regime,
                    volatility,
                    signal,

                    color_score,
                    size_score,
                    parity_score,

                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                target_issue,
                latest["issue"],

                self.joint_key(joint_state),
                color_prediction,
                size_prediction,
                parity_prediction,
                number_prediction,

                float(result["joint"]["score"]),
                float(result["joint"]["margin"]),
                float(result["joint"]["entropy"]),
                int(result["joint"]["total_support"]),
                int(result["agreement"]),
                str(result["joint"]["regime"]),
                float(result["joint"]["volatility"]),
                str(result["signal"]),

                float(result["color"]["score"]),
                float(result["size"]["score"]),
                float(result["parity"]["score"]),

                int(time.time()),
            ))
            conn.commit()

        return self.get_prediction(target_issue)

    def verify_pending_predictions(self) -> int:
        verified_now = 0

        with self._connect() as conn:
            pending = conn.execute("""
                SELECT *
                FROM predictions
                WHERE verified = 0
                ORDER BY created_at ASC
            """).fetchall()

            for prediction in pending:
                actual = conn.execute("""
                    SELECT *
                    FROM rounds
                    WHERE issue = ?
                """, (prediction["issue"],)).fetchone()

                if actual is None:
                    continue

                joint_win = int(
                    prediction["predicted_color"] == actual["color"]
                    and prediction["predicted_size"] == actual["size"]
                    and prediction["predicted_parity"] == actual["parity"]
                )

                color_win = int(prediction["predicted_color"] == actual["color"])
                size_win = int(prediction["predicted_size"] == actual["size"])
                parity_win = int(prediction["predicted_parity"] == actual["parity"])

                conn.execute("""
                    UPDATE predictions
                    SET
                        verified = 1,
                        actual_number = ?,
                        actual_color = ?,
                        actual_size = ?,
                        actual_parity = ?,
                        joint_win = ?,
                        color_win = ?,
                        size_win = ?,
                        parity_win = ?
                    WHERE issue = ?
                """, (
                    actual["number"],
                    actual["color"],
                    actual["size"],
                    actual["parity"],
                    joint_win,
                    color_win,
                    size_win,
                    parity_win,
                    prediction["issue"],
                ))

                verified_now += 1

            conn.commit()

        return verified_now

    # ---------------------------------------------------------
    # Backfill and worker
    # ---------------------------------------------------------

    def backfill(self) -> None:
        existing = self.count_rounds()

        if existing >= BACKFILL_TARGET:
            return

        seen = set()
        page_no = 1
        max_pages = 25

        while (
            self.count_rounds() < BACKFILL_TARGET
            and page_no <= max_pages
            and not self.stop_event.is_set()
        ):
            data = self.fetch_history(
                page_no=page_no,
                page_size=BACKFILL_PAGE_SIZE,
            )

            if not data or data.get("code") != 0:
                break

            games = self.parse_api_history(data)
            if not games:
                break

            inserted = 0

            for game in games:
                if game["issue"] in seen:
                    continue

                seen.add(game["issue"])

                if self.save_round(game):
                    inserted += 1

                if self.count_rounds() >= BACKFILL_TARGET:
                    break

            try:
                total_page = int(data.get("data", {}).get("totalPage", page_no))
            except Exception:
                total_page = page_no

            if page_no >= total_page:
                break

            # API may ignore pageNo/pageSize and return the same records.
            if inserted == 0:
                break

            page_no += 1
            time.sleep(0.15)

    def poll_once(self) -> None:
        self.last_poll_at = int(time.time())

        data = self.fetch_history(page_no=1, page_size=100)

        if not data or data.get("code") != 0:
            return

        games = self.parse_api_history(data)

        for game in games:
            self.save_round(game)

        self.verify_pending_predictions()

        stored = self.load_history(HISTORY_LIMIT)

        if not stored:
            return

        latest_issue = stored[-1]["issue"]

        if latest_issue != self.last_seen_issue:
            self.last_seen_issue = latest_issue
            self.save_next_prediction(stored)

    def worker_loop(self) -> None:
        try:
            self.backfill()
        except Exception as exc:
            self.last_error = f"Backfill: {type(exc).__name__}: {exc}"

        while not self.stop_event.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self.last_error = f"Poll: {type(exc).__name__}: {exc}"

            self.stop_event.wait(POLL_SECONDS)

    def start_worker(self) -> None:
        with self.lock:
            if self.worker and self.worker.is_alive():
                return

            self.stop_event.clear()
            self.worker = threading.Thread(
                target=self.worker_loop,
                name="predictor-worker",
                daemon=True,
            )
            self.worker.start()

    def stop_worker(self) -> None:
        self.stop_event.set()

        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=3)

    # ---------------------------------------------------------
    # API serialization helpers
    # ---------------------------------------------------------

    def get_latest_round(self) -> Optional[Dict[str, Any]]:
        history = self.load_history(1)
        return history[-1] if history else None

    def get_prediction(self, issue: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE issue = ?",
                (issue,),
            ).fetchone()

        return dict(row) if row else None

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT *
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 1
            """).fetchone()

        return dict(row) if row else None

    def recent_rounds(self, limit: int = 30) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 200))

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

        return [dict(row) for row in rows]

    def recent_predictions(self, limit: int = 30) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 200))

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT *
                FROM predictions
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

        return [dict(row) for row in rows]

    def accuracy_stats(self, window: int = 200) -> Dict[str, Any]:
        window = max(1, min(window, 1000))

        with self._connect() as conn:
            rows = conn.execute("""
                SELECT
                    joint_win,
                    color_win,
                    size_win,
                    parity_win,
                    signal
                FROM predictions
                WHERE verified = 1
                ORDER BY created_at DESC
                LIMIT ?
            """, (window,)).fetchall()

        def metric(field: str) -> Dict[str, Any]:
            values = [
                int(row[field])
                for row in rows
                if row[field] is not None
            ]

            if not values:
                return {
                    "correct": 0,
                    "tested": 0,
                    "accuracy": None,
                }

            correct = sum(values)
            tested = len(values)

            return {
                "correct": correct,
                "tested": tested,
                "accuracy": correct / tested,
            }

        signal_counts = defaultdict(int)
        for row in rows:
            signal_counts[row["signal"]] += 1

        return {
            "window": window,
            "joint": metric("joint_win"),
            "color": metric("color_win"),
            "size": metric("size_win"),
            "parity": metric("parity_win"),
            "signals": dict(signal_counts),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "service": "online",
            "worker_running": bool(self.worker and self.worker.is_alive()),
            "db_path": DB_PATH,
            "stored_rounds": self.count_rounds(),
            "history_limit": HISTORY_LIMIT,
            "minimum_history": MIN_HISTORY,
            "source_worker_enabled": ENABLE_SOURCE_WORKER,
            "last_poll_at": self.last_poll_at,
            "last_api_ok_at": self.last_api_ok_at,
            "last_error": self.last_error,
        }


engine = PredictorEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ENABLE_SOURCE_WORKER:
        engine.start_worker()
    yield
    if ENABLE_SOURCE_WORKER:
        engine.stop_worker()


app = FastAPI(
    title="WinGo Statistical Predictor API",
    version="8.3-flash",
    lifespan=lifespan,
)

cors_raw = os.getenv("CORS_ORIGINS", "*").strip()

if cors_raw == "*":
    allow_origins = ["*"]
else:
    allow_origins = [
        origin.strip()
        for origin in cors_raw.split(",")
        if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "WinGo Statistical Predictor API",
        "version": "8.3-flash",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/api/memory")
def memory_status():
    proc = psutil.Process(os.getpid())
    m = proc.memory_info()
    return {
        "rss_mb": round(m.rss / 1024 / 1024, 1),
        "vms_mb": round(m.vms / 1024 / 1024, 1),
        "threads": proc.num_threads(),
        "history_limit": HISTORY_LIMIT,
        "source_worker_enabled": ENABLE_SOURCE_WORKER,
    }


@app.get("/health")
def health():
    return engine.status()


@app.get("/api/dashboard")
def dashboard():
    return {
        "status": engine.status(),
        "latest_round": engine.get_latest_round(),
        "latest_prediction": engine.get_latest_prediction(),
        "accuracy": engine.accuracy_stats(200),
        "history": engine.recent_rounds(20),
        "predictions": engine.recent_predictions(20),
    }


@app.get("/api/history")
def history(limit: int = 30):
    return {
        "items": engine.recent_rounds(limit),
    }


@app.get("/api/predictions")
def predictions(limit: int = 30):
    return {
        "items": engine.recent_predictions(limit),
    }



@app.get("/api/compare")
def compare_predictions(limit: int = 50):
    limit = max(1, min(limit, 200))
    engine.verify_pending_predictions()
    with engine._connect() as conn:
        rows = conn.execute("""
            SELECT
                issue, based_on_issue, predicted_number,
                predicted_color, predicted_size, predicted_parity,
                signal, joint_score, created_at, verified,
                actual_number, actual_color, actual_size, actual_parity,
                joint_win, color_win, size_win, parity_win
            FROM predictions
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

    items = []
    verified = wins = 0
    for r in rows:
        x = dict(r)
        if x["verified"]:
            verified += 1
            wins += int(x["joint_win"] or 0)
            x["result"] = "WIN" if x["joint_win"] else "LOSS"
        else:
            x["result"] = "PENDING"
        items.append(x)

    return {
        "model_version": "8.3-flash",
        "verified": verified,
        "joint_wins": wins,
        "joint_accuracy": round((wins / verified * 100), 2) if verified else None,
        "items": items,
    }


@app.get("/api/stats")
def stats(window: int = 200):
    return engine.accuracy_stats(window)


INGEST_SECRET = os.getenv("INGEST_SECRET", "")

class RoundInput(BaseModel):
    issue: str
    number: int
    color: str
    secret: str

@app.post("/api/ingest")
def ingest_round(round_data: RoundInput):
    if not INGEST_SECRET or round_data.secret != INGEST_SECRET:
        return {"ok": False, "error": "Unauthorized"}

    color = engine.parse_color(round_data.color)
    if color is None:
        return {"ok": False, "error": "Invalid color"}

    number = int(round_data.number)
    if number < 0 or number > 9:
        return {"ok": False, "error": "Invalid number"}

    game = {
        "issue": str(round_data.issue),
        "number": number,
        "color": color,
        "size": "big" if number >= 5 else "small",
        "parity": "even" if number % 2 == 0 else "odd",
    }

    firebase_error = None
    restored = 0
    restored_predictions = 0

    # Firestore is authoritative; SQLite is disposable cache only.
    try:
        if engine.count_rounds() == 0:
            restored = engine.hydrate_from_firestore(HISTORY_LIMIT)
            restored_predictions = engine.hydrate_predictions_from_firestore()
    except Exception as exc:
        firebase_error = f"hydrate: {type(exc).__name__}: {exc}"

    inserted = engine.save_round(game)

    # Persist the completed/public round before doing model work.
    try:
        engine.save_round_cloud(game)
    except Exception as exc:
        firebase_error = f"round-write: {type(exc).__name__}: {exc}"

    verified_now = engine.verify_pending_predictions()
    if verified_now:
        try:
            engine.sync_verified_predictions_to_cloud()
        except Exception as exc:
            firebase_error = f"verify-sync: {type(exc).__name__}: {exc}"
    history = engine.load_history(HISTORY_LIMIT)
    prediction = engine.save_next_prediction(history) if inserted else None

    try:
        engine.save_prediction_cloud(prediction)
    except Exception as exc:
        firebase_error = f"prediction-write: {type(exc).__name__}: {exc}"

    return {
        "ok": True,
        "inserted": inserted,
        "round": game,
        "stored_rounds": engine.count_rounds(),
        "restored_from_firestore": restored,
        "restored_predictions": restored_predictions,
        "prediction": prediction,
        "firebase_error": firebase_error,
        "storage": "firestore-primary/sqlite-cache",
        "model_version": "8.3-flash",
    }

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
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "50"))
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

    def candidate_number_distributions(self, sequence: Sequence[int]) -> Dict[str, Dict[int, float]]:
        ema = self.ema_distribution(sequence, NUMBER_STATES)
        markov, _ = self.markov_distribution(sequence, NUMBER_STATES)
        pattern, _, _ = self.ngram_distribution(sequence, NUMBER_STATES, max_order=3)
        freq = self.frequency_number_distribution(sequence)
        return {"ema": ema, "markov": markov, "pattern": pattern, "frequency": freq}

    def walk_forward_model_scores(
        self,
        sequence: Sequence[int],
        lookback: int = 80,
        min_train: int = 30,
    ) -> Dict[str, Dict[str, float]]:
        names = ("ema", "markov", "pattern", "frequency")
        stats = {name: {"correct": 0.0, "tested": 0.0, "logloss": 0.0} for name in names}

        start = max(min_train, len(sequence) - lookback)
        for target_index in range(start, len(sequence)):
            train = list(sequence[:target_index])
            actual = int(sequence[target_index])
            candidates = self.candidate_number_distributions(train)

            for name, dist in candidates.items():
                predicted = max(dist, key=dist.get)
                stats[name]["tested"] += 1.0
                stats[name]["correct"] += float(predicted == actual)
                stats[name]["logloss"] += -math.log(max(float(dist.get(actual, 0.0)), 1e-9))

        for name in names:
            tested = stats[name]["tested"]
            if tested:
                stats[name]["accuracy"] = stats[name]["correct"] / tested
                stats[name]["logloss"] /= tested
            else:
                stats[name]["accuracy"] = 0.10
                stats[name]["logloss"] = math.log(10.0)
        return stats

    def adaptive_number_weights(self, sequence: Sequence[int]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        stats = self.walk_forward_model_scores(sequence)
        raw = {}

        # Weight by out-of-sample probability quality, not in-sample fit.
        # exp(-logloss) is bounded and avoids a lucky tiny sample dominating.
        for name, row in stats.items():
            tested = max(row["tested"], 1.0)
            reliability = min(tested / 50.0, 1.0)
            quality = math.exp(-float(row["logloss"]))
            raw[name] = 0.05 + reliability * quality

        total = sum(raw.values())
        return ({name: value / total for name, value in raw.items()}, stats)

    def predict_number(self, games: List[Dict[str, Any]]) -> Dict[str, Any]:
        sequence = [int(game["number"]) for game in games]
        candidates = self.candidate_number_distributions(sequence)
        weights, backtest = self.adaptive_number_weights(sequence)

        distribution = {
            n: sum(weights[name] * candidates[name][n] for name in candidates)
            for n in NUMBER_STATES
        }
        distribution = self.normalize(distribution, NUMBER_STATES)
        ordered = sorted(distribution.items(), key=lambda item: item[1], reverse=True)

        prediction = int(ordered[0][0])
        score = float(ordered[0][1])
        margin = score - float(ordered[1][1])

        return {
            "prediction": prediction,
            "score": score,
            "margin": margin,
            "entropy": self.normalized_entropy(distribution),
            "distribution": distribution,
            "weights": weights,
            "backtest": backtest,
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
        number = self.predict_number(games)
        color_value, size_value, parity_value = self.features_from_number(number["prediction"])

        # Keep the old feature models only as diagnostics/agreement checks.
        color = self.predict_feature(games, "color", COLOR_STATES)
        size = self.predict_feature(games, "size", SIZE_STATES)
        parity = self.predict_feature(games, "parity", PARITY_STATES)

        joint_state = (color_value, size_value, parity_value)
        agreement = self.agreement_count(
            joint_state,
            color["prediction"],
            size["prediction"],
            parity["prediction"],
        )

        # V7 confidence is deliberately conservative. A ten-way digit model
        # should not claim HIGH confidence from a small sample.
        signal = "SKIP"
        if number["score"] >= 0.18 and number["margin"] >= 0.025 and agreement >= 2:
            signal = "MEDIUM"

        joint = {
            "prediction": joint_state,
            "score": number["score"],
            "margin": number["margin"],
            "entropy": number["entropy"],
            "distribution": {},
            "weights": number["weights"],
            "total_support": len(games),
            "volatility": self.combined_volatility(games),
            "regime": self.regime_from_volatility(self.combined_volatility(games)),
            "number_prediction": number["prediction"],
            "number_distribution": number["distribution"],
            "backtest": number["backtest"],
        }

        return {
            "joint": joint,
            "number": number,
            "color": color,
            "size": size,
            "parity": parity,
            "agreement": agreement,
            "signal": signal,
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
                        actual_color = ?,
                        actual_size = ?,
                        actual_parity = ?,
                        joint_win = ?,
                        color_win = ?,
                        size_win = ?,
                        parity_win = ?
                    WHERE issue = ?
                """, (
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
            "last_poll_at": self.last_poll_at,
            "last_api_ok_at": self.last_api_ok_at,
            "last_error": self.last_error,
        }


engine = PredictorEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start_worker()
    yield
    engine.stop_worker()



# ============================================================
# V9.3 FLASH SIZE ENGINE
# Lightweight, strict-signal, walk-forward calibrated research model.
# Firestore persistence remains handled by the backend below.
# ============================================================

V93_MIN_TRAIN = int(os.getenv("V93_MIN_TRAIN", "10"))
V93_WINDOW = int(os.getenv("V93_WINDOW", "300"))
V93_STABLE_THRESHOLD = float(os.getenv("V93_STABLE_THRESHOLD", "0.80"))
V93_CHOPPY_THRESHOLD = float(os.getenv("V93_CHOPPY_THRESHOLD", "0.85"))
V93_ENTROPY_STOP = float(os.getenv("V93_ENTROPY_STOP", "0.78"))
V93_SWITCH_RATE_STOP = float(os.getenv("V93_SWITCH_RATE_STOP", "0.72"))
V93_MIN_SUPPORT = int(os.getenv("V93_MIN_SUPPORT", "50"))
V93_MIN_AGREEMENT = int(os.getenv("V93_MIN_AGREEMENT", "3"))

def _v93_b(x):
    return 1 if str(x).lower() == "big" else 0

def _v93_clip(x, lo=0.02, hi=0.98):
    return float(np.clip(float(x), lo, hi))

def _v93_binary_entropy(p):
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(-(p*math.log2(p) + (1-p)*math.log2(1-p)))

def _v93_switch_rate(vals, n=20):
    v = vals[-n:]
    if len(v) < 2:
        return 0.5
    return float(sum(a != b for a, b in zip(v[:-1], v[1:])) / (len(v)-1))

def _v93_transition_matrix(vals, n=40):
    v = vals[-n:]
    c = [[1.0, 1.0], [1.0, 1.0]]
    for a, b in zip(v[:-1], v[1:]):
        c[int(a)][int(b)] += 1.0
    out = []
    for row in c:
        s = sum(row)
        out.append([row[0]/s, row[1]/s])
    return out

def _v93_transition_entropy(vals, n=30):
    tm = _v93_transition_matrix(vals, n)
    # Average conditional entropy H(next | current)
    return float(sum(_v93_binary_entropy(row[1]) for row in tm) / 2.0)

def _v93_gap_since(vals, target):
    g = 0
    for x in reversed(vals):
        if x == target:
            return g
        g += 1
    return len(vals)

def _v93_ema(vals, span):
    if not vals:
        return 0.5
    alpha = 2.0 / (span + 1.0)
    e = float(vals[0])
    for x in vals[1:]:
        e = alpha*float(x) + (1-alpha)*e
    return float(e)

def _v93_sequence_stability(vals, n=16):
    v = vals[-n:]
    if len(v) < 5:
        return 0.5
    trans_entropy = _v93_transition_entropy(v, len(v))
    lag2 = np.mean([1.0 if v[i] == v[i-2] else 0.0 for i in range(2, len(v))])
    return float(np.clip(0.7*(1.0-trans_entropy) + 0.3*lag2, 0, 1))

def _v93_regime(vals):
    v = vals[-30:]
    if len(v) < 12:
        return "UNKNOWN", 1.0
    r20 = sum(v[-20:])/min(20, len(v))
    entropy = _v93_binary_entropy(r20)
    t_entropy = _v93_transition_entropy(v)
    switch_rate = _v93_switch_rate(v)
    stability = _v93_sequence_stability(v)
    choppy_score = float(np.clip(
        0.30*entropy +
        0.30*t_entropy +
        0.25*switch_rate +
        0.15*(1-stability),
        0, 1
    ))
    return ("CHOPPY" if choppy_score >= 0.68 else "STABLE"), choppy_score

def _v93_ngram_probability(vals, order=3):
    """Pattern matcher: P(next=Big | recent context), Laplace smoothed."""
    if len(vals) <= order:
        return 0.5, 0
    ctx = tuple(vals[-order:])
    big = small = 1.0
    support = 0
    for i in range(order, len(vals)):
        if tuple(vals[i-order:i]) == ctx:
            support += 1
            if vals[i] == 1:
                big += 1
            else:
                small += 1
    return float(big/(big+small)), support

def _v93_trend_model(vals):
    """Trend follower: short/medium EMA + recent momentum."""
    ema3 = _v93_ema(vals[-30:], 3)
    ema8 = _v93_ema(vals[-50:], 8)
    ema20 = _v93_ema(vals[-80:], 20)
    momentum = (ema3 - ema20)
    slope = (ema8 - ema20)
    score = 0.5 + 0.75*momentum + 0.35*slope
    return _v93_clip(score)

def _v93_reversion_model(vals):
    """Reversionist: recent imbalance + gap pressure as a hypothesis feature.
    It does not assume the process must revert; walk-forward calibration decides
    whether this heuristic deserves confidence.
    """
    if len(vals) < 20:
        return 0.5
    r10 = sum(vals[-10:]) / 10.0
    r30 = sum(vals[-30:]) / min(30, len(vals))
    gap_big = min(_v93_gap_since(vals, 1), 12) / 12.0
    gap_small = min(_v93_gap_since(vals, 0), 12) / 12.0
    gap_to_mean = 0.5 - r10
    # Positive -> Big reversion hypothesis, negative -> Small.
    score = 0.5 + 0.55*gap_to_mean + 0.18*(0.5-r30) + 0.12*(gap_big-gap_small)
    return _v93_clip(score)

def _v93_pattern_model(vals):
    """Pattern matcher: first-order Markov + 2/3-gram context."""
    tm = _v93_transition_matrix(vals, 60)
    markov = tm[int(vals[-1])][1] if vals else 0.5
    p2, s2 = _v93_ngram_probability(vals, 2)
    p3, s3 = _v93_ngram_probability(vals, 3)
    # Context gets more weight only when repeated support exists.
    w3 = min(s3/10.0, 1.0)
    w2 = min(s2/15.0, 1.0)
    denom = 1.0 + 0.8*w2 + 1.0*w3
    p = (markov + 0.8*w2*p2 + 1.0*w3*p3) / denom
    return _v93_clip(p), {"markov": markov, "ngram2_support": s2, "ngram3_support": s3}

def _v93_raw_ensemble(vals):
    trend = _v93_trend_model(vals)
    revert = _v93_reversion_model(vals)
    pattern, pmeta = _v93_pattern_model(vals)
    regime, _ = _v93_regime(vals)

    # Diversified equal-weight base. In CHOPPY conditions, pattern matching gets
    # a little more weight, but permission to predict is handled separately.
    if regime == "CHOPPY":
        raw = 0.25*trend + 0.25*revert + 0.50*pattern
    else:
        raw = (trend + revert + pattern) / 3.0

    votes = [
        1 if trend >= 0.5 else 0,
        1 if revert >= 0.5 else 0,
        1 if pattern >= 0.5 else 0,
    ]
    return _v93_clip(raw), (trend, revert, pattern), votes, pmeta

def _v93_walkforward(vals):
    raw, actual = [], []
    first = max(V93_MIN_TRAIN, len(vals)-120)
    for i in range(first, len(vals)):
        tr = vals[max(0, i-V93_WINDOW):i]
        if len(tr) < V93_MIN_TRAIN:
            continue
        try:
            p, _, _, _ = _v93_raw_ensemble(tr)
            raw.append(p)
            actual.append(vals[i])
        except Exception:
            continue
    iso = None
    if len(raw) >= 30 and len(set(actual)) >= 2:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        iso.fit(np.asarray(raw), np.asarray(actual))
    return iso, raw, actual

def build_v93_flash(games):
    games = list(reversed(games))
    vals = [_v93_b(g["size"]) for g in games if str(g.get("size","")).lower() in ("big","small")]
    if len(vals) < V93_MIN_TRAIN:
        return {
            "ready": False,
            "reason": f"need {V93_MIN_TRAIN} rounds",
            "rounds": len(vals),
            "signal": "SKIP",
            "model": "V9.4.1 Flash"
        }

    vals = vals[-V93_WINDOW:]
    raw_p, components, votes, pmeta = _v93_raw_ensemble(vals)
    iso, wf_raw, wf_y = _v93_walkforward(vals)
    # V9.4: calibration is support-aware. Small calibration samples must not
    # flatten the model into a permanent ~54-56% SMALL output.
    calibrated_p = float(iso.predict([raw_p])[0]) if iso is not None else raw_p
    calibration_weight = 0.0
    if iso is not None:
        calibration_weight = float(np.clip((len(wf_y) - 30) / 70.0, 0.0, 1.0))
    p_big = float((1.0 - calibration_weight) * raw_p + calibration_weight * calibrated_p)
    p_big = _v93_clip(p_big)

    predicted_big = p_big >= 0.5
    predicted = "big" if predicted_big else "small"
    conf = p_big if predicted_big else 1-p_big

    regime, choppy_score = _v93_regime(vals)
    entropy20 = _v93_binary_entropy(sum(vals[-20:])/min(20, len(vals)))
    transition_entropy = _v93_transition_entropy(vals[-30:])
    switch_rate = _v93_switch_rate(vals[-20:])
    stability = _v93_sequence_stability(vals)

    agreement = sum((v == 1) == predicted_big for v in votes)
    threshold = V93_CHOPPY_THRESHOLD if regime == "CHOPPY" else V93_STABLE_THRESHOLD

    reasons = []
    if conf < threshold:
        reasons.append("confidence")
    if entropy20 > V93_ENTROPY_STOP:
        reasons.append("entropy")
    if switch_rate > V93_SWITCH_RATE_STOP:
        reasons.append("volatility")
    if len(wf_y) < 30:
        reasons.append("support")
    if agreement < V93_MIN_AGREEMENT:
        reasons.append("agreement")

    signal = "PREDICT" if not reasons else "SKIP"

    # Honest historical coverage/accuracy estimate using the same strict base threshold.
    hits = preds = 0
    if iso is not None:
        for rp, y in zip(wf_raw, wf_y):
            cp = float(iso.predict([rp])[0])
            pc = max(cp, 1-cp)
            if pc >= V93_STABLE_THRESHOLD:
                preds += 1
                hits += int((cp >= 0.5) == bool(y))
    oos_acc = (hits/preds) if preds else None
    coverage = (preds/len(wf_y)) if wf_y else 0.0

    return {
        "ready": True,
        "model": "V9.4.1 Flash",
        "prediction": predicted,
        "decision": predicted.upper() if signal == "PREDICT" else "NO EDGE",
        "raw_p_big": round(raw_p, 4),
        "p_big": round(p_big, 4),
        "calibration_weight": round(calibration_weight, 4),
        "calibration_support": len(wf_y),
        "calibrated_confidence": round(conf, 4),
        "signal": signal,
        "skip_reasons": reasons,
        "regime": regime,
        "threshold_used": threshold,
        "choppy_score": round(choppy_score, 4),
        "entropy20": round(entropy20, 4),
        "transition_entropy": round(transition_entropy, 4),
        "switch_rate": round(switch_rate, 4),
        "sequence_stability": round(stability, 4),
        "agreement": f"{agreement}/3",
        "support": len(wf_y),
        "rolling_oos_accuracy": None if oos_acc is None else round(oos_acc, 4),
        "rolling_oos_coverage": round(coverage, 4),
        "rolling_oos_predictions": preds,
        "components": {
            "trend_follower": round(components[0], 4),
            "reversionist": round(components[1], 4),
            "pattern_matcher": round(components[2], 4),
        },
        "pattern_meta": {
            "markov": round(float(pmeta["markov"]), 4),
            "ngram2_support": int(pmeta["ngram2_support"]),
            "ngram3_support": int(pmeta["ngram3_support"]),
        },
        "features": {
            "ema3": round(_v93_ema(vals[-30:], 3), 4),
            "ema8": round(_v93_ema(vals[-50:], 8), 4),
            "ema20": round(_v93_ema(vals[-80:], 20), 4),
            "gap_since_big": int(_v93_gap_since(vals, 1)),
            "gap_since_small": int(_v93_gap_since(vals, 0)),
            "gap_to_mean_10": round(0.5 - sum(vals[-10:])/10.0, 4),
        },
        "note": "V9.4 research signal. Direction is available from 10 completed rounds; NO EDGE/SKIP remains possible when evidence is weak. Confidence is support-aware and is not a guaranteed probability."
    }



# ============================================================
# V9.3 VIRTUAL PKR SIMULATOR
# Historical/backtest only. No live bet placement or wallet integration.
# ============================================================

def simulate_v93_virtual_balance(
    games,
    starting_balance: float,
    target_balance: float,
    stake_percent: float = 0.02,
    max_rounds: int = 200,
):
    starting_balance = float(max(starting_balance, 1.0))
    target_balance = float(max(target_balance, starting_balance))
    stake_percent = float(np.clip(stake_percent, 0.001, 0.05))
    max_rounds = int(np.clip(max_rounds, 1, 1000))

    ordered = list(reversed(games))
    usable = [g for g in ordered if str(g.get("size","")).lower() in ("big","small")]
    if len(usable) < V93_MIN_TRAIN + 1:
        return {
            "ready": False,
            "reason": f"need at least {V93_MIN_TRAIN + 1} completed rounds",
            "starting_balance_pkr": round(starting_balance, 2),
            "target_balance_pkr": round(target_balance, 2),
        }

    balance = starting_balance
    peak = balance
    lowest = balance
    start_i = max(V93_MIN_TRAIN, len(usable) - max_rounds)
    rows = []
    predictions = wins = losses = skips = 0
    stop_reason = "history_exhausted"

    for i in range(start_i, len(usable)):
        # Only information available before this historical round is used.
        train_games = list(reversed(usable[:i]))
        result = build_v93_flash(train_games)
        actual = str(usable[i]["size"]).lower()

        if not result.get("ready") or result.get("signal") != "PREDICT":
            skips += 1
            rows.append({
                "issue": str(usable[i].get("issue","")),
                "action": "SKIP",
                "actual": actual,
                "balance_pkr": round(balance, 2),
            })
            continue

        prediction = str(result["prediction"]).lower()
        confidence = float(result.get("calibrated_confidence", 0.5))

        # Virtual fixed-fraction stake. This is deliberately capped and does
        # not use Kelly/Martingale or any live-wallet logic.
        virtual_stake = min(balance * stake_percent, max(balance, 0.0))
        won = prediction == actual

        if won:
            balance += virtual_stake
            wins += 1
        else:
            balance -= virtual_stake
            losses += 1
        predictions += 1

        peak = max(peak, balance)
        lowest = min(lowest, balance)

        rows.append({
            "issue": str(usable[i].get("issue","")),
            "prediction": prediction,
            "actual": actual,
            "confidence": round(confidence, 4),
            "virtual_stake_pkr": round(virtual_stake, 2),
            "result": "WIN" if won else "LOSS",
            "balance_pkr": round(balance, 2),
        })

        if balance >= target_balance:
            stop_reason = "target_reached"
            break
        if balance <= 0.01:
            balance = 0.0
            stop_reason = "virtual_balance_depleted"
            break

    hit_rate = (wins / predictions) if predictions else None
    pnl = balance - starting_balance
    return {
        "ready": True,
        "mode": "historical_virtual_simulation_only",
        "currency": "PKR",
        "starting_balance_pkr": round(starting_balance, 2),
        "target_balance_pkr": round(target_balance, 2),
        "ending_balance_pkr": round(balance, 2),
        "max_balance_reached_pkr": round(peak, 2),
        "min_balance_reached_pkr": round(lowest, 2),
        "profit_loss_pkr": round(pnl, 2),
        "return_percent": round((pnl / starting_balance) * 100.0, 2),
        "virtual_stake_percent": round(stake_percent * 100.0, 2),
        "predictions": predictions,
        "wins": wins,
        "losses": losses,
        "skips": skips,
        "hit_rate": None if hit_rate is None else round(hit_rate, 4),
        "stop_reason": stop_reason,
        "rounds_processed": len(rows),
        "history": rows[-250:],
        "note": "Backtest with virtual PKR only. It does not place bets or control a real wallet."
    }


app = FastAPI(
    title="WinGo Statistical Predictor API",
    version="9.4.0",
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
        "version": "9.3.0",
        "status": "online",
        "docs": "/docs",
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


@app.get("/api/stats")
def stats(window: int = 200):
    return engine.accuracy_stats(window)


INGEST_SECRET = os.getenv("INGEST_SECRET", "")

class RoundInput(BaseModel):
    issue: str
    number: int
    color: str
    secret: str


@app.get("/api/v9.4/flash")
@app.get("/api/v9.3/flash")
def v93_flash():
    history = engine.load_history(HISTORY_LIMIT)
    return build_v93_flash(history)


@app.get("/api/v9.4/simulate")
@app.get("/api/v9.3/simulate")
def v93_simulate(
    starting_balance: float = Query(..., gt=0, le=100000000),
    target_balance: float = Query(..., gt=0, le=1000000000),
    stake_percent: float = Query(2.0, ge=0.1, le=5.0),
    max_rounds: int = Query(200, ge=10, le=1000),
):
    history = engine.load_history(max(HISTORY_LIMIT, max_rounds + V93_MIN_TRAIN + 20))
    return simulate_v93_virtual_balance(
        history,
        starting_balance=starting_balance,
        target_balance=target_balance,
        stake_percent=stake_percent / 100.0,
        max_rounds=max_rounds,
    )


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

    restored_from_firestore = 0


    firebase_error = None


    try:


        # New Render deploys start with an empty ephemeral disk.


        # Rebuild the cache BEFORE inserting/predicting.


        if engine.count_rounds() == 0:


            restored_from_firestore = engine.hydrate_rounds_firestore(HISTORY_LIMIT)


    except Exception as exc:


        firebase_error = f"hydrate: {type(exc).__name__}: {exc}"



    inserted = engine.save_round(game)



    try:


        # Firestore is the permanent source of truth across deployments.


        engine.persist_round_firestore(game)


    except Exception as exc:


        firebase_error = f"persist: {type(exc).__name__}: {exc}"



    engine.verify_pending_predictions()


    history = engine.load_history(HISTORY_LIMIT)


    prediction = engine.save_next_prediction(history) if inserted else None

    return {
        "ok": True,
        "inserted": inserted,
        "round": game,
        "stored_rounds": engine.count_rounds(),
        "prediction": prediction,
        "restored_from_firestore": restored_from_firestore,
        "firebase_error": firebase_error,
        "storage": "firestore-persistent/sqlite-cache",
    }

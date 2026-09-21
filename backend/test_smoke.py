import asyncio
import importlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class BackendSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Windows may briefly retain a SQLite handle after TestClient closes.
        cls.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        os.environ["DB_PATH"] = str(Path(cls.temp_dir.name) / "test.db")
        os.environ["INGEST_SECRET"] = "test-secret"
        os.environ["ADMIN_RESET_SECRET"] = "test-reset-secret"
        os.environ["MIN_HISTORY"] = "50"
        sys.path.insert(0, str(Path(__file__).parent))
        cls.module = importlib.import_module("app")
        cls.module.engine.persist_round_firestore = lambda game: None
        cls.module.engine.hydrate_rounds_firestore = lambda limit: 0
        cls.module._persist_flash_live_firestore = lambda row: None
        cls.client = TestClient(cls.module.app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp_dir.cleanup()

    def post_round(self, issue, number, secret="test-secret"):
        return self.client.post(
            "/api/ingest",
            headers={"X-Ingest-Secret": secret},
            json={
                "issue": str(issue),
                "number": number,
                "color": "green" if number % 2 else "red",
            },
        )

    def test_1_ingest_live_flow_and_validation(self):
        self.assertEqual(self.post_round(1000, 0, "wrong").status_code, 401)
        self.assertEqual(
            self.client.post(
                "/api/ingest",
                headers={"X-Ingest-Secret": "test-secret"},
                json={"issue": "bad", "number": 1, "color": "green"},
            ).status_code,
            422,
        )

        original_build = self.module.build_v93_flash
        self.module.build_v93_flash = lambda games: {
            "ready": True,
            "signal": "PREDICT",
            "prediction": "big",
            "calibrated_confidence": 0.64,
            "quality": "STRONG",
        }
        try:
            for issue in range(1000, 1012):
                response = self.post_round(issue, issue % 10)
                self.assertEqual(response.status_code, 200, response.text)

            status = self.client.get("/api/v9.5/live-sim").json()
            self.assertEqual(status["pending_prediction"]["issue"], "1012")

            # A duplicate must still return/open the idempotent live prediction.
            duplicate = self.post_round(1011, 1)
            self.assertEqual(duplicate.status_code, 200)
            self.assertFalse(duplicate.json()["inserted"])
            self.assertEqual(duplicate.json()["live_flash_prediction"]["issue"], "1012")

            conflict = self.post_round(1011, 8)
            self.assertEqual(conflict.status_code, 409)

            settled = self.post_round(1012, 2)
            self.assertEqual(settled.status_code, 200)
            self.assertEqual(settled.json()["live_flash_settled"]["issue"], "1012")
            self.assertEqual(settled.json()["live_flash_prediction"]["issue"], "1013")
        finally:
            self.module.build_v93_flash = original_build

        simulation = self.client.get(
            "/api/v9.5/simulate",
            params={
                "starting_balance": 1000,
                "target_balance": 2000,
                "stake_mode": "flat",
                "flat_stake": 25,
                "stop_loss_percent": 20,
                "max_consecutive_losses": 3,
                "max_rounds": 10,
            },
        )
        self.assertEqual(simulation.status_code, 200, simulation.text)
        result = simulation.json()
        self.assertTrue(result["ready"])
        self.assertEqual(result["stake_mode"], "flat")
        self.assertEqual(result["flat_stake_pkr"], 25.0)
        self.assertIn("max_drawdown_percent", result)

    def test_2_internal_poller_also_advances_live_simulator(self):
        game = {
            "issue": "1013",
            "number": 3,
            "color": "green",
            "size": "small",
            "parity": "odd",
        }
        original_fetch = self.module.engine.fetch_history
        original_build = self.module.build_v93_flash
        try:
            self.module.build_v93_flash = lambda games: {
                "ready": True,
                "signal": "PREDICT",
                "prediction": "small",
                "calibrated_confidence": 0.63,
                "quality": "STRONG",
            }
            self.module.engine.fetch_history = lambda page_no=1, page_size=100: {
                "code": 0,
                "data": {
                    "list": [{"issueNumber": "1013", "number": 3, "color": "green"}]
                },
            }
            self.module.engine.poll_once()
        finally:
            self.module.engine.fetch_history = original_fetch
            self.module.build_v93_flash = original_build

        status = self.client.get("/api/v9.5/live-sim").json()
        self.assertEqual(status["pending_prediction"]["issue"], "1014")
        settled = next(row for row in status["history"] if row["issue"] == "1013")
        self.assertIn(settled["result"], ("WIN", "LOSS"))

    def test_3_walk_forward_has_strict_past_only_training(self):
        sequence = [index % 10 for index in range(12)]
        calls = []
        original = self.module.engine.candidate_number_distributions

        def spy(train):
            calls.append(tuple(train))
            return original(train)

        self.module.engine.candidate_number_distributions = spy
        try:
            self.module.engine.walk_forward_model_scores(
                sequence,
                lookback=5,
                min_train=3,
            )
        finally:
            self.module.engine.candidate_number_distributions = original

        # len=12/lookback=5 starts targets at index 7. Every call must receive
        # exactly the prefix before its target, never the target or its future.
        self.assertEqual(calls, [tuple(sequence[:index]) for index in range(7, 12)])

    def test_4_calibration_threshold_and_numerical_guards(self):
        below = [index % 2 for index in range(self.module.V93_ISOTONIC_MIN_SAMPLES + 9)]
        at_threshold = [index % 2 for index in range(self.module.V93_ISOTONIC_MIN_SAMPLES + 10)]
        self.assertIsNone(self.module._v93_walkforward(below)[0])
        self.assertIsNotNone(self.module._v93_walkforward(at_threshold)[0])
        self.assertEqual(self.module.safe_divide(1, 0, 0.5), 0.5)
        self.assertTrue(self.module.safe_negative_log_probability(0) > 0)

    def test_5_health_checks_database_and_worker(self):
        class AliveWorker:
            @staticmethod
            def is_alive():
                return True

        original_worker = self.module.engine.worker
        original_poll = self.module.engine.last_poll_at
        try:
            self.module.engine.worker = AliveWorker()
            self.module.engine.last_poll_at = int(time.time())
            response = asyncio.run(self.module.health())
        finally:
            self.module.engine.worker = original_worker
            self.module.engine.last_poll_at = original_poll

        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["checks"]["database"]["ok"])
        self.assertTrue(payload["checks"]["worker"]["ok"])

    def test_6_strict_signal_gate_and_choppy_weights(self):
        alternating = [index % 2 for index in range(120)]
        games = [
            {"issue": str(index), "size": "big" if value else "small"}
            for index, value in enumerate(alternating)
        ]
        result = self.module.build_v93_flash(games)
        self.assertEqual(result["threshold_used"], self.module.V93_CONFIDENCE_THRESHOLD)
        if result["calibrated_confidence"] < self.module.V93_CONFIDENCE_THRESHOLD:
            self.assertEqual(result["signal"], "SKIP")
            self.assertIn("confidence_below_58_percent", result["skip_reasons"])

        _, _, _, meta = self.module._v93_raw_ensemble(alternating)
        self.assertGreaterEqual(meta["choppy_score"], 0.70)
        self.assertGreater(meta["weights"]["reversion"], meta["weights"]["trend"])
        self.assertLessEqual(meta["weights"]["pattern"], 0.10)

        # Five consecutive Big outcomes should push the reversion model to Small.
        streak_sequence = ([0, 1] * 40) + [1, 1, 1, 1, 1]
        self.assertLess(self.module._v93_reversion_model(streak_sequence), 0.5)

    def test_7_admin_reset_requires_secret_and_clears_local_data(self):
        unauthorized = self.client.post(
            "/api/admin/reset-data",
            json={"confirmation": "DELETE ALL DATA"},
            headers={"X-Admin-Secret": "wrong"},
        )
        self.assertEqual(unauthorized.status_code, 401)

        original_flag = self.module.ENABLE_INTERNAL_COLLECTOR
        self.module.ENABLE_INTERNAL_COLLECTOR = False
        try:
            response = self.client.post(
                "/api/admin/reset-data",
                json={"confirmation": "DELETE ALL DATA"},
                headers={"X-Admin-Secret": "test-reset-secret"},
            )
        finally:
            self.module.ENABLE_INTERNAL_COLLECTOR = original_flag
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["ok"])
        self.assertEqual(self.module.engine.count_rounds(), 0)
        self.assertEqual(self.client.get("/api/v9.5/live-sim").json()["settled"], 0)


if __name__ == "__main__":
    unittest.main()

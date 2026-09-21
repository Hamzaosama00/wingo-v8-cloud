import importlib
import os
import sys
import tempfile
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

    def test_2_internal_poller_also_advances_live_simulator(self):
        game = {
            "issue": "1013",
            "number": 3,
            "color": "green",
            "size": "small",
            "parity": "odd",
        }
        original_fetch = self.module.engine.fetch_history
        try:
            self.module.engine.fetch_history = lambda page_no=1, page_size=100: {
                "code": 0,
                "data": {
                    "list": [{"issueNumber": "1013", "number": 3, "color": "green"}]
                },
            }
            self.module.engine.poll_once()
        finally:
            self.module.engine.fetch_history = original_fetch

        status = self.client.get("/api/v9.5/live-sim").json()
        self.assertEqual(status["pending_prediction"]["issue"], "1014")
        settled = next(row for row in status["history"] if row["issue"] == "1013")
        self.assertIn(settled["result"], ("WIN", "LOSS"))


if __name__ == "__main__":
    unittest.main()

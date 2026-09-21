import asyncio
import importlib.util
import os
import unittest
from pathlib import Path


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class CollectorSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["BACKEND_URL"] = "https://backend.example"
        os.environ["INGEST_SECRET"] = "test-secret"
        path = Path(__file__).with_name("collector.py")
        spec = importlib.util.spec_from_file_location("wingo_collector", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def setUp(self):
        self.module.seen.clear()
        self.posts = []

        class FakeSource:
            async def get(inner_self, *args, **kwargs):
                return FakeResponse({
                    "code": 0,
                    "data": {
                        "list": [
                            {"issueNumber": "102", "number": "2", "color": "red"},
                            {"issueNumber": "bad-number", "number": "x", "color": "green"},
                            {"issueNumber": "101", "number": "1", "color": "green"},
                        ]
                    },
                })

        outer = self

        class FakeBackend:
            async def post(inner_self, url, json, headers):
                outer.posts.append((url, json, headers))
                return FakeResponse({"ok": True, "stored_rounds": len(outer.posts)})

        self.source = FakeSource()
        self.backend = FakeBackend()

    def test_oldest_first_header_auth_and_invalid_source_skip(self):
        asyncio.run(self.module.poll_once(self.source, self.backend))
        self.assertEqual([post[1]["issue"] for post in self.posts], ["101", "102"])
        self.assertNotIn("secret", self.posts[0][1])
        self.assertEqual(self.posts[0][2]["X-Ingest-Secret"], "test-secret")
        self.assertIn("bad-number", self.module.seen)


if __name__ == "__main__":
    unittest.main()

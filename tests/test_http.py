import copy
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from support import DEFAULT
from lazyrig.server import handler
from support import Service

class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        config = copy.deepcopy(DEFAULT)
        config["data_dir"] = self.temp.name
        self.service = Service(config)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.service))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)
        self.service.close()
        self.temp.cleanup()

    def request(self, path, body=None, headers=None):
        return urllib.request.urlopen(urllib.request.Request(self.base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})}), timeout=15)

    def json(self, path, body=None):
        with self.request(path, body) as result:
            return json.load(result)

    def test_http_sse_image_roundtrip_and_static_assets(self):
        sid = self.json("/api/sessions", {"title": "HTTP test"})["id"]
        body = {"session_id": sid, "model": "qwen-chat", "image_model": "qwen-image",
                "prompt": "Generate an image", "max_tokens": 1024, "steps": 2}
        context = self.json("/api/context", {**body, "draft": body["prompt"]})
        self.assertEqual(context["accuracy"], "estimated-test")
        jid = self.json("/api/jobs", body)["id"]
        with self.request("/api/jobs/" + jid + "/events") as response:
            self.assertEqual(response.headers["Content-Type"], "text/event-stream")
            events = response.read().decode()
        self.assertIn('"kind": "artifact"', events)
        self.assertIn('"kind": "finished"', events)
        self.assertIn("event: end", events)
        result = self.json("/api/jobs/" + jid)
        self.assertEqual(result["status"], "completed", result["error"])
        self.assertEqual(len(self.json("/api/sessions/" + sid)["messages"]), 4)
        with self.request(result["artifacts"][0]["url"]) as response:
            self.assertIn(b"TEST OUTPUT", response.read())
        for path in ("/", "/app.js", "/style.css", "/ui.js", "/redesign.css", "/favicon.svg"):
            with self.request(path) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(len(response.read()), 10)

    def test_cross_origin_and_traversal_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/api/sessions", {}, {"Origin": "https://unrelated.example"})
        self.assertEqual(error.exception.code, 400)
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.request("/artifacts/../state.sqlite3")
        self.assertEqual(error.exception.code, 404)

if __name__ == "__main__":
    unittest.main(verbosity=2)

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from support import DEFAULT
from lazyrig.context import parse_image_tool
from support import TokenizerService
from lazyrig.scheduler import Scheduler
from support import Service
from lazyrig.service import TERMINAL
from lazyrig.storage import Store

def wait_job(service, jid, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = service.store.job(jid)
        if job["status"] in TERMINAL and jid not in service.running:
            return job
        time.sleep(.03)
    raise AssertionError("Job did not finish")

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = copy.deepcopy(DEFAULT)
        self.config["data_dir"] = self.temp.name
        self.service = Service(self.config)
        self.session = self.service.store.session()

    def tearDown(self):
        self.service.close()
        self.temp.cleanup()

    def submit(self, prompt="Hello", **extra):
        return self.service.submit({"session_id": self.session, "model": "qwen-chat", "image_model": "qwen-image",
                                    "prompt": prompt, "max_tokens": 1024, "steps": 2, **extra})

    def test_image_roundtrip_unloads_before_loading_and_retains_history(self):
        job = wait_job(self.service, self.submit("Generate an image of a quiet city")["id"])
        self.assertEqual(job["status"], "completed", job["error"])
        lifecycle = [(e["kind"], e.get("model")) for e in self.service.store.events(job["id"])
                     if e["kind"] in {"loading", "unloaded"}]
        self.assertEqual(lifecycle, [("loading", "qwen-chat"), ("unloaded", "qwen-chat"),
            ("loading", "qwen-image"), ("unloaded", "qwen-image"), ("loading", "qwen-chat")])
        messages = self.service.store.messages(self.session)
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], messages[2]["tool_call_id"])
        self.assertEqual(len(job["artifacts"]), 1)
        self.assertTrue((Path(self.temp.name) / "artifacts" / job["artifacts"][0]["name"]).is_file())
        self.assertEqual(job["context"]["accuracy"], "estimated-test")
        self.assertIsNone(job["context"]["cache_occupancy"])
        self.assertEqual(set(self.service.scheduler.resident), {"qwen-chat"})

    def test_context_overflow_rejected_before_model_load_or_history_mutation(self):
        self.service.config["models"][0]["context_limit"] = 512
        job = wait_job(self.service, self.submit("x" * 8000, max_tokens=32)["id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("Context overflow", job["error"])
        self.assertEqual(self.service.store.messages(self.session), [])
        self.assertFalse(self.service.scheduler.resident)

    def test_memory_admission_failure_is_visible(self):
        self.service.config["devices"][0]["budget_gib"] = 10
        job = wait_job(self.service, self.submit()["id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("exceeding", job["error"])
        self.assertFalse(self.service.scheduler.resident)

    def test_cancel_during_image_generation_releases_worker(self):
        jid = self.submit("Draw an image", steps=100)["id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.service.store.job(jid).get("progress"):
                break
            time.sleep(.03)
        self.assertTrue(self.service.store.job(jid).get("progress"))
        self.service.cancel(jid)
        job = wait_job(self.service, jid)
        self.assertEqual(job["status"], "cancelled")
        self.assertFalse(self.service.scheduler.resident)
        messages = self.service.store.messages(self.session)
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertFalse(json.loads(messages[-1]["content"])["ok"])

    def test_same_session_cannot_submit_twice_and_settings_are_guarded(self):
        jid = self.submit()["id"]
        with self.assertRaisesRegex(ValueError, "active job"):
            self.submit()
        with self.assertRaisesRegex(ValueError, "active or queued"):
            self.service.settings({"model": "qwen-chat", "memory_gib": 15})
        self.service.cancel(jid)
        wait_job(self.service, jid)

    def test_events_replay_and_saved_conversation_survive_restart(self):
        job = wait_job(self.service, self.submit()["id"])
        all_events = self.service.store.events(job["id"])
        after = all_events[2]["seq"]
        self.assertEqual(self.service.store.events(job["id"], after), all_events[3:])
        before = self.service.store.messages(self.session)
        self.service.close()
        self.service = Service(self.config)
        self.assertEqual(self.service.store.messages(self.session), before)
        self.assertEqual(self.service.store.job(job["id"])["status"], "completed")
        self.assertFalse(self.service.scheduler.resident)

    def test_restart_marks_active_job_interrupted_without_replay(self):
        job = {"id": "a" * 32, "status": "running", "session_id": self.session, "artifacts": []}
        self.service.store.save_job(job)
        self.service.close()
        self.service = Service(self.config)
        self.assertEqual(self.service.store.job(job["id"])["status"], "interrupted")
        self.assertFalse(self.service.running)

    def test_artifact_survives_llm_restore_failure(self):
        original = self.service.scheduler.ensure
        counts = {"llm": 0}
        def fail_restore(model, cancelled, emit):
            if model["id"] == "qwen-chat":
                counts["llm"] += 1
                if counts["llm"] == 2:
                    raise RuntimeError("Injected restore failure")
            return original(model, cancelled, emit)
        self.service.scheduler.ensure = fail_restore
        job = wait_job(self.service, self.submit("Generate an image")["id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("restore failure", job["error"])
        self.assertEqual(len(job["artifacts"]), 1)
        self.assertTrue((Path(self.temp.name) / "artifacts" / job["artifacts"][0]["name"]).exists())

class SchedulerTests(unittest.TestCase):
    def test_conflicting_jobs_are_fifo_and_other_devices_can_progress(self):
        scheduler = Scheduler(copy.deepcopy(DEFAULT))
        cancel = threading.Event()
        order = []
        release = threading.Event()
        entered = threading.Event()
        def first():
            with scheduler.lease(["A"], cancel):
                entered.set()
                release.wait(3)
        def later(name, device):
            with scheduler.lease([device], cancel):
                order.append(name)
        t1 = threading.Thread(target=first); t1.start(); self.assertTrue(entered.wait(2))
        t2 = threading.Thread(target=later, args=("second", "A")); t2.start()
        deadline = time.monotonic() + 2
        while not scheduler.waiting and time.monotonic() < deadline:
            time.sleep(.005)
        t3 = threading.Thread(target=later, args=("third", "A")); t3.start()
        t4 = threading.Thread(target=later, args=("independent", "B")); t4.start(); t4.join(2)
        self.assertEqual(order, ["independent"])
        release.set()
        for t in (t1, t2, t3): t.join(2)
        self.assertEqual(order, ["independent", "second", "third"])

    def test_budget_keeps_models_only_when_combined_reservations_fit(self):
        class FakeWorker:
            def __init__(self, *args): self.process = self
            def is_alive(self): return True
            def close(self): pass
        config = copy.deepcopy(DEFAULT)
        config["devices"][0].update(policy="budget", budget_gib=40)
        s = Scheduler(config, FakeWorker)
        cancel = threading.Event()
        emit = lambda *a, **k: None
        for model in config["models"]: s.ensure(model, cancel, emit)
        self.assertEqual(set(s.resident), {"qwen-chat", "qwen-image"})
        s.close()
        config["devices"][0]["budget_gib"] = 22
        for model in config["models"]: s.ensure(model, cancel, emit)
        self.assertEqual(set(s.resident), {"qwen-image"})

class ContextTests(unittest.TestCase):
    def test_tools_and_template_are_counted_and_overflow_is_not_truncated(self):
        t = TokenizerService("openvino")
        model = DEFAULT["models"][0]
        history = [{"role": "user", "content": "hello"}]
        with_tools = t.prepare(model, history, 200, True)
        without = t.prepare(model, history, 200, False)
        self.assertGreater(with_tools["context"]["input_tokens"], without["context"]["input_tokens"])
        self.assertNotEqual(with_tools["context"]["prompt_sha256"], without["context"]["prompt_sha256"])

    def test_only_complete_allowed_tool_calls_execute(self):
        tool = '<tool_call>{"name":"generate_image","arguments":{"prompt":"A tree"}}</tool_call>'
        self.assertEqual(parse_image_tool(tool), {"prompt": "A tree"})
        self.assertIsNone(parse_image_tool('<think>'+tool+'</think>Normal reply'))
        self.assertIsNone(parse_image_tool('Here is an example: '+tool))
        with self.assertRaises(ValueError): parse_image_tool(tool.replace('generate_image','shell'))
        with self.assertRaises(ValueError): parse_image_tool(tool[:-4])

if __name__ == "__main__":
    unittest.main(verbosity=2)

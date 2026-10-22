"""Optional per-pass timings measured inside the model worker."""
import math
import time


class GenerationTimer:
    def __init__(self, prompt_tokens, source="worker_clock", clock=time.perf_counter):
        self.clock = clock
        self.start = clock()
        self.first = self.end = None
        self.first_count = 0
        self.prompt_tokens = prompt_tokens
        self.source = source

    def observe(self, generated):
        if generated and self.first is None:
            self.first, self.first_count = self.clock(), generated

    def finish(self):
        self.end = self.clock()

    def snapshot(self, generated, prompt_tokens=None, perf=None):
        now = self.end if self.end is not None else self.clock()
        prompt_n = self.prompt_tokens if prompt_tokens is None else prompt_tokens
        prompt_ms = max(0.0, (self.first - self.start) * 1000) if self.first is not None else None
        predicted_ms = max(0.0, (now - self.first) * 1000) if self.first is not None else 0.0
        decode_n = max(0, generated - self.first_count)
        result = {
            "prompt_n": prompt_n, "predicted_n": generated,
            "prompt_ms": prompt_ms, "predicted_ms": predicted_ms,
            "prompt_per_token_ms": prompt_ms / prompt_n if prompt_ms is not None and prompt_n else None,
            "prompt_per_second": prompt_n * 1000 / prompt_ms if prompt_ms and prompt_n else None,
            "predicted_per_token_ms": predicted_ms / decode_n if decode_n and predicted_ms else None,
            "predicted_per_second": decode_n * 1000 / predicted_ms if decode_n and predicted_ms else None,
            "x_generation_ms": max(0.0, (now - self.start) * 1000),
            "x_decode_tokens": decode_n, "x_source": self.source,
            "x_prompt_ms_basis": "time_to_first_token_callback",
            "x_scope": "current_text_pass",
        }
        # Native metrics are kept separate from the live wall-clock estimates.
        # Missing metrics in older runtimes must not fail a successful generation.
        if perf is not None:
            native = {}
            for name, method in (("ttft_ms", "get_ttft"), ("tpot_ms", "get_tpot"),
                                 ("tokens_per_second", "get_throughput"), ("generate_ms", "get_generate_duration")):
                try:
                    value = float(getattr(perf, method)().mean)
                    if math.isfinite(value) and value >= 0:
                        native[name] = value
                except Exception:
                    continue
            if native:
                result["x_openvino_metrics"] = native
        return result

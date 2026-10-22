"""Render exported HF templates with Jinja; OpenVINO still owns tokenization."""
import json
import copy
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _choose(value, tools):
    if isinstance(value, list):
        value = {item["name"]: item["template"] for item in value}
    if isinstance(value, dict):
        name = "tool_use" if tools and "tool_use" in value else "default"
        if name not in value:
            raise ValueError("The export has named chat templates but no default template")
        value = value[name]
    return value if isinstance(value, str) else ""


def template_source(path, tokenizer, tools):
    root = Path(path)
    metadata = _json(root / "tokenizer_config.json")
    if tools and (root / "chat_templates" / "tool_use.jinja").is_file():
        source = (root / "chat_templates" / "tool_use.jinja").read_text(encoding="utf-8")
    elif (root / "chat_template.jinja").is_file():
        source = (root / "chat_template.jinja").read_text(encoding="utf-8")
    else:
        source = _choose(metadata.get("chat_template"), tools)
        if not source:
            source = _choose(_json(root / "chat_template.json").get("chat_template"), tools)
        if not source and hasattr(tokenizer, "get_original_chat_template"):
            source = tokenizer.get_original_chat_template()
    if not source:
        raise ValueError("No chat template in this export. Add its original chat_template.jinja to the model folder.")
    special = {**metadata, **_json(root / "special_tokens_map.json")}
    tokens = {}
    for key, value in special.items():
        if key.endswith("_token"):
            tokens[key] = value.get("content", "") if isinstance(value, dict) else value
        elif key == "additional_special_tokens" and isinstance(value, list):
            tokens[key] = [x.get("content", "") if isinstance(x, dict) else x for x in value]
    for name in ("bos_token", "eos_token", "pad_token"):
        method = getattr(tokenizer, "get_" + name, None)
        if name not in tokens and method:
            tokens[name] = method()
    return source, tokens


@lru_cache(maxsize=64)
def compile_template(source):
    try:
        from jinja2 import nodes, TemplateError
        from jinja2.ext import Extension
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as exc:
        raise ValueError("Jinja2 is missing. Run setup-windows.bat (or setup.sh), then restart the server.") from exc

    class GenerationBlock(Extension):
        tags = {"generation"}

        def parse(self, parser):
            token = next(parser.stream)
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return nodes.CallBlock(self.call_method("render_body"), [], [], body).set_lineno(token.lineno)

        def render_body(self, caller):
            return caller()

    def fail(message):
        raise TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                       extensions=["jinja2.ext.loopcontrols", GenerationBlock])
    env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=kwargs.pop("ensure_ascii", False), **kwargs)
    env.globals.update(raise_exception=fail, strftime_now=lambda fmt: datetime.now().strftime(fmt))
    return env.from_string(source)


def render_chat_details(path, tokenizer, messages, tools=None, extra_context=None):
    source, tokens = template_source(path, tokenizer, tools)
    options = copy.deepcopy(extra_context or {})
    notes = []
    # This normalization is tied to the template's declared supported levels,
    # not an alias or model name, so unrelated templates keep their own levels.
    levels = re.search(r"resolved_reasoning_effort\s+not\s+in\s+\(([^)]+)\)", source)
    if levels and set(re.findall(r"['\"]([^'\"]+)['\"]", levels[1])) == {"low", "medium", "xhigh"}:
        effort = options.get("reasoning_effort")
        normalized = {"minimal": "low", "high": "xhigh"}.get(effort, effort)
        if normalized != effort:
            notes.append(f"This template supports low, medium and xhigh; {effort} maps to {normalized}.")
            options["reasoning_effort"] = normalized
    messages = copy.deepcopy(messages)
    # OpenAI transports arguments as JSON strings; HF templates expect objects.
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", call)
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    context = {**tokens, **options, "messages": messages,
               "tools": tools, "add_generation_prompt": True}
    try:
        return {"text": compile_template(source).render(**context), "template_context": options, "template_notes": notes}
    except Exception as exc:
        raise ValueError("Cannot render this model's chat template: " + str(exc)) from exc


def render_chat(path, tokenizer, messages, tools=None, extra_context=None):
    return render_chat_details(path, tokenizer, messages, tools, extra_context)["text"]

"""Normalize complete JSON and Qwen XML-style tool calls before dispatch."""
import ast
import io
import json
import math
import re
import tokenize
from .reasoning import END_PATTERN


class ToolCallError(ValueError):
    def __init__(self, message, code="invalid_tool_call", incomplete=False):
        super().__init__(message + "; no tool was executed")
        self.code, self.incomplete = code, incomplete


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallError("Duplicate tool argument: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ToolCallError("Non-finite tool argument: " + value)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        return _constant(value)
    return result


DECODER = json.JSONDecoder(object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)


def _quoted_prefix(prefix):
    """Is this marker inside literal Markdown/quotes, rather than after them?"""
    fence, inline = None, None
    for token in re.finditer(r"(?m)^ {0,3}(`{3,}|~{3,})([^\n]*)|(`+)", prefix):
        if token[1]:
            marker = token[1]
            if fence:
                if marker[0] == fence[0] and len(marker) >= len(fence) and not token[2].strip():
                    fence = None
            elif inline is None:
                fence = marker
        elif fence is None:
            ticks = len(token[3])
            inline = None if ticks == inline else (ticks if inline is None else inline)
    if fence or inline:
        return True
    line = prefix.rsplit("\n", 1)[-1]
    if re.match(r"\s*>| {4,}\S*\s*$", line):
        return True
    if re.search(r"\b(?:example|ejemplo|sample)\s*:\s*$", prefix, re.I):
        return True
    # A quoted topic in an earlier sentence does not quote the subsequent call.
    for quote in ('"', "'"):
        if line.rstrip().endswith(quote) and len(re.findall(r"(?<!\\)" + quote, line)) % 2:
            return True
    return False


def _reasoning(text, thinking_open):
    clean, reasoning = text.rstrip(), []
    if thinking_open and not clean.lstrip().startswith("<think>"):
        end = clean.find("</think>")
        if end < 0:
            raise ToolCallError("The LLM stopped inside its reasoning", "incomplete_reasoning", True)
        reasoning.append(clean[:end]); clean = clean[end + len("</think>"):].lstrip("\r\n")
    while clean.lstrip().startswith("<think>"):
        clean = clean.lstrip()
        end = clean.find("</think>")
        if end < 0:
            raise ToolCallError("The LLM stopped inside its reasoning", "incomplete_reasoning", True)
        reasoning.append(clean[len("<think>"):end]); clean = clean[end + len("</think>"):].lstrip("\r\n")
    return clean, "\n".join(reasoning).strip()


def _parameter_types(schema):
    """Read explicit primitive types, including simple unions; this is not a schema validator."""
    if not isinstance(schema, dict):
        return None
    known = {"string", "object", "array", "integer", "number", "boolean", "null"}
    kind = schema.get("type")
    if isinstance(kind, str):
        return {kind} if kind in known else None
    if isinstance(kind, list) and kind and all(isinstance(t, str) and t in known for t in kind):
        return set(kind)
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword)
        if isinstance(branches, list) and branches:
            types = [_parameter_types(branch) for branch in branches]
            if all(t is not None for t in types):
                return set().union(*types)
    return None


def _literal_collection(value):
    """Translate a limited JSON/Python data literal, then use the strict decoder.

    Tokenization protects quoted text from replacements. Only isolated string
    literals reach literal_eval; expressions, calls, comments and unquoted keys
    never do. Re-decoding preserves duplicate-key and finite-number checks.
    """
    if len(value) > 1024 * 1024 or not value.startswith(("[", "{")):
        raise ValueError("Expected a bounded array or object literal")
    literals = {"true": "true", "false": "false", "null": "null",
                "True": "true", "False": "false", "None": "null"}
    output, depth = [], 0
    for count, token in enumerate(tokenize.generate_tokens(io.StringIO(value).readline)):
        if count >= 100000:
            raise ValueError("Too many structured-value tokens")
        kind, text = token.type, token.string
        if kind in {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER}:
            continue
        if kind == tokenize.STRING:
            if not text.startswith(("'", '"')) or text.startswith(("'''", '"""')):
                raise ValueError("Unsupported string literal")
            try:
                DECODER.decode(text)  # Retain JSON escapes, including \\/, as written.
            except json.JSONDecodeError:
                text = json.dumps(ast.literal_eval(text))
        elif kind == tokenize.NAME and text in literals:
            text = literals[text]
        elif kind == tokenize.NUMBER and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", text):
            if output and output[-1] == "-":
                output.pop()
                text = "-" + text
        elif kind == tokenize.OP and text in {"[", "]", "{", "}", ":", ",", "-"}:
            depth += int(text in {"[", "{"}) - int(text in {"]", "}"})
            if not 0 <= depth <= 64:
                raise ValueError("Structured value nesting is too deep")
            if (text in {"]", "}"} and len(output) >= 2 and output[-1] == ","
                    and output[-2] not in {"[", "{", ":", ",", "-"}):
                output.pop()  # One trailing comma after a value, never an empty item.
        else:
            raise ValueError("Unsupported structured-value token")
        output.append(text)
    return DECODER.decode(" ".join(output))


def _xml_collection(value):
    # Some models wrap the complete parameter in a code fence or JSON string.
    # Unwrap at most one of each; never repair missing content or delimiters.
    fence = re.fullmatch(r"```(?:json|python)?\s*\n(.*?)\n```", value, re.S | re.I)
    if fence:
        value = fence[1].strip()
    try:
        parsed = DECODER.decode(value)
    except json.JSONDecodeError:
        return _literal_collection(value)
    if isinstance(parsed, str):
        value = parsed.strip()
        try:
            unwrapped = DECODER.decode(value)
        except json.JSONDecodeError:
            unwrapped = _literal_collection(value)
        if isinstance(unwrapped, (dict, list)):
            return unwrapped
    if fence and not isinstance(parsed, (dict, list)):
        raise ValueError("Expected an array or object inside the fence")
    return parsed


def _xml_value(value, schema, key):
    types = _parameter_types(schema)
    if types == {"string"}:
        return value
    # Qwen XML can contain Python-style True/False, sometimes quoted. Normalize
    # only whole Boolean literals where the schema rules out a string. Never use
    # bool(text), eval, or replacements inside commands or prompts.
    if types and "boolean" in types and "string" not in types:
        literal = value
        if len(literal) >= 2 and literal[0] in ('"', "'") and literal[-1] == literal[0]:
            literal = literal[1:-1]
        if literal.lower() in {"true", "false"}:
            return literal.lower() == "true"
    structured = types and types.intersection({"array", "object"}) and "string" not in types
    try:
        parsed = _xml_collection(value) if structured else DECODER.decode(value)
    except ToolCallError:
        raise
    except (ValueError, SyntaxError, tokenize.TokenError, RecursionError) as exc:
        if types and "string" not in types:
            raise ToolCallError("Invalid value for parameter " + key +
                                " (expected " + " or ".join(sorted(types)) + ")") from exc
        parsed = value
    valid = {"string": isinstance(parsed, str), "object": isinstance(parsed, dict),
             "array": isinstance(parsed, list), "integer": type(parsed) is int,
             "number": type(parsed) in {int, float}, "boolean": type(parsed) is bool,
             "null": parsed is None}
    if types and not any(valid[t] for t in types):
        raise ToolCallError("Incorrect type for parameter " + key +
                            " (expected " + " or ".join(sorted(types)) + ")")
    return parsed


def _xml_arguments(body, schema):
    values = {}
    properties = schema.get("properties", {})
    while body.strip():
        body = body.lstrip()
        start = re.match(r"<parameter=([A-Za-z0-9_.:-]+)>", body)
        if not start:
            raise ToolCallError("Invalid Qwen parameter block")
        key = start[1]
        if key in values:
            raise ToolCallError("Duplicate tool argument: " + key)
        end = body.find("</parameter>", start.end())
        if end < 0:
            raise ToolCallError("The LLM stopped before closing a tool parameter", "incomplete_tool_call", True)
        value = body[start.end():end].strip()
        if re.search(r"</?(?:tool_call|function|parameter)(?:[=>])", value):
            raise ToolCallError("Nested tool markup is not a parameter value")
        values[key] = _xml_value(value, properties.get(key, {}), key)
        body = body[end + len("</parameter>"):]
    return values


def parse_tool_calls(text, tools, thinking_open=False):
    """Allow natural-language preambles; leave quoted examples as ordinary text."""
    clean, reasoning = _reasoning(text, thinking_open)
    # Some tokenizers retain the final assistant-turn marker.
    clean = END_PATTERN.sub("", clean).rstrip()
    result = {"content": clean, "reasoning_content": reasoning, "calls": []}
    candidates = [match.start() for match in re.finditer(re.escape("<tool_call>"), clean)
                  if not _quoted_prefix(clean[:match.start()])]
    if not candidates:
        partial = clean.find("<tool_call")
        if partial >= 0 and "<tool_call>" not in clean and not _quoted_prefix(clean[:partial]):
            raise ToolCallError("The LLM stopped inside a tool-call marker", "incomplete_tool_call", True)
        # Some exported templates emit a complete bare JSON envelope. Accept
        # only the whole response, never a JSON snippet embedded in prose or a
        # quoted example. The declared function name remains authoritative.
        bare = clean.strip()
        if bare.startswith('{'):
            try:
                # Inspect only the envelope here. Re-decode recognized calls
                # strictly below, so ordinary JSON objects stay ordinary text.
                call = json.loads(bare)
            except json.JSONDecodeError as exc:
                if re.match(r'\{\s*"(?:name|function|arguments|args)"\s*:', bare):
                    raise ToolCallError("Incomplete or invalid JSON tool call", "incomplete_tool_call", True) from exc
                return result
            if (set(call) == {'name', 'arguments'} or set(call) == {'function', 'args'} or
                    set(call) <= {'id', 'type', 'function'} and isinstance(call.get('function'), dict)):
                call = DECODER.decode(bare)
                if isinstance(call.get('function'), str) and set(call) == {'function', 'args'}:
                    call = {'name': call['function'], 'arguments': call['args']}
                return parse_tool_calls('<tool_call>' + json.dumps(call) + '</tool_call>', tools) | {
                    'reasoning_content': reasoning}
        return result
    start = candidates[0]
    prefix = clean[:start].strip()
    if prefix.rfind("<think>") > prefix.rfind("</think>"):
        raise ToolCallError("Tool call is mixed with unfinished reasoning", "incomplete_reasoning", True)
    declared = {tool["function"]["name"]: tool["function"].get("parameters", {}) for tool in tools}
    remaining = clean[start:]
    while remaining:
        if not remaining.startswith("<tool_call>"):
            raise ToolCallError("Unexpected text after the tool call")
        body = remaining[len("<tool_call>"):].lstrip()
        if body.startswith("{"):
            try:
                call, end = DECODER.raw_decode(body)
            except json.JSONDecodeError as exc:
                raise ToolCallError("Incomplete or invalid JSON tool call", "incomplete_tool_call",
                                    "</tool_call>" not in body) from exc
            body = body[end:].lstrip()
            if isinstance(call.get("function"), dict):
                call = call["function"]
            name, arguments = call.get("name"), call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = DECODER.decode(arguments)
                except json.JSONDecodeError as exc:
                    raise ToolCallError("Tool arguments are not valid JSON") from exc
        elif body.startswith("<function="):
            opening = re.match(r"<function=([A-Za-z0-9_.:-]+)>", body)
            if not opening:
                raise ToolCallError("Invalid Qwen function marker", "incomplete_tool_call", ">" not in body)
            name = opening[1]
            end = body.find("</function>", opening.end())
            if end < 0:
                raise ToolCallError("The LLM stopped before closing its function call", "incomplete_tool_call", True)
            arguments = _xml_arguments(body[opening.end():end], declared.get(name, {}))
            body = body[end + len("</function>"):].lstrip()
        else:
            raise ToolCallError("Expected a JSON or Qwen XML-style function call", "incomplete_tool_call", not body)
        if not isinstance(name, str) or name not in declared:
            raise ToolCallError("Unsupported tool: " + str(name), "undeclared_tool")
        if not isinstance(arguments, dict):
            raise ToolCallError("Tool arguments must be an object")
        if not body.startswith("</tool_call>"):
            raise ToolCallError("The LLM did not close its tool call", "incomplete_tool_call", not body or "</tool_call>" not in body)
        result["calls"].append({"name": name, "arguments": arguments})
        remaining = body[len("</tool_call>"):].strip()
    result["content"] = prefix
    return result

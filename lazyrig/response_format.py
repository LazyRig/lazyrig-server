"""OpenAI response formats mapped to native OpenVINO JSON constraints."""
import copy
import json
import re


class ResponseFormatError(ValueError):
    """Completed inference returned invalid output; the worker itself is healthy."""

    code = 'invalid_response_format'


def validate_format(value):
    if value is None:
        return {"type": "text"}
    if not isinstance(value, dict) or value.get("type") not in ("text", "json_object", "json_schema"):
        raise ValueError("response_format must specify type text, json_object or json_schema")
    if set(value) - {"type", "json_schema"}:
        raise ValueError("Unsupported response_format options")
    if value["type"] != "json_schema":
        if "json_schema" in value:
            raise ValueError("json_schema requires response_format.type=json_schema")
        return copy.deepcopy(value)
    spec = value.get("json_schema")
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), (dict, bool)):
        raise ValueError("response_format.json_schema requires a schema object or boolean")
    if set(spec) - {"name", "description", "schema", "strict"}:
        raise ValueError("Unsupported response_format.json_schema options")
    if not isinstance(spec.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", spec["name"]):
        raise ValueError("response_format.json_schema.name must be 1–64 letters, numbers, underscores or hyphens")
    if spec.get("strict") is not None and not isinstance(spec["strict"], bool):
        raise ValueError("response_format.json_schema.strict must be a boolean")
    if "description" in spec and not isinstance(spec["description"], str):
        raise ValueError("response_format.json_schema.description must be a string")
    try:
        json.dumps(spec["schema"], allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("response_format schema must contain finite JSON values") from exc
    return copy.deepcopy(value)


def is_json(value):
    return isinstance(value, dict) and value.get("type") in ("json_object", "json_schema")


def schema_for(value):
    return value["json_schema"]["schema"] if value["type"] == "json_schema" else {"type": "object"}


def format_messages(messages, value):
    if not is_json(value):
        return messages
    messages = copy.deepcopy(messages)
    instruction = "Return only JSON matching this schema, without commentary or code fences:\n" + json.dumps(schema_for(value), ensure_ascii=False)
    if messages and messages[0]["role"] == "system":
        # Preserve caller-supplied system instructions byte for byte. Put the
        # schema reminder in the last input, including multimodal requests.
        content = messages[-1].get('content')
        if isinstance(content, list):
            content.append({'type': 'text', 'text': instruction})
        elif isinstance(content, str):
            messages[-1]['content'] = content + '\n\n' + instruction
        else:
            messages.append({'role': 'user', 'content': instruction})
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


def configure_output(genai, config, value):
    if not hasattr(config, "structured_output_config"):
        if is_json(value):
            raise ValueError("This OpenVINO GenAI runtime lacks structured output. Run setup-windows.bat and restart.")
        return
    # Always clear the previous constraint, including on a reused pipeline.
    if not is_json(value):
        config.structured_output_config = None
        return
    config.structured_output_config = genai.StructuredOutputConfig(json_schema=json.dumps(schema_for(value)))


def check_json_output(value, text, limited=False):
    if not is_json(value):
        return None
    def invalid(constant):
        raise ValueError("Non-finite JSON number")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field: ' + key)
            result[key] = value
        return result
    try:
        parsed = json.loads(text, parse_constant=invalid, object_pairs_hook=unique)
        json.dumps(parsed, allow_nan=False)
        if value["type"] == "json_object" and not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, TypeError) as exc:
        if limited:
            return False  # Partial JSON is accompanied by finish_reason=length.
        raise ResponseFormatError("Model did not produce valid JSON for response_format: " + str(exc)) from exc
    return True

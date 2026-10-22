"""Split cumulative Qwen/DeepSeek think output without leaking partial markers."""
import re

OPEN, CLOSE = "<think>", "</think>"
ENDS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>")
END_PATTERN = re.compile(r"(?:(?:" + "|".join(re.escape(x) for x in ENDS) + r")\s*)+$")


def _without_partial_marker(text, markers):
    keep = max((size for marker in markers for size in range(1, len(marker))
                if text.endswith(marker[:size])), default=0)
    return text[:-keep] if keep else text


def split_reasoning(text, thinking_open=False, final=True):
    # Keep whitespace before an end token: stripping previously streamed bytes
    # would make the final response disagree with its incremental deltas.
    rest = END_PATTERN.sub("", text)
    if not final:
        rest = _without_partial_marker(rest, ENDS)
    reasoning = []
    implicit = thinking_open
    seen = False
    while True:
        candidate = rest.lstrip()
        if not final and not candidate:
            return {"content": "", "reasoning_content": "\n".join(reasoning), "reasoning_incomplete": implicit}
        if not final and candidate and OPEN.startswith(candidate) and candidate != OPEN:
            return {"content": "", "reasoning_content": "\n".join(reasoning), "reasoning_incomplete": implicit or seen}
        if candidate.startswith(OPEN):
            rest = candidate[len(OPEN):]
        elif not implicit:
            return {"content": rest.lstrip() if seen else rest,
                    "reasoning_content": "\n".join(reasoning), "reasoning_incomplete": False}
        implicit, seen = False, True
        end = rest.find(CLOSE)
        if end < 0:
            reasoning.append(rest if final else _without_partial_marker(rest, (CLOSE,)))
            return {"content": "", "reasoning_content": "\n".join(reasoning), "reasoning_incomplete": True}
        reasoning.append(rest[:end])
        rest = rest[end + len(CLOSE):].lstrip()


def assistant_message(text, thinking_open=False):
    parts = split_reasoning(text, thinking_open)
    return {"role": "assistant", "content": parts["content"],
            **({"reasoning_content": parts["reasoning_content"]} if parts["reasoning_content"] else {})}


def tool_stream_parts(parts):
    """Publish ordinary text while holding incomplete/complete tool markup.

    A final parser validates and publishes the complete structured tool call.
    Withhold trailing whitespace because that parser trims call preambles.
    Leading whitespace is conservatively buffered to preserve exact final text.
    Quoted marker examples are released by final parsing, never dispatched here.
    """
    content = parts.get('content') or ''
    if content.lstrip().startswith('{'):
        # A bare JSON tool envelope must be validated before it is published.
        # Ordinary JSON is released unchanged at completion.
        return {'content': '', 'reasoning_content': parts.get('reasoning_content') or ''}
    marker = '<tool_call>'
    start = content.find(marker)
    if start >= 0:
        content = content[:start]
    content = _without_partial_marker(content, (marker,)).rstrip()
    if content[:1].isspace():
        content = ''
    return {'content': content, 'reasoning_content': parts.get('reasoning_content') or ''}

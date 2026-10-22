"""OpenAI image parts, bounded image ingestion, and ordered VLM inputs."""
import base64
import copy
import hashlib
import http.client
import io
import ipaddress
import re
import socket
import ssl
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from . import __version__
from .deadlines import RequestDeadline, check_deadline, time_budget

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 20_000_000
VLM_PIPELINES = {"vlm", "vlm-text"}


def is_vlm(model):
    return model.get("pipeline") in VLM_PIPELINES


def text_content(content):
    if isinstance(content, str):
        return content
    return "\n".join(p.get("text", "") for p in content or [] if p.get("type") == "text")


def check_content(messages, model=None, internal=False):
    image_count = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str) or (content is None and message.get("tool_calls")):
            continue
        if not isinstance(content, list) or not content:
            raise ValueError("Message content must be text or a nonempty array of text/image_url parts")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Message parts must be objects")
            if part.get("type") == "text":
                if not isinstance(part.get("text"), str):
                    raise ValueError("Text parts require a text string")
            elif part.get("type") == "image_url":
                # Hermes vision_analyze/computer-use tools return image parts
                # on the tool row. Keep its role and call ID through capture
                # and flattening so the model template preserves the pairing.
                if message["role"] not in {"user", "tool"}:
                    raise ValueError("Image input is supported in user messages and tool results")
                if message["role"] == "tool" and (not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"].strip()):
                    raise ValueError("Image tool results require a nonempty tool_call_id")
                if model is not None and not is_vlm(model):
                    raise ValueError("This model uses the LLM pipeline. Select VLM · text + images for a vision-capable export.")
                image = part.get("image_url")
                if not isinstance(image, dict) or not isinstance(image.get("url"), str):
                    raise ValueError("image_url requires an object with a url string")
                if image.get("detail", "auto") not in {"auto", "low", "high"}:
                    raise ValueError("Image detail must be auto, low, or high")
                scheme = urlsplit(image["url"]).scheme.lower()
                if scheme not in ({"https", "http", "data", "lazyrig-input"} if internal else {"https", "http", "data"}):
                    raise ValueError("Use an image data URL or a public HTTP(S) image URL")
                image_count += 1
            else:
                raise ValueError("Supported message parts are text and image_url; audio/video input is not implemented")
    return image_count


def fetch_image(url):
    """Fetch only public addresses; pin the connection to the checked DNS result."""
    for _ in range(5):
        check_deadline()
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Image URL must be public HTTP(S) without credentials")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ValueError("Cannot resolve image URL host") from exc
        check_deadline()
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global or ipaddress.ip_address(a[4][0]).is_multicast for a in addresses):
            raise ValueError("Private/local image URLs are not fetched; upload the image as a data URL")
        connection = http.client.HTTPConnection(parsed.hostname, port, timeout=time_budget(15))
        try:
            sock = socket.create_connection(addresses[0][4][:2], timeout=time_budget(15))
            if parsed.scheme == "https":
                try:
                    sock = ssl.create_default_context().wrap_socket(sock, server_hostname=parsed.hostname)
                except Exception:
                    sock.close()
                    raise
            connection.sock = sock
            check_deadline()
            sock.settimeout(time_budget(15))
            path = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
            connection.request("GET", path, headers={"Accept": "image/*", "User-Agent": "LazyRig/" + __version__})
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("Image redirect has no destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError("Image URL returned HTTP " + str(response.status))
            length = response.getheader("Content-Length")
            if length and int(length) > MAX_IMAGE_BYTES:
                raise ValueError("Input image exceeds 10 MiB")
            data = bytearray()
            # read1 returns available bytes instead of waiting for the entire
            # image; a slow trickle must not reset the request's total budget.
            while True:
                check_deadline()
                sock.settimeout(time_budget(15))
                chunk = response.read1(min(65536, MAX_IMAGE_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_IMAGE_BYTES:
                    raise ValueError("Input image exceeds 10 MiB")
            check_deadline()
            return bytes(data)
        except RequestDeadline:
            raise
        except (OSError, http.client.HTTPException) as exc:
            check_deadline()
            raise ValueError("Could not download image: " + type(exc).__name__) from exc
        finally:
            connection.close()
    raise ValueError("Too many image URL redirects")


class ImageStore:
    def __init__(self, directory):
        self.directory = Path(directory) / "inputs"
        self.directory.mkdir(parents=True, exist_ok=True)

    def read_source(self, source):
        check_deadline()
        if isinstance(source, bytes):
            data = source
        elif isinstance(source, str) and source.startswith('data:'):
            match = re.fullmatch(r"data:image/(?:png|jpeg|jpg|webp|gif);base64,([A-Za-z0-9+/=\r\n]+)", source, re.I)
            if not match or len(match[1]) > (MAX_IMAGE_BYTES * 4 // 3 + 8):
                raise ValueError("Invalid image data URL or image exceeds 10 MiB")
            try:
                data = base64.b64decode(match[1].replace("\r", "").replace("\n", ""), validate=True)
            except ValueError as exc:
                raise ValueError("Invalid base64 image") from exc
        elif isinstance(source, str):
            data = fetch_image(source)
        else:
            raise ValueError('Invalid image source')
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError('Input image must contain 1 byte–10 MiB')
        check_deadline()
        return data

    def capture(self, messages, model):
        check_deadline()
        check_content(messages, model)
        messages = copy.deepcopy(messages)
        total = 0
        for message_index, message in enumerate(messages, 1):
            if not isinstance(message.get("content"), list):
                continue
            image_index = 0
            for part in message["content"]:
                if part["type"] != "image_url":
                    continue
                image_index += 1
                image = part["image_url"]
                try:
                    data = self.read_source(image['url'])
                    total += len(data)
                    if len(data) > MAX_IMAGE_BYTES or total > MAX_TOTAL_BYTES:
                        raise ValueError("Image input exceeds 10 MiB per image or 24 MiB total")
                    ref = self.save(data, image.get("detail", "auto"))
                except ValueError as exc:
                    raise ValueError(f"Image {image_index} in message {message_index}: {exc}") from exc
                part["image_url"] = {"url": ref, "detail": image.get("detail", "auto")}
        return messages

    def save(self, data, detail):
        from PIL import Image, ImageOps, UnidentifiedImageError
        check_deadline()
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                    raise ValueError(f"Unsupported image format: {source.format or 'unknown'}. "
                                     "Use PNG, JPEG, WebP, or a nonanimated GIF")
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("Input image exceeds 20 megapixels")
                if getattr(source, "is_animated", False):
                    raise ValueError("Animated images are not supported; submit one still frame")
                rgb = ImageOps.exif_transpose(source).convert("RGB")
                if detail == "low":
                    rgb.thumbnail((512, 512))
                output = io.BytesIO(); rgb.save(output, format="PNG")
                normalized = output.getvalue()
                check_deadline()
                digest = hashlib.sha256(normalized).hexdigest()
                target = self.directory / (digest + ".png")
                if not target.exists():
                    temp = target.with_suffix("." + uuid.uuid4().hex + ".tmp")
                    temp.write_bytes(normalized); temp.replace(target)
                return "lazyrig-input:" + digest
        except RequestDeadline:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Input is not a supported, decodable image") from exc

    def describe(self, ref):
        from PIL import Image
        if not re.fullmatch(r"lazyrig-input:[0-9a-f]{64}", ref):
            raise ValueError("Invalid stored image reference")
        digest = ref.split(":", 1)[1]
        path = self.directory / (digest + ".png")
        if not path.is_file():
            raise ValueError("Saved input image is missing; resend the original image")
        with Image.open(path) as image:
            return {"path": str(path.resolve()), "sha256": digest, "width": image.width, "height": image.height}

    def flatten(self, messages, model):
        check_content(messages, model, internal=True)
        result, images = copy.deepcopy(messages), []
        for message in result:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            text = []
            for part in content:
                if part["type"] == "text":
                    text.append(part["text"])
                else:
                    info = self.describe(part["image_url"]["url"])
                    text.append(f"<ov_genai_image_{len(images)}>")
                    images.append(info)
            message["content"] = "\n".join(text)
        return result, images

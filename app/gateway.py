"""OpenAI-compatible gateway (Timeweb AI Gateway): chat with a JSON schema, speech to text,
text to speech. Stdlib only.

No retries after a timeout: it could be billed twice. 429 is not billed: wait once and repeat
(the key is shared with other demos). Errors never carry the key or the URL.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

from app.pricing import STT_PROMPT_TOKENS, call_cost_rub, tts_cost_rub

STT_PROMPT = "Звонок в сервисную компанию: ремонт, аренда, монтаж оборудования, консультация."
TTS_INSTRUCTIONS = "Говори дружелюбно и спокойно, как оператор сервисной службы. Обычный темп."

# The page records webm/opus (Chrome, Firefox, Android) or mp4/aac (Safari, iPhone).
AUDIO_TYPES = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3",
               "audio/wav": "wav", "audio/x-m4a": "m4a"}


class GatewayError(Exception):
    """The gateway failed. ``detail`` is safe to log. Whether a failed call was billed is
    unknown (a timeout may be), so callers settle the reserved worst case."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


@dataclass
class Chat:
    content: str
    tokens_in: int
    tokens_out: int
    ms: int
    cost_rub: float


@dataclass
class Transcript:
    text: str
    audio_tokens: int
    ms: int
    cost_rub: float


@dataclass
class Speech:
    audio: bytes
    ms: int
    cost_rub: float


@dataclass
class Gateway:
    base_url: str
    api_key: str
    timeout_s: float = 20.0
    wait_429_s: float = 2.0

    def _post(self, path: str, data: bytes, content_type: str) -> tuple[int, bytes, int]:
        def once() -> tuple[int, bytes]:
            req = urllib.request.Request(self.base_url.rstrip("/") + "/" + path, data=data,
                                         headers={"Authorization": "Bearer " + self.api_key,
                                                  "Content-Type": content_type})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                    return 200, r.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()[:300]
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise GatewayError(type(e).__name__) from None

        started = time.perf_counter()
        code, body = once()
        if code == 429:
            time.sleep(self.wait_429_s)
            started = time.perf_counter()  # the wait is the gateway's queue, not the model's time
            code, body = once()
        return code, body, int((time.perf_counter() - started) * 1000)

    def chat(self, model: str, messages: list[dict], schema: dict, max_tokens: int,
             params: dict | None = None) -> Chat:
        payload = {"model": model, "messages": messages, "max_completion_tokens": max_tokens, "temperature": 0,
                   "response_format": {"type": "json_schema", "json_schema": schema}, **(params or {})}
        code, body, ms = self._post("chat/completions", json.dumps(payload).encode(), "application/json")
        if code != 200:
            raise GatewayError(f"chat HTTP {code}")
        try:
            data = json.loads(body)
            content = data["choices"][0]["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise GatewayError("chat: unexpected response") from None
        usage = data.get("usage") or {}
        tin, tout = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        return Chat(content, tin, tout, ms, call_cost_rub(model, tin, tout))

    def transcribe(self, model: str, audio: bytes, content_type: str) -> Transcript:
        ext = AUDIO_TYPES.get(content_type.split(";")[0].strip().lower(), "webm")
        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in (("model", model), ("language", "ru"), ("response_format", "json"),
                              ("prompt", STT_PROMPT))]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="turn.{ext}"\r\n'
                     f'Content-Type: {content_type}\r\n\r\n'.encode())
        parts += [audio, f"\r\n--{boundary}--\r\n".encode()]
        code, body, ms = self._post("audio/transcriptions", b"".join(parts),
                                    f"multipart/form-data; boundary={boundary}")
        if code != 200:
            raise GatewayError(f"stt HTTP {code}")
        try:
            data = json.loads(body)
            text = str(data.get("text") or "")
        except (ValueError, AttributeError):
            raise GatewayError("stt: unexpected response") from None
        usage = data.get("usage") or {}
        tin = int(usage.get("input_tokens") or STT_PROMPT_TOKENS)
        tout = int(usage.get("output_tokens") or 0)
        audio_tokens = int((usage.get("input_token_details") or {}).get("audio_tokens") or 0)
        return Transcript(text.strip(), audio_tokens, ms, call_cost_rub(model, tin, tout))

    def speak(self, model: str, voice: str, text: str) -> Speech:
        payload = {"model": model, "voice": voice, "input": text, "response_format": "mp3",
                   "instructions": TTS_INSTRUCTIONS}
        code, body, ms = self._post("audio/speech", json.dumps(payload).encode(), "application/json")
        if code != 200:
            raise GatewayError(f"tts HTTP {code}")
        return Speech(body, ms, tts_cost_rub(model, text))

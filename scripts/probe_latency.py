"""Latency probe: TTS time to first byte (streamed), STT on phone formats, LLM speed spread.

Needs AUDIO = {"webm": <b64>, "mp4": <b64>} defined before this file (phone formats of one phrase):

    python3 -c 'import base64,json,sys; print("AUDIO =", json.dumps({k: base64.b64encode(open(p,"rb").read()).decode()
        for k, p in (("webm", sys.argv[1]), ("mp4", sys.argv[2]))}))' phrase.webm phrase.mp4 > /tmp/head.py
    cat /tmp/head.py scripts/probe_latency.py | ssh vps 'docker run --rm -i --env-file /opt/demos/llm.env \
        --entrypoint python demo-rag:local -u -' > evals/probe-latency-2026-10-08.jsonl
"""
import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ["LLM_BASE_URL"].rstrip("/") + "/"
KEY = os.environ["LLM_API_KEY"]

REPLY = "Понял, кондиционер в Казани до пятницы. Какой у вас бюджет?"
LONG_REPLY = ("Понял: в офисе потёк кондиционер, Казань, нужно до пятницы. "
              "Подскажите, на какой бюджет рассчитываете и как с вами удобнее связаться?")


def emit(**row):
    print(json.dumps(row, ensure_ascii=False), flush=True)


def tts_stream(text, extra):
    payload = {"model": "openai/gpt-4o-mini-tts", "input": text, "voice": "alloy", "response_format": "mp3", **extra}
    req = urllib.request.Request(BASE + "audio/speech", data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            first = r.read1(2048) if hasattr(r, "read1") else r.read(2048)
            ttfb = int((time.perf_counter() - started) * 1000)
            chunks, total = 1, len(first)
            while True:
                part = r.read1(16384) if hasattr(r, "read1") else r.read(16384)
                if not part:
                    break
                chunks += 1
                total += len(part)
            emit(step="tts", chars=len(text), extra=extra, code=200, ttfb_ms=ttfb,
                 total_ms=int((time.perf_counter() - started) * 1000), bytes=total, chunks=chunks,
                 transfer=r.headers.get("Transfer-Encoding"), length=r.headers.get("Content-Length"),
                 ctype=r.headers.get("Content-Type"))
    except urllib.error.HTTPError as e:
        emit(step="tts", chars=len(text), extra=extra, code=e.code, error=e.read()[:300].decode(errors="replace"))


def stt(model, fmt, audio, ctype):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("model", model), ("language", "ru"), ("response_format", "json")):
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="turn.{fmt}"\r\n'
                 f'Content-Type: {ctype}\r\n\r\n'.encode())
    parts += [audio, f'\r\n--{boundary}--\r\n'.encode()]
    req = urllib.request.Request(BASE + "audio/transcriptions", data=b"".join(parts),
                                 headers={"Authorization": "Bearer " + KEY,
                                          "Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
            emit(step="stt", model=model, fmt=fmt, code=200, ms=int((time.perf_counter() - started) * 1000),
                 text=body.get("text"), usage=body.get("usage"))
    except urllib.error.HTTPError as e:
        emit(step="stt", model=model, fmt=fmt, code=e.code, error=e.read()[:300].decode(errors="replace"))


SYSTEM = ("Ты принимаешь заявку голосом для сервисной компании. Заполни карточку: что случилось, город, когда, "
          "бюджет. Что не названо — null. reply — одна фраза до 15 слов: подтверди и спроси одно недостающее поле. "
          "В reply не пиши метки вида [телефон скрыт].")
SCHEMA = {"name": "card_update", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["problem", "city", "when", "budget_rub", "reply", "done"],
    "properties": {"problem": {"type": ["string", "null"]}, "city": {"type": ["string", "null"]},
                   "when": {"type": ["string", "null"]}, "budget_rub": {"type": ["integer", "null"]},
                   "reply": {"type": "string"}, "done": {"type": "boolean"}}}}
USER = ('Карточка сейчас: {"problem": null, "city": null, "when": null, "budget_rub": null, "contact": false}\n'
        "Новая реплика клиента: <реплика>Здравствуйте. У меня потёк кондиционер в офисе, Казань, нужно до пятницы."
        "</реплика>")


def llm(model, params):
    payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
               "response_format": {"type": "json_schema", "json_schema": SCHEMA}, "max_completion_tokens": 300,
               "temperature": 0, **params}
    req = urllib.request.Request(BASE + "chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        emit(step="llm", model=model, code=e.code, error=e.read()[:300].decode(errors="replace"))
        return
    ms = int((time.perf_counter() - started) * 1000)
    content = body["choices"][0]["message"].get("content") or ""
    try:
        out = json.loads(content)
    except ValueError:
        out = {"raw": content[:300]}
    emit(step="llm", model=model, code=200, ms=ms, usage=body.get("usage"), out=out)


tts_stream(REPLY, {})
tts_stream(LONG_REPLY, {})
tts_stream(REPLY, {"stream_format": "audio"})
stt("openai/gpt-4o-transcribe", "webm", base64.b64decode(AUDIO["webm"]), "audio/webm")  # noqa: F821
stt("openai/gpt-4o-transcribe", "mp4", base64.b64decode(AUDIO["mp4"]), "audio/mp4")  # noqa: F821
stt("openai/gpt-4o-mini-transcribe", "webm", base64.b64decode(AUDIO["webm"]), "audio/webm")  # noqa: F821
for _ in range(3):
    llm("yandex/yandexgpt-lite", {})
for _ in range(2):
    llm("timeweb/gemma4:31b", {"reasoning_effort": "none"})

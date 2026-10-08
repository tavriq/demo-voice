"""FastAPI app. Run: uvicorn --factory app.main:create_app --no-proxy-headers

``--no-proxy-headers`` matters: the client address (and X-Forwarded-For trust) is resolved by
app.netutil according to TRUSTED_PROXY. Contract: docs/api.md.

Audio is read into memory, sent to recognition and dropped; it is never written to disk.
The live part of a reply is synthesized right after the model answers and lives in memory for
SPEECH_TTL_S; fixed phrases (app.dialog.PHRASES) are synthesized once and kept in memory.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app import n8n
from app.config import Settings
from app.dialog import PHRASES, ChatClient, State, missing, run_turn, triage_text
from app.gateway import AUDIO_TYPES, Gateway, GatewayError
from app.netutil import client_ip, rate_limit_key
from app.pricing import N8N_RESERVE_RUB, tts_cost_rub, turn_worst_rub
from app.store import Store

log = logging.getLogger("demo_voice")
TEXT_LIMIT = 300
SPEECH_TTL_S = 120
MAX_AUDIO_S = 20

SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, **extra}, status_code=status)


class Speeches:
    """Synthesis started right after the model answers; the page fetches the audio by id."""

    def __init__(self, workers: int = 4):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tts")
        self.items: dict[str, tuple[Future, float]] = {}
        self.lock = threading.Lock()

    def submit(self, fn, *args) -> str:
        sid = secrets.token_urlsafe(12)
        now = time.monotonic()
        with self.lock:
            for key in [k for k, (_, ts) in self.items.items() if now - ts > SPEECH_TTL_S]:
                del self.items[key]
            self.items[sid] = (self.pool.submit(fn, *args), now)
        return sid

    def get(self, sid: str) -> Future | None:
        with self.lock:
            item = self.items.get(sid)
        return item[0] if item else None


class Phrases:
    """Fixed phrases: synthesized on first request, then served from memory."""

    def __init__(self):
        self.audio: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def get(self, name: str, synthesize) -> bytes:
        with self.lock:
            if name not in self.audio:
                self.audio[name] = synthesize(PHRASES[name])
            return self.audio[name]


class Locks:
    """One turn at a time per conversation (one worker process)."""

    def __init__(self):
        self.locks: dict[str, threading.Lock] = {}
        self.guard = threading.Lock()

    def get(self, conv_id: str) -> threading.Lock:
        with self.guard:
            if len(self.locks) > 1000:
                self.locks = {k: v for k, v in self.locks.items() if v.locked()}
            return self.locks.setdefault(conv_id, threading.Lock())


def create_app(settings: Settings | None = None, client: ChatClient | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = Store(settings.var_dir / "voice.sqlite3", settings.daily_budget_rub, settings.conversations_per_hour,
                  settings.retention_days)
    if client is None and settings.live:
        client = Gateway(settings.base_url, settings.api_key, settings.timeout_s)
    speeches = Speeches()
    phrases = Phrases()
    locks = Locks()
    worst_text_turn = turn_worst_rub(settings.stt_model, settings.model, settings.tts_model, 0)
    worst_audio_turn = turn_worst_rub(settings.stt_model, settings.model, settings.tts_model,
                                      settings.max_audio_bytes)

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def limits_and_headers(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_audio_bytes:
            response = _error(413, "too_long", f"Реплика длиннее {MAX_AUDIO_S} секунд.")
        else:
            response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        return response

    def visitor_ip(request: Request) -> str:
        return client_ip(request.client.host if request.client else None, request.headers.get("x-forwarded-for"),
                         settings.trusted_proxies)

    @app.get("/api/voice/health")
    def health():
        return {"ok": True}

    @app.get("/api/voice/status")
    def status():
        spent = store.spent_today()
        return {"live": client is not None and spent + worst_audio_turn <= settings.daily_budget_rub,
                "budget_rub_today": round(spent, 2), "budget_rub_limit": settings.daily_budget_rub,
                "n8n": bool(settings.n8n_url), "models": {"stt": settings.stt_model, "llm": settings.model,
                                                          "tts": settings.tts_model},
                "counters": store.counters()}

    @app.post("/api/voice/conversations")
    def start(request: Request):
        if client is None:
            return _error(503, "budget_exhausted", "Живой режим выключен. Записанный разговор работает.")
        decision = store.start(rate_limit_key(visitor_ip(request)))
        if not decision.allowed:
            return _error(429, decision.code, decision.message, retry_after_s=decision.retry_after_s)
        conv_id = secrets.token_urlsafe(16)
        store.create(conv_id, State().to_json())
        return {"id": conv_id, "greeting": {"text": PHRASES["greeting"], "audio": "/api/voice/phrase/greeting"},
                "max_turns": settings.max_turns, "max_audio_s": MAX_AUDIO_S}

    def synthesize(text: str):
        return client.speak(settings.tts_model, settings.tts_voice, text)

    def do_turn(conv_id: str, audio: bytes | None, content_type: str, text: str | None) -> JSONResponse | dict:
        started = time.perf_counter()
        with locks.get(conv_id):
            loaded = store.load(conv_id)
            if loaded is None:
                return _error(404, "not_found", "Разговор не найден. Начните новый.")
            state = State.from_json(loaded[0])
            if state.done or state.turns >= settings.max_turns:
                return _error(409, "done", "Разговор закончен. Начните новый.")
            worst = worst_audio_turn if audio is not None else worst_text_turn
            decision = store.reserve(worst)
            if not decision.allowed:
                return _error(503, decision.code, decision.message)
            stt_ms, cost = None, 0.0
            try:
                if audio is not None:
                    heard = client.transcribe(settings.stt_model, audio, content_type)
                    stt_ms, cost, text = heard.ms, heard.cost_rub, heard.text
                result = run_turn(client, settings.model, state, text or "", settings.max_turns)
            except GatewayError as e:
                # The step may have been billed: keep the worst case. The turn is used up.
                store.settle(decision.reservation_id, worst)
                state.turns += 1
                state.done = state.turns >= settings.max_turns
                store.save(conv_id, state.to_json(), worst)
                log.warning("gateway error: %s", e.detail)
                return _error(502, "model_error", "Сервис распознавания или модель не ответили. "
                                                  "Скажите ещё раз.", turns_left=settings.max_turns - state.turns)
            except BaseException:
                store.settle(decision.reservation_id, worst)
                raise
            cost += result.cost_rub
            sentences = []
            for text, phrase_name in result.parts:
                if phrase_name:
                    sentences.append({"text": text, "audio": "/api/voice/phrase/" + phrase_name, "live": False})
                else:
                    sentences.append({"text": text, "audio": "/api/voice/speech/" + speeches.submit(synthesize, text),
                                      "live": True})
                    cost += tts_cost_rub(settings.tts_model, text)
            store.settle(decision.reservation_id, cost)
            store.save(conv_id, state.to_json(), cost)
        log.info("turn conv=%s turn=%s done=%s model_ok=%s cost=%.3f stt=%s llm=%s", conv_id[:6], state.turns,
                 result.done, result.model_ok, cost, stt_ms, result.llm_ms)
        return {"turn": state.turns, "turns_left": settings.max_turns - state.turns, "heard": result.utterance,
                "card": result.card, "missing": missing(result.card), "asks": result.asks, "reply": result.reply,
                "sentences": sentences, "done": result.done,
                "model_ok": result.model_ok, "model_error": result.model_error,
                "timings_ms": {"stt": stt_ms, "llm": result.llm_ms,
                               "server": int((time.perf_counter() - started) * 1000)},
                "cost_rub": round(cost, 4)}

    @app.post("/api/voice/conversations/{conv_id}/turn")
    async def turn(conv_id: str, request: Request):
        if client is None:
            return _error(503, "budget_exhausted", "Живой режим выключен. Записанный разговор работает.")
        ctype = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        body = await request.body()
        if ctype == "application/json":
            try:
                text = json.loads(body).get("text")
            except (ValueError, AttributeError):
                text = None
            if not isinstance(text, str) or not text.strip() or len(text) > TEXT_LIMIT:
                return _error(422, "bad_request", f"Нужен текст от 1 до {TEXT_LIMIT} символов.")
            return await run_in_threadpool(do_turn, conv_id, None, ctype, text.strip())
        if ctype not in AUDIO_TYPES:
            return _error(415, "bad_request", "Нужна запись голоса (webm, ogg, mp4) или текст.")
        if not body:
            return _error(422, "bad_request", "Пустая запись.")
        if len(body) > settings.max_audio_bytes:
            return _error(413, "too_long", f"Реплика длиннее {MAX_AUDIO_S} секунд.")
        return await run_in_threadpool(do_turn, conv_id, body, request.headers.get("content-type"), None)

    @app.get("/api/voice/speech/{sid}")
    def speech(sid: str):
        future = speeches.get(sid)
        if future is None:
            return _error(404, "not_found", "Озвучка устарела.")
        try:
            spoken = future.result(timeout=settings.timeout_s + 5)
        except GatewayError as e:
            log.warning("tts error: %s", e.detail)
            return _error(502, "model_error", "Озвучка не ответила.")
        except TimeoutError:
            return _error(504, "model_error", "Озвучка не успела.")
        return Response(spoken.audio, media_type="audio/mpeg",
                        headers={"X-Synth-Ms": str(spoken.ms), "Cache-Control": "private, max-age=120"})

    @app.get("/api/voice/phrase/{name}")
    def phrase(name: str):
        if name not in PHRASES:
            return _error(404, "not_found", "Нет такой фразы.")
        if client is None:
            return _error(503, "budget_exhausted", "Живой режим выключен.")

        def synthesize_counted(text: str) -> bytes:
            decision = store.reserve(tts_cost_rub(settings.tts_model, text))
            if not decision.allowed:
                raise GatewayError("budget")
            try:
                spoken = synthesize(text)
            except BaseException:
                store.settle(decision.reservation_id, tts_cost_rub(settings.tts_model, text))
                raise
            store.settle(decision.reservation_id, spoken.cost_rub)
            return spoken.audio

        try:
            audio = phrases.get(name, synthesize_counted)
        except GatewayError as e:
            log.warning("phrase tts error: %s", e.detail)
            return _error(502, "model_error", "Озвучка не ответила.")
        return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})

    def do_submit(conv_id: str, ip: str) -> JSONResponse | dict:
        with locks.get(conv_id):
            loaded = store.load(conv_id)
            if loaded is None:
                return _error(404, "not_found", "Разговор не найден. Начните новый.")
            state, triage = State.from_json(loaded[0]), loaded[1]
            if triage:
                return {"triage": triage, "text": triage_text(state)}
            if not state.card.get("problem"):
                return _error(409, "not_ready", "Сначала расскажите, что случилось.")
            decision = store.reserve(N8N_RESERVE_RUB)
            if not decision.allowed:
                return _error(503, decision.code, decision.message)
            text = triage_text(state)
            try:
                result = n8n.send(settings.n8n_url, text, ip)
            except n8n.TriageError as e:
                store.settle(decision.reservation_id, e.cost_rub)
                return _error(429 if e.code == "n8n_rate_limited" else 502, e.code, e.message)
            except BaseException:
                store.settle(decision.reservation_id, N8N_RESERVE_RUB)
                raise
            store.settle(decision.reservation_id, result.cost_rub)
            store.save(conv_id, state.to_json(), result.cost_rub, result.to_json())
        log.info("submit conv=%s category=%s urgency=%s cost=%.3f", conv_id[:6], result.category, result.urgency,
                 result.cost_rub)
        return {"triage": result.to_json(), "text": text}

    @app.post("/api/voice/conversations/{conv_id}/submit")
    async def submit(conv_id: str, request: Request):
        if not settings.n8n_url:
            return _error(503, "n8n_off", "Разбор n8n не подключён.")
        return await run_in_threadpool(do_submit, conv_id, visitor_ip(request))

    return app


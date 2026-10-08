"""Prices in rubles per 1M tokens and the worst case of one turn.

Prices: Timeweb AI Gateway cabinet, «Модели». Text models 07.10.2026 18:45 MSK and
08.10.2026 12:05 MSK (flash-lite, 4.1 mini, Haiku), speech models 08.10.2026 11:13 MSK.
The page does not say whether VAT is included. The gateway does not return prices in the
API, so they live here and must be updated by hand.

Speech recognition returns usage (10 audio tokens per second of speech, probe 08.10).
Speech synthesis returns only audio, no usage: its cost is an estimate per character,
see TTS_TOKENS_PER_CHAR.
"""

from __future__ import annotations

PRICES_RUB_PER_1M: dict[str, tuple[float, float]] = {
    "yandex/yandexgpt-lite": (210, 210),
    "timeweb/gpt-oss-120b": (20, 82),
    "timeweb/gemma4:31b": (189, 540),
    "gemini/gemini-3.1-flash-lite": (34, 203),
    "openai/gpt-4.1-mini": (54, 216),
    "anthropic/claude-haiku-4-5": (135, 1080),
    "anthropic/claude-haiku-5-5": (14, 68),
    "openai/gpt-5.6-terra": (270, 1620),
    "openai/gpt-4o-transcribe": (810, 1350),
    "openai/gpt-4o-mini-transcribe": (405, 675),
    "openai/gpt-4o-mini-tts": (81, 1620),
}
# Unknown model: assume the most expensive known price, so budgets stay on the safe side.
FALLBACK = max(PRICES_RUB_PER_1M.values(), key=lambda p: p[0] + p[1])

# Recognition: 10 audio tokens per second (probe 08.10); text out ~4 per second, 8 assumed.
STT_AUDIO_TOKENS_PER_S = 10
STT_TEXT_TOKENS_PER_S = 8
STT_PROMPT_TOKENS = 60
# The lowest bitrate the page records with: the worst duration of an upload of N bytes.
MIN_AUDIO_BITRATE = 32_000
# Synthesis: ~14 characters per second of speech (probe 08.10) and, as OpenAI estimates for
# this model, ~21 audio tokens per second -> 1.5 tokens per character. Not measured: the
# gateway returns no usage for speech. Check against the key's daily spend in the cabinet.
TTS_TOKENS_PER_CHAR = 1.5
# One reply to the model: the conversation so far, the card and the instructions.
WORST_LLM_IN = 2_000
LLM_MAX_TOKENS = 300
MAX_REPLY_CHARS = 220
# n8n triage on its own model: ~690 tokens per request on terra, up to two attempts.
N8N_RESERVE_RUB = 0.6


def call_cost_rub(model: str, tokens_in: int, tokens_out: int) -> float:
    pin, pout = PRICES_RUB_PER_1M.get(model, FALLBACK)
    return (tokens_in * pin + tokens_out * pout) / 1_000_000


def stt_worst_rub(model: str, audio_bytes: int) -> float:
    seconds = audio_bytes * 8 / MIN_AUDIO_BITRATE
    return call_cost_rub(model, STT_PROMPT_TOKENS + int(seconds * STT_AUDIO_TOKENS_PER_S),
                         int(seconds * STT_TEXT_TOKENS_PER_S))


def tts_cost_rub(model: str, text: str) -> float:
    return call_cost_rub(model, len(text), int(len(text) * TTS_TOKENS_PER_CHAR))


def turn_worst_rub(stt_model: str, llm_model: str, tts_model: str, audio_bytes: int) -> float:
    """Recognition of the upload, one model call, synthesis of the longest reply."""
    return (stt_worst_rub(stt_model, audio_bytes) + call_cost_rub(llm_model, WORST_LLM_IN, LLM_MAX_TOKENS)
            + tts_cost_rub(tts_model, "x" * MAX_REPLY_CHARS))

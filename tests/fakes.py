"""Scripted gateway: chat replies, transcripts and speech in order; records requests."""

from __future__ import annotations

import json

from app.gateway import Chat, GatewayError, Speech, Transcript


def update(problem=None, city=None, when=None, budget_rub=None, ack="", off_topic=False):
    return json.dumps({"problem": problem, "city": city, "when": when, "budget_rub": budget_rub, "ack": ack,
                       "off_topic": off_topic}, ensure_ascii=False)


class FakeGateway:
    def __init__(self, replies=(), transcripts=()):
        """replies: JSON strings or exceptions for chat; transcripts: strings or exceptions for STT."""
        self.replies = list(replies)
        self.transcripts = list(transcripts)
        self.chats: list[dict] = []
        self.heard: list[tuple[bytes, str]] = []
        self.spoken: list[str] = []

    def chat(self, model, messages, schema, max_tokens, params=None):
        self.chats.append({"model": model, "messages": messages, "params": params})
        if not self.replies:
            raise AssertionError("no more scripted replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Chat(reply, 400, 60, 15, 0.1)

    def transcribe(self, model, audio, content_type):
        self.heard.append((audio, content_type))
        text = self.transcripts.pop(0)
        if isinstance(text, Exception):
            raise text
        return Transcript(text, 57, 20, 0.09)

    def speak(self, model, voice, text):
        self.spoken.append(text)
        return Speech(b"ID3fake", 25, 0.2)


__all__ = ["FakeGateway", "GatewayError", "update"]

"""Hand the finished card to the n8n triage (demo-n8n, POST /webhook/triage {"text"}).

The voice container sits in the n8n docker network and calls it directly. n8n trusts
X-Real-IP (TRUST_PROXY_HEADER=x-real-ip), so its own per-address limit (5 an hour) counts
the visitor, not this container.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from app.pricing import N8N_RESERVE_RUB, call_cost_rub

CATEGORY = {"repair": "ремонт", "rental": "аренда", "installation": "монтаж", "consultation": "консультация",
            "complaint": "жалоба", "spam": "спам", "other": "другое"}
URGENCY = {"low": "низкая", "normal": "обычная", "high": "высокая"}


class TriageError(Exception):
    def __init__(self, code: str, message: str, cost_rub: float):
        super().__init__(code)
        self.code = code
        self.message = message
        self.cost_rub = cost_rub


@dataclass
class Triage:
    category: str
    urgency: str
    needs_human: bool
    summary: str
    next_step: str
    route_reasons: list[str]
    model: str | None
    ms: int
    n8n_ms: int | None
    cost_rub: float

    def to_json(self) -> dict:
        return {"category": self.category, "category_label": CATEGORY.get(self.category, self.category),
                "urgency": self.urgency, "urgency_label": URGENCY.get(self.urgency, self.urgency),
                "needs_human": self.needs_human, "summary": self.summary, "next_step": self.next_step,
                "route_reasons": self.route_reasons, "model": self.model, "ms": self.ms, "n8n_ms": self.n8n_ms}


def send(url: str, text: str, client_ip: str, timeout_s: float = 70.0) -> Triage:
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json", "X-Real-IP": client_ip})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # 400/429 are refused before the model: not billed.
        if e.code == 429:
            raise TriageError("n8n_rate_limited", "Разбор n8n принимает 5 заявок в час с одного адреса. "
                              "Карточка готова, разбор — позже.", 0.0) from None
        raise TriageError("n8n_error", "Разбор n8n не ответил.", 0.0 if e.code < 500 else N8N_RESERVE_RUB) \
            from None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        raise TriageError("n8n_error", "Разбор n8n не ответил.", N8N_RESERVE_RUB) from None
    ms = int((time.perf_counter() - started) * 1000)
    try:
        result, meta = body["result"], body.get("meta") or {}
        model = meta.get("model")
        cost = call_cost_rub(model, int(meta.get("tokens_in") or 0), int(meta.get("tokens_out") or 0)) \
            if model and body.get("mode") != "mock" else 0.0
        trace = meta.get("trace") or {}
        return Triage(str(result["category"]), str(result["urgency"]), bool(result["needs_human"]),
                      str(result.get("summary") or ""), str(result.get("next_step") or ""),
                      list((trace.get("route") or {}).get("reasons") or []), model, ms,
                      (trace.get("timings_ms") or {}).get("total"), cost)
    except (KeyError, TypeError):
        raise TriageError("n8n_error", "Разбор n8n ответил не по форме.", N8N_RESERVE_RUB) from None

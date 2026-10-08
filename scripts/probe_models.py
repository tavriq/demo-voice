"""Which model fills the card: the service's own turn logic (app.dialog) on a fixed script, per model.

Runs inside the demo-voice image (it has app/) with the gateway env; prints one JSON line per turn:

    ssh vps 'docker run --rm -i --env-file /opt/demos/llm.env --entrypoint python demo-voice:local -u -' \
        < scripts/probe_models.py > evals/probe-models-2026-10-08.jsonl

Since the code asks the questions (08.10), the checks are the card and the spoken acknowledgement:
t1 problem/city/when, budget empty; t2 budget; t3 done with the contact; off-topic flagged.
Two scripts with different words, so the gateway's cache of identical requests does not skew latency.
"""
import json
import os

from app.dialog import OFF_TOPIC, State, run_turn
from app.gateway import Gateway, GatewayError

MODELS = [
    "yandex/yandexgpt-lite",
    "gemini/gemini-3.1-flash-lite",
    "anthropic/claude-haiku-5-5",
    "openai/gpt-4.1-mini",
]
SCRIPTS = {
    "kazan": [
        ("t1", "Здравствуйте. У меня потек кондиционер в офисе, Казань, нужно до пятницы."),
        ("t2", "Бюджет тысяч пятнадцать."),
        ("t3", "Телефон 8 912 345 67 89."),
    ],
    "samara": [
        ("t1", "Добрый день, в Самаре сломался котёл, нужен мастер завтра утром."),
        ("t2", "Бюджет до двадцати тысяч."),
        ("t3", "Почта ivan собака mail точка ru."),
    ],
    "off": [("off", "А сколько у вас стоит аренда генератора на выходные?")],
}
EXPECT = {("kazan", "t1"): ("Казань", "кондиц"), ("samara", "t1"): ("Самара", "кот")}
BUDGET = {"kazan": 15000, "samara": 20000}

gw = Gateway(os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"], timeout_s=30)


def check(script, step, r):
    c = r.card
    ack_ok = r.said.startswith("Принято") and "?" not in r.said
    if step == "t1":
        city, word = EXPECT[(script, step)]
        return c["city"] == city and word in (c["problem"] or "").lower() and bool(c["when"]) \
            and c["budget_rub"] is None and ack_ok
    if step == "t2":
        return c["budget_rub"] == BUDGET[script] and ack_ok
    if step == "t3":
        return r.done and c["contact"] is not None
    return c["city"] is None and "генератор" in (c["problem"] or "") and OFF_TOPIC in r.said


def run(model, script):
    state = State()
    for step, text in SCRIPTS[script]:
        try:
            r = run_turn(gw, model, state, text, 6)
        except GatewayError as e:
            print(json.dumps({"model": model, "script": script, "step": step, "error": e.detail},
                             ensure_ascii=False), flush=True)
            return
        print(json.dumps({"model": model, "script": script, "step": step, "ok": check(script, step, r),
                          "ms": r.llm_ms, "cost_rub": round(r.cost_rub, 4), "model_ok": r.model_ok,
                          "card": r.card, "said": r.said, "asks": r.asks, "done": r.done}, ensure_ascii=False),
              flush=True)


for m in MODELS:
    for name in SCRIPTS:
        run(m, name)

import sqlite3
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import n8n
from app.config import Settings
from app.main import create_app
from tests.fakes import FakeGateway, GatewayError, update

AUDIO = b"\x1aE\xdf\xa3fake-webm"


@pytest.fixture
def settings(tmp_path):
    return replace(Settings.from_env({}), api_key="k", base_url="http://gw.example", var_dir=tmp_path / "var",
                   n8n_url="http://n8n:5678/webhook/triage")


def make(settings, replies=(), transcripts=(), **overrides):
    gw = FakeGateway(replies, transcripts)
    return TestClient(create_app(replace(settings, **overrides), client=gw)), gw


def start(client):
    r = client.post("/api/voice/conversations")
    assert r.status_code == 200, r.text
    return r.json()["id"]


def say(client, conv, audio=AUDIO, ctype="audio/webm;codecs=opus"):
    return client.post(f"/api/voice/conversations/{conv}/turn", content=audio, headers={"Content-Type": ctype})


def test_audio_turn_end_to_end(settings):
    client, gw = make(settings, [update("потёк кондиционер", "Казань", "до пятницы", None,
                                        "Принято: кондиционер, Казань, до пятницы.")],
                      ["У меня потёк кондиционер, Казань, до пятницы. Телефон 8 912 345 67 89"])
    conv = start(client)
    r = say(client, conv)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["card"]["city"] == "Казань" and data["card"]["contact"] == "[телефон скрыт]"
    assert "345 67 89" not in data["heard"] and data["missing"] == ["budget"]
    assert data["timings_ms"]["stt"] == 20 and data["timings_ms"]["llm"] == 15 and data["turns_left"] == 5
    assert gw.heard == [(AUDIO, "audio/webm;codecs=opus")]
    ack, live, question = data["sentences"]
    assert ack == {"text": "Принято.", "audio": "/api/voice/phrase/ack", "live": False} and live["live"]
    audio = client.get(live["audio"])
    assert audio.status_code == 200 and audio.content == b"ID3fake" and audio.headers["x-synth-ms"] == "25"
    assert question == {"text": "Какой бюджет закладываете?", "audio": "/api/voice/phrase/budget", "live": False}
    assert client.get(question["audio"]).content == b"ID3fake"
    client.get(question["audio"])  # the second time from memory
    assert gw.spoken == ["Кондиционер, Казань, до пятницы.", "Какой бюджет закладываете?"]
    assert r.headers["cache-control"] == "no-store"


def test_audio_and_contacts_never_stored(settings):
    client, _ = make(settings, [update("сломался насос")], ["Сломался насос, пишите boss@corp.example"])
    conv = start(client)
    say(client, conv)
    db = sqlite3.connect(settings.var_dir / "voice.sqlite3")
    dump = "\n".join(str(row) for row in db.execute("SELECT * FROM conversations"))
    assert "boss@corp.example" not in dump and "[email скрыт]" in dump and "fake-webm" not in dump


def test_text_turn_and_submit(settings, monkeypatch):
    client, gw = make(settings, [update("течёт кран", "Казань", "завтра", 5000, "Принято: кран, Казань."),
                                 update("течёт кран", "Казань", "завтра", 5000),
                                 update("течёт кран", "Казань", "завтра", 5000)])
    sent = {}

    def fake_send(url, text, ip, timeout_s=70.0):
        sent.update(url=url, text=text, ip=ip)
        return n8n.Triage("repair", "normal", False, "течёт кран", "позвонить", [], "openai/gpt-5.6-terra",
                          900, 850, 0.3)

    monkeypatch.setattr(n8n, "send", fake_send)
    conv = start(client)
    r1 = client.post(f"/api/voice/conversations/{conv}/turn", json={"text": "Течёт кран, Казань, завтра, 5 тысяч"})
    assert r1.json()["done"] is False and r1.json()["asks"] == "contact"
    r2 = client.post(f"/api/voice/conversations/{conv}/turn", json={"text": "По этому телефону"})
    assert r2.json()["asks"] == "contact_again"
    r2 = client.post(f"/api/voice/conversations/{conv}/turn", json={"text": "Не надо связываться"})
    assert r2.json()["done"] is True and r2.json()["asks"] == "closing" and gw.heard == []
    assert r2.json()["sentences"][-1]["audio"] == "/api/voice/phrase/closing"
    s = client.post(f"/api/voice/conversations/{conv}/submit")
    assert s.status_code == 200, s.text
    assert s.json()["triage"]["category_label"] == "ремонт" and "Город: Казань" in sent["text"]
    assert sent["ip"] == "testclient"
    # idempotent: the second submit does not call n8n again
    monkeypatch.setattr(n8n, "send", lambda *a, **k: pytest.fail("called twice"))
    assert client.post(f"/api/voice/conversations/{conv}/submit").json()["triage"]["category"] == "repair"
    assert client.post(f"/api/voice/conversations/{conv}/turn", json={"text": "ещё"}).status_code == 409


@pytest.mark.parametrize("transcript", [("", 40), ("Звонок в сервисную компанию", 3)])
def test_recording_without_speech_is_not_a_turn(settings, transcript):
    client, gw = make(settings, [], [transcript])
    conv = start(client)
    r = say(client, conv)
    assert r.status_code == 200 and r.json()["empty"] is True and r.json()["turns_left"] == 6
    assert r.json()["sentences"][0]["audio"] == "/api/voice/phrase/not_heard" and gw.chats == []


def test_too_many_empty_recordings_end_the_conversation(settings):
    client, _ = make(settings, [], [("", 0)] * 7)
    conv = start(client)
    for _ in range(6):
        assert say(client, conv).json()["done"] is False
    assert say(client, conv).json()["done"] is True
    assert say(client, conv).status_code == 409


def test_submit_needs_problem(settings):
    client, _ = make(settings)
    conv = start(client)
    assert client.post(f"/api/voice/conversations/{conv}/submit").json()["code"] == "not_ready"


def test_conversations_per_hour(settings):
    client, _ = make(settings, conversations_per_hour=2)
    start(client), start(client)
    r = client.post("/api/voice/conversations")
    assert r.status_code == 429 and r.json()["code"] == "rate_limited"


def test_budget_exhausted_blocks_before_any_call(settings):
    client, gw = make(settings, [update()], ["текст"], daily_budget_rub=0.5)
    conv = start(client)
    r = say(client, conv)
    assert r.status_code == 503 and r.json()["code"] == "budget_exhausted" and gw.heard == []


def test_gateway_error_uses_up_the_turn(settings):
    client, _ = make(settings, [], [GatewayError("stt HTTP 500")], max_turns=1)
    conv = start(client)
    r = say(client, conv)
    assert r.status_code == 502 and r.json()["turns_left"] == 0
    assert say(client, conv).status_code == 409


@pytest.mark.parametrize("body, ctype, code", [
    (b"", "audio/webm", 422), (b"x", "video/mp4", 415), (b"x" * 500_000, "audio/webm", 413),
    (b'{"text": ""}', "application/json", 422), (b'{"text": 5}', "application/json", 422),
])
def test_bad_turns(settings, body, ctype, code):
    client, gw = make(settings)
    conv = start(client)
    r = client.post(f"/api/voice/conversations/{conv}/turn", content=body, headers={"Content-Type": ctype})
    assert r.status_code == code and gw.heard == [] and gw.chats == []


def test_unknown_phrase(settings):
    client, gw = make(settings)
    assert client.get("/api/voice/phrase/../../etc").status_code == 404
    assert client.get("/api/voice/phrase/hello").status_code == 404 and gw.spoken == []


def test_unknown_conversation(settings):
    client, _ = make(settings, [], ["x"])
    assert say(client, "nope").status_code == 404


def test_offline_without_key(settings):
    client = TestClient(create_app(replace(settings, api_key="")))
    assert client.post("/api/voice/conversations").status_code == 503
    assert client.get("/api/voice/status").json()["live"] is False

from app.pricing import call_cost_rub, stt_worst_rub, tts_cost_rub, turn_worst_rub


def test_probe_prices():
    # probe 08.10: 75 tokens in, 22 out for a 5.7 s phrase -> ~0.09 ₽
    assert round(call_cost_rub("openai/gpt-4o-transcribe", 75, 22), 3) == 0.090
    assert round(call_cost_rub("yandex/yandexgpt-lite", 225, 48), 3) == 0.057


def test_worst_turn_bounds():
    # 400 KB at the lowest bitrate is ~100 s of speech: the reservation must cover it
    assert 1.5 < stt_worst_rub("openai/gpt-4o-transcribe", 400 * 1024) < 2.5
    assert 2.0 < turn_worst_rub("openai/gpt-4o-transcribe", "yandex/yandexgpt-lite", "openai/gpt-4o-mini-tts",
                                400 * 1024) < 3.5
    assert tts_cost_rub("openai/gpt-4o-mini-tts", "x" * 60) < 0.2


def test_unknown_model_priced_as_the_most_expensive():
    assert call_cost_rub("someone/new-model", 1000, 1000) >= call_cost_rub("openai/gpt-5.6-terra", 1000, 1000)


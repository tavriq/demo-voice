from app.gateway import is_noise


def test_noise_filter_keeps_real_speech():
    for text in ("Сочи, завтра", "8 912 345 67 89", "Через сто п'ять днів.", "Клиент сломал насос"):
        assert not is_noise(text), text
    for text in ("Hej.", "ありがとう。", "Клиент по-русски описывает заявку.", "Субтитры сделал DimaTorzok"):
        assert is_noise(text), text

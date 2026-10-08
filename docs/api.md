# API голосового приёма (`/api/voice/`)

| Метод и путь | Что делает |
|---|---|
| `GET health` | жив ли сервис |
| `GET status` | живой режим, расход за день и лимит, модели, разговоров за сутки |
| `POST conversations` | новый разговор (5 в час с адреса) → `id`, приветствие (`greeting.audio`) |
| `POST conversations/{id}/turn` | реплика: тело — запись голоса (`audio/webm`, `audio/ogg`, `audio/mp4`, до 400 КБ) или `{"text"}` до 300 знаков |
| `GET speech/{sid}` | живая часть ответа (mp3), хранится в памяти 2 минуты; `X-Synth-Ms` — время озвучки |
| `GET phrase/{name}` | готовая фраза (mp3): озвучена один раз и лежит в памяти |
| `POST conversations/{id}/submit` | карточка уходит в разбор n8n (один раз; повтор отдаёт тот же результат) |

Ответ реплики:

```json
{
  "turn": 1, "turns_left": 5,
  "heard": "Здравствуйте. У меня потек кондиционер в офисе, Казань, нужно до пятницы.",
  "card": {"problem": "…", "city": "Казань", "when": "до пятницы", "budget_rub": null, "contact": null},
  "missing": ["budget", "contact"], "asks": "budget",
  "reply": "Принято. Ремонт кондиционера в Казани до пятницы. Какой бюджет закладываете?",
  "sentences": [
    {"text": "Принято.", "audio": "/api/voice/phrase/ack", "live": false},
    {"text": "Ремонт кондиционера в Казани до пятницы.", "audio": "/api/voice/speech/…", "live": true},
    {"text": "Какой бюджет закладываете?", "audio": "/api/voice/phrase/budget", "live": false}
  ],
  "done": false, "model_ok": true, "model_error": null,
  "timings_ms": {"stt": 968, "llm": 1656, "server": 2655}, "cost_rub": 0.22
}
```

`heard` и `card` — с контактами, скрытыми метками (`[телефон скрыт]`). Ошибки: `{"code", "message"}` —
`rate_limited` (429), `budget_exhausted` (503), `model_error` (502, реплика засчитана), `done` (409),
`too_long` (413), `bad_request` (415/422), `not_found` (404), `n8n_rate_limited` (429).

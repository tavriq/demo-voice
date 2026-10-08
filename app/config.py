"""Settings from the environment. The key is never logged or returned."""

from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import IPv4Network, IPv6Network, ip_network
from pathlib import Path


def parse_trusted_proxies(raw: str) -> tuple[IPv4Network | IPv6Network, ...]:
    """TRUSTED_PROXY: comma-separated IPs or CIDR ranges. Empty = trust nobody."""
    return tuple(ip_network(p.strip(), strict=False) for p in raw.split(",") if p.strip())


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    stt_model: str
    tts_model: str
    tts_voice: str
    n8n_url: str
    daily_budget_rub: float
    conversations_per_hour: int
    max_turns: int
    max_audio_bytes: int
    retention_days: int
    var_dir: Path
    trusted_proxies: tuple
    timeout_s: float

    @property
    def live(self) -> bool:
        return bool(self.api_key and self.base_url)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = os.environ if env is None else env
        root = Path(__file__).resolve().parent.parent
        return cls(
            api_key=env.get("LLM_API_KEY", "").strip(),
            base_url=env.get("LLM_BASE_URL", "").strip(),
            model=env.get("VOICE_MODEL", "gemini/gemini-3.1-flash-lite").strip(),
            stt_model=env.get("STT_MODEL", "openai/gpt-4o-transcribe").strip(),
            tts_model=env.get("TTS_MODEL", "openai/gpt-4o-mini-tts").strip(),
            tts_voice=env.get("TTS_VOICE", "alloy").strip(),
            n8n_url=env.get("N8N_TRIAGE_URL", "").strip(),
            daily_budget_rub=float(env.get("DAILY_BUDGET_RUB", "15")),
            conversations_per_hour=int(env.get("CONVERSATIONS_PER_HOUR", "5")),
            max_turns=int(env.get("MAX_TURNS", "6")),
            max_audio_bytes=int(env.get("MAX_AUDIO_BYTES", str(400 * 1024))),
            retention_days=int(env.get("RETENTION_DAYS", "30")),
            var_dir=Path(env.get("VAR_DIR", str(root / "var"))),
            trusted_proxies=parse_trusted_proxies(env.get("TRUSTED_PROXY", "")),
            timeout_s=float(env.get("LLM_TIMEOUT_S", "20")),
        )

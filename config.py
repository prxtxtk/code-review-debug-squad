"""Application configuration and the provider-agnostic LLM factory.

The app supports two real LLM transports (hosted APIs and local Ollama) plus a
clearly labelled offline demo transport so the complete UI can be evaluated
without credentials.  All three transports implement LangChain's
``BaseChatModel`` interface and therefore work in exactly the same LCEL chains.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

PROVIDER_LABELS: dict[str, str] = {
    "google": "Google Gemini (hosted API)",
    "openai": "OpenAI-compatible (hosted API)",
    "ollama": "Ollama (local model)",
    "demo": "Offline Demo (deterministic, no API key)",
}
PROVIDER_IDS = {label: key for key, label in PROVIDER_LABELS.items()}

DEFAULT_MODELS: dict[str, str] = {
    "google": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5-coder:7b",
    "demo": "code-review-squad-demo",
}

MODEL_PRESETS: dict[str, list[str]] = {
    "google": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "ollama": ["qwen2.5-coder:7b", "llama3.1:8b", "deepseek-coder-v2:16b", "mistral:7b"],
    "demo": ["code-review-squad-demo"],
}

_PROVIDER_ALIASES = {
    "gemini": "google",
    "google-genai": "google",
    "google_genai": "google",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "local": "ollama",
    "offline": "demo",
    "mock": "demo",
}


class ConfigurationError(RuntimeError):
    """Raised when the selected provider is not usable."""


@dataclass(slots=True)
class AppConfig:
    """Resolved runtime configuration (environment + sidebar overrides)."""

    provider: str = "demo"
    model: str = DEFAULT_MODELS["demo"]
    api_key: str = ""
    base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    top_k: int = 4
    chroma_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "chroma_db")
    knowledge_base_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "knowledge_base")
    collection_name: str = "coding_guidelines"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sandbox_timeout: int = 5
    max_test_retries: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_demo(self) -> bool:
        return self.provider == "demo"

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)

    @property
    def uses_api_key(self) -> bool:
        return self.provider in {"google", "openai"}

    def public_dict(self) -> dict[str, Any]:
        """Serializable configuration safe to render in the trace inspector."""
        return {
            "provider": self.provider,
            "provider_label": self.provider_label,
            "model": self.model,
            "base_url": self.base_url if self.provider == "openai" else "",
            "ollama_base_url": self.ollama_base_url if self.provider == "ollama" else "",
            "api_key": "***configured***" if self.api_key else "",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "collection": self.collection_name,
            "sandbox_timeout_seconds": self.sandbox_timeout,
        }


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def normalize_provider(value: str) -> str:
    provider = (value or "demo").strip().lower()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    return provider if provider in PROVIDER_LABELS else "demo"


def provider_api_key(provider: str, explicit: str | None = None) -> str:
    if explicit:
        return explicit.strip()
    if provider == "google":
        return _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY") or _env("LLM_API_KEY")
    if provider == "openai":
        return _env("OPENAI_API_KEY") or _env("LLM_API_KEY")
    return ""


def provider_base_url(provider: str, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit.strip()
    if provider == "openai":
        return _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return ""


def default_model(provider: str) -> str:
    env_provider = normalize_provider(_env("LLM_PROVIDER", "demo"))
    env_model = _env("LLM_MODEL")
    if provider == env_provider and env_model:
        return env_model
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["demo"])


def config_from_env() -> AppConfig:
    provider = normalize_provider(_env("LLM_PROVIDER", "demo"))
    top_k_raw = _env("RAG_TOP_K", "4")
    timeout_raw = _env("SANDBOX_TIMEOUT_SECONDS", "5")
    try:
        top_k = max(1, min(int(top_k_raw), 10))
    except ValueError:
        top_k = 4
    try:
        sandbox_timeout = max(1, min(int(timeout_raw), 30))
    except ValueError:
        sandbox_timeout = 5
    chroma_dir = Path(_env("CHROMA_DB_DIR", str(PROJECT_ROOT / "chroma_db")))
    if not chroma_dir.is_absolute():
        chroma_dir = (PROJECT_ROOT / chroma_dir).resolve()
    kb_dir = Path(_env("KNOWLEDGE_BASE_DIR", str(PROJECT_ROOT / "knowledge_base")))
    if not kb_dir.is_absolute():
        kb_dir = (PROJECT_ROOT / kb_dir).resolve()
    return AppConfig(
        provider=provider,
        model=default_model(provider),
        api_key=provider_api_key(provider),
        base_url=provider_base_url(provider),
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        temperature=float(_env("LLM_TEMPERATURE", "0.1") or 0.1),
        top_k=top_k,
        chroma_dir=chroma_dir,
        knowledge_base_dir=kb_dir,
        collection_name=_env("CHROMA_COLLECTION", "coding_guidelines"),
        embedding_model=_env(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        sandbox_timeout=sandbox_timeout,
        max_test_retries=max(0, min(int(_env("MAX_TEST_RETRIES", "1") or 1), 2)),
    )


def build_llm(config: AppConfig):
    """Create the selected LangChain chat model.

    Provider SDKs are imported lazily so the app can start in offline demo mode
    even when an optional integration package is not installed.
    """
    if config.provider == "demo":
        from demo_llm import DemoChatModel

        return DemoChatModel()

    if not config.model:
        raise ConfigurationError("A model name is required.")
    if config.provider in {"google", "openai"} and not config.api_key:
        env_name = "GOOGLE_API_KEY / GEMINI_API_KEY" if config.provider == "google" else "OPENAI_API_KEY"
        raise ConfigurationError(
            f"No API key configured for {config.provider_label}. "
            f"Add {env_name} to .env or paste a key in the sidebar."
        )

    if config.provider == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ConfigurationError(
                "langchain-google-genai is not installed. Run: pip install -r requirements.txt"
            ) from exc
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_retries=2,
        )

    if config.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ConfigurationError(
                "langchain-openai is not installed. Run: pip install -r requirements.txt"
            ) from exc
        kwargs: dict[str, Any] = {
            "model": config.model,
            "api_key": config.api_key,
            "temperature": config.temperature,
            "max_retries": 2,
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url.rstrip("/")
        return ChatOpenAI(**kwargs)

    if config.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ConfigurationError(
                "langchain-ollama is not installed. Run: pip install -r requirements.txt"
            ) from exc
        return ChatOllama(
            model=config.model,
            base_url=config.ollama_base_url.rstrip("/"),
            temperature=config.temperature,
            validate_model_on_init=False,
        )

    raise ConfigurationError(f"Unknown LLM provider: {config.provider}")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", item)))
        return "\n".join(parts)
    return str(content)


def llm_status(config: AppConfig, probe: bool = False) -> dict[str, Any]:
    """Return a truthful, side-effect-light status for the sidebar indicator."""
    status: dict[str, Any] = {
        "ready": False,
        "level": "error",
        "label": "Not configured",
        "detail": "",
        "probe": probe,
    }
    if config.provider == "demo":
        status.update(
            ready=True,
            level="warning",
            label="Offline demo ready",
            detail="Deterministic local model. No network or key required.",
        )
        return status

    if config.provider in {"google", "openai"}:
        if config.api_key:
            status.update(
                ready=True,
                level="success",
                label="API configured",
                detail=f"{config.provider_label} / {config.model}",
            )
        else:
            status.update(
                level="error",
                label="API key missing",
                detail="Paste a key in the sidebar or add it to .env.",
            )
        if probe and status["ready"]:
            probe_result = test_llm_connection(config)
            status["probe_result"] = probe_result
            if not probe_result["ok"]:
                status.update(
                    ready=False,
                    level="error",
                    label="Connection failed",
                    detail=probe_result["message"],
                )
            else:
                status.update(
                    label="API connected",
                    detail=f"{probe_result['latency_ms']} ms round-trip",
                )
        return status

    if config.provider == "ollama":
        base = config.ollama_base_url.rstrip("/")
        if not base:
            status.update(level="error", label="Ollama URL missing", detail="Set OLLAMA_BASE_URL.")
            return status
        status.update(
            ready=True,
            level="warning",
            label="Ollama configured",
            detail=f"{base} / {config.model}",
        )
        if probe:
            import json
            import urllib.error
            import urllib.request

            try:
                with urllib.request.urlopen(f"{base}/api/tags", timeout=4) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                installed = [m.get("name", "") for m in payload.get("models", [])]
                if any(name == config.model or name.split(":")[0] == config.model.split(":")[0] for name in installed):
                    status.update(
                        ready=True,
                        level="success",
                        label="Ollama connected",
                        detail=f"Model available: {config.model}",
                    )
                else:
                    status.update(
                        ready=False,
                        level="warning",
                        label="Ollama connected, model missing",
                        detail=(
                            f"Installed: {', '.join(installed) or 'none'}. "
                            f"Run: ollama pull {config.model}"
                        ),
                    )
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                status.update(
                    ready=False,
                    level="error",
                    label="Ollama unreachable",
                    detail=f"{base}: {exc}",
                )
        return status

    return status


def test_llm_connection(config: AppConfig) -> dict[str, Any]:
    """Perform a tiny real invocation and measure latency."""
    started = time.perf_counter()
    try:
        llm = build_llm(config)
        response = llm.invoke("Reply with exactly the single word OK and nothing else.")
        elapsed = int((time.perf_counter() - started) * 1000)
        text = _message_text(response).strip()
        return {
            "ok": True,
            "latency_ms": elapsed,
            "message": text[:200] or "Empty response",
        }
    except Exception as exc:  # surfaced in the UI as an actionable error
        elapsed = int((time.perf_counter() - started) * 1000)
        return {
            "ok": False,
            "latency_ms": elapsed,
            "message": f"{type(exc).__name__}: {exc}",
        }


def llm_identity(config: AppConfig) -> dict[str, Any]:
    return {
        "provider": config.provider,
        "provider_label": config.provider_label,
        "model": config.model,
        "transport": "offline-demo" if config.provider == "demo" else "chat-model",
    }

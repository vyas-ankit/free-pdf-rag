# src/core/llm.py

import os

from src.core.config import get_config, get_provider_config

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")


def get_llm_api_key(provider: str = None, llm_api_key: str = None) -> str:
    """Return the API key for the configured LLM provider."""
    if llm_api_key:
        return llm_api_key

    provider = (provider or get_llm_provider()).lower()
    provider_cfg = get_provider_config(provider)
    api_key_env = provider_cfg.get("api_key_env")

    if api_key_env:
        key = os.getenv(api_key_env)
        if key:
            return key

    return os.getenv("LLM_API_KEY")


def get_llm_provider(provider: str = None, llm_api_key: str = None) -> str:
    """Resolve the LLM provider: explicit arg > env var > config default."""
    if provider:
        return provider.strip().lower()

    env_provider = os.getenv("LLM_PROVIDER")
    if env_provider and env_provider.strip().lower() != "auto":
        return env_provider.strip().lower()

    return get_config().get("default_provider", "groq").lower()


def get_default_model(provider: str = None) -> str:
    """Return the default model for the provider.

    Env var LLM_MODEL acts as a global override only when no provider is
    explicitly specified — otherwise the provider's config default is used.
    This prevents a Google model name leaking into a Groq client (or vice versa).
    """
    if provider is None:
        configured_model = os.getenv("LLM_MODEL")
        if configured_model:
            return configured_model
        provider = get_llm_provider()

    provider = provider.lower()
    provider_cfg = get_provider_config(provider)
    return provider_cfg["default_model"]


def get_llm(model_name: str = None, llm_api_key: str = None, provider: str = None, temperature: float = None):
    """Initialize the configured chat model behind a provider-neutral interface."""
    provider = get_llm_provider(provider, llm_api_key)
    provider_cfg = get_provider_config(provider)

    model_name = model_name or get_default_model(provider)
    api_key = get_llm_api_key(provider, llm_api_key)
    if temperature is None:
        temperature = provider_cfg.get("temperature", 0)

    if not api_key:
        raise ValueError(
            f"Missing LLM API key. Set {provider_cfg.get('api_key_env')} or LLM_API_KEY."
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_kwargs = {
            "model": model_name,
            "google_api_key": api_key,
            "temperature": temperature,
            "request_timeout": int(os.getenv("LLM_REQUEST_TIMEOUT", str(provider_cfg.get("request_timeout", 120))))
        }
        api_transport = os.getenv("LLM_API_TRANSPORT")
        if api_transport:
            google_kwargs["api_transport"] = api_transport

        return ChatGoogleGenerativeAI(**google_kwargs)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temperature,
            timeout=int(os.getenv("LLM_REQUEST_TIMEOUT", str(provider_cfg.get("request_timeout", 120)))),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

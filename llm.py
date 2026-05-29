# llm.py

import os


DEFAULT_PROVIDER = "auto"
DEFAULT_GOOGLE_MODEL = "gemini-flash-latest"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


def get_llm_api_key(provider: str = None, llm_api_key: str = None) -> str:
    """Return the API key for the configured LLM provider."""
    if llm_api_key:
        return llm_api_key

    provider = (provider or get_llm_provider()).lower()
    if provider == "google":
        return os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "groq":
        return os.getenv("LLM_API_KEY") or os.getenv("GROQ_API_KEY")

    return os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")


def get_llm_provider(provider: str = None, llm_api_key: str = None) -> str:
    """Resolve the LLM provider from explicit input or environment."""
    resolved_provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if resolved_provider != "auto":
        return resolved_provider

    if llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("GROQ_API_KEY"):
        return "groq"

    return "google"


def get_default_model(provider: str = None) -> str:
    """Return a sensible model default for the active LLM provider."""
    provider = (provider or get_llm_provider()).lower()
    configured_model = os.getenv("LLM_MODEL")
    if configured_model:
        return configured_model
    if provider == "groq":
        return DEFAULT_GROQ_MODEL
    return DEFAULT_GOOGLE_MODEL


def get_llm(model_name: str = None, llm_api_key: str = None, provider: str = None, temperature: float = 0):
    """Initialize the configured chat model behind a provider-neutral interface."""
    provider = get_llm_provider(provider, llm_api_key)
    model_name = model_name or get_default_model(provider)
    api_key = get_llm_api_key(provider, llm_api_key)

    if not api_key:
        raise ValueError(
            "Missing LLM API key. Set LLM_API_KEY or the provider-specific key "
            "(GOOGLE_API_KEY for Google, GROQ_API_KEY for Groq)."
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=temperature
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

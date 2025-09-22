"""
Factory pattern per la gestione di diversi provider LLM e Embedding.
Supporta Google, OpenAI, Anthropic, Ollama per LLM e Google, OpenAI, HuggingFace per embeddings.
"""

import logging
from typing import Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Classe astratta per i provider LLM."""

    @abstractmethod
    def create_llm(self, model: str, **kwargs) -> Any:
        """Crea un'istanza del modello LLM."""
        pass

    @abstractmethod
    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Crea un'istanza del modello di embeddings."""
        pass


class GoogleProvider(LLMProvider):
    """Provider per Google Generative AI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_llm(self, model: str, **kwargs) -> Any:
        """Crea un modello Google Generative AI."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare Google provider installa: pip install langchain-google-genai"
            )

    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Crea embeddings Google Generative AI."""
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=self.api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare Google embeddings installa: pip install langchain-google-genai"
            )


class OpenAIProvider(LLMProvider):
    """Provider per OpenAI."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_llm(self, model: str, **kwargs) -> Any:
        """Crea un modello OpenAI."""
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                openai_api_key=self.api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare OpenAI provider installa: pip install langchain-openai"
            )

    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Crea embeddings OpenAI."""
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=self.api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare OpenAI embeddings installa: pip install langchain-openai"
            )


class AnthropicProvider(LLMProvider):
    """Provider per Anthropic Claude."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def create_llm(self, model: str, **kwargs) -> Any:
        """Crea un modello Anthropic Claude."""
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model,
                anthropic_api_key=self.api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare Anthropic provider installa: pip install langchain-anthropic"
            )

    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Anthropic non fornisce embeddings, usa un altro provider."""
        raise NotImplementedError(
            "Anthropic non fornisce embeddings. Usa Google o OpenAI per gli embeddings."
        )


class OllamaProvider(LLMProvider):
    """Provider per Ollama (modelli locali)."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def create_llm(self, model: str, **kwargs) -> Any:
        """Crea un modello Ollama."""
        try:
            from langchain_community.llms import Ollama
            return Ollama(
                model=model,
                base_url=self.base_url,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare Ollama provider installa: pip install langchain-community"
            )

    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Crea embeddings Ollama."""
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings(
                model=model,
                base_url=self.base_url,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare Ollama embeddings installa: pip install langchain-community"
            )


class HuggingFaceProvider:
    """Provider per HuggingFace embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Crea embeddings HuggingFace."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name=model,
                **kwargs
            )
        except ImportError:
            raise ImportError(
                "Per usare HuggingFace embeddings installa: pip install langchain-huggingface"
            )


class LLMFactory:
    """Factory per creare istanze di LLM e Embeddings."""

    @staticmethod
    def create_llm(provider: str, model: str, config, **kwargs) -> Any:
        """
        Crea un'istanza LLM basata sul provider specificato.

        Args:
            provider: Nome del provider ("google", "openai", "anthropic", "ollama")
            model: Nome del modello
            config: Configurazione dell'applicazione
            **kwargs: Parametri aggiuntivi per il modello

        Returns:
            Istanza del modello LLM

        Raises:
            ValueError: Se il provider non è supportato
        """
        logger.info(f"Creazione LLM: provider={provider}, model={model}")

        if provider == "google":
            if not config.google_api_key:
                raise ValueError("GOOGLE_API_KEY richiesta per provider Google")
            provider_instance = GoogleProvider(config.google_api_key)
            return provider_instance.create_llm(model, **kwargs)

        elif provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY richiesta per provider OpenAI")
            provider_instance = OpenAIProvider(config.openai_api_key)
            return provider_instance.create_llm(model, **kwargs)

        elif provider == "anthropic":
            if not config.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY richiesta per provider Anthropic")
            provider_instance = AnthropicProvider(config.anthropic_api_key)
            return provider_instance.create_llm(model, **kwargs)

        elif provider == "ollama":
            provider_instance = OllamaProvider()
            return provider_instance.create_llm(model, **kwargs)

        else:
            raise ValueError(f"Provider LLM non supportato: {provider}")

    @staticmethod
    def create_embeddings(provider: str, model: str, config, **kwargs) -> Any:
        """
        Crea un'istanza di embeddings basata sul provider specificato.

        Args:
            provider: Nome del provider ("google", "openai", "huggingface")
            model: Nome del modello
            config: Configurazione dell'applicazione
            **kwargs: Parametri aggiuntivi per il modello

        Returns:
            Istanza del modello di embeddings

        Raises:
            ValueError: Se il provider non è supportato
        """
        logger.info(f"Creazione Embeddings: provider={provider}, model={model}")

        if provider == "google":
            if not config.google_api_key:
                raise ValueError("GOOGLE_API_KEY richiesta per provider Google")
            provider_instance = GoogleProvider(config.google_api_key)
            return provider_instance.create_embeddings(model, **kwargs)

        elif provider == "openai":
            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY richiesta per provider OpenAI")
            provider_instance = OpenAIProvider(config.openai_api_key)
            return provider_instance.create_embeddings(model, **kwargs)

        elif provider == "huggingface":
            provider_instance = HuggingFaceProvider(config.huggingface_api_key)
            return provider_instance.create_embeddings(model, **kwargs)

        else:
            raise ValueError(f"Provider embeddings non supportato: {provider}")

    @staticmethod
    def get_available_models() -> dict:
        """
        Restituisce un dizionario dei modelli disponibili per provider.

        Returns:
            dict: Dizionario con provider e modelli disponibili
        """
        return {
            "google": {
                "llm": [
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash-lite"  # Supporto legacy
                ],
                "embeddings": [
                    "models/embedding-001",
                    "models/text-embedding-004"
                ]
            },
            "openai": {
                "llm": [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4-turbo-preview",
                    "gpt-4",
                    "gpt-5",
                    "o1-preview",
                    "o1-mini"
                ],
                "embeddings": [
                    "text-embedding-3-large",
                    "text-embedding-3-small",
                    "text-embedding-ada-002"
                ]
            },
            "anthropic": {
                "llm": [
                    "claude-3.7-sonnet",     # Claude Sonnet 3.7
                    "claude-4-sonnet",       # Claude Sonnet 4
                    "claude-3.5-haiku",      # Claude Haiku 3.5
                    "claude-4-opus"          # Claude Opus 4.x
                ],
                "embeddings": []
            }
        }

    @staticmethod
    def validate_model_combination(llm_provider: str, llm_model: str,
                                 embedding_provider: str, embedding_model: str) -> bool:
        """
        Valida se la combinazione di modelli è supportata.

        Args:
            llm_provider: Provider per LLM
            llm_model: Modello LLM
            embedding_provider: Provider per embeddings
            embedding_model: Modello embeddings

        Returns:
            bool: True se la combinazione è valida
        """
        available = LLMFactory.get_available_models()

        # Verifica LLM
        if llm_provider not in available:
            return False
        if llm_model not in available[llm_provider]["llm"]:
            return False

        # Verifica embeddings
        if embedding_provider not in available:
            return False
        if embedding_model not in available[embedding_provider]["embeddings"]:
            return False

        return True
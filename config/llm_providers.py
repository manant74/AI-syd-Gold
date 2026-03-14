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



class HuggingFaceProvider:
    """Provider per HuggingFace embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def create_embeddings(self, model: str, **kwargs) -> Any:
        """Crea embeddings HuggingFace."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            # La famiglia E5 richiede prefissi query:/passage: per la modalità asimmetrica
            e5_models = ("intfloat/multilingual-e5", "intfloat/e5")
            if any(model.startswith(prefix) for prefix in e5_models):
                kwargs.setdefault("encode_kwargs", {}).update({"prompt": "passage: "})
                kwargs.setdefault("query_encode_kwargs", {}).update({"prompt": "query: "})
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

        elif provider == "huggingface":
            provider_instance = HuggingFaceProvider(config.huggingface_api_key)
            return provider_instance.create_embeddings(model, **kwargs)

        else:
            raise ValueError(f"Provider embeddings non supportato: {provider}")

    # Mappa etichette UI -> model ID Google
    GOOGLE_MODEL_LABELS = {
        "Lite":  "gemini-3.1-flash-lite-preview",
        "Flash": "gemini-3.1-flash-preview",
        "Pro":   "gemini-3.1-pro-preview",
    }

    @staticmethod
    def get_available_models() -> dict:
        """
        Restituisce un dizionario dei modelli disponibili per provider.
        Solo Google è attualmente abilitato.

        Returns:
            dict: Dizionario con provider e modelli disponibili
        """
        return {
            "google": {
                "llm": list(LLMFactory.GOOGLE_MODEL_LABELS.values()),
                "embeddings": [
                    "models/text-embedding-004",
                    "models/embedding-001",
                ]
            },
            "huggingface": {
                "llm": [],
                "embeddings": [
                    "intfloat/multilingual-e5-small",
                    "BAAI/bge-m3",
                    "intfloat/multilingual-e5-large",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ]
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
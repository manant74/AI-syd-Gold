"""
Configurazione centralizzata per l'applicazione AI-syd-Gold.
Gestisce tutte le variabili di configurazione e ambiente.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class AppConfig:
    """
    Configurazione centralizzata dell'applicazione.
    Tutti i parametri possono essere sovrascritti tramite variabili d'ambiente.
    """

    # === CONFIGURAZIONE MODELLI ===
    llm_provider: str = "google"  # "google", "openai", "anthropic", "ollama"
    llm_model: str = "gemini-3.1-flash-lite-preview"
    embedding_provider: str = "huggingface"  # "google", "openai", "huggingface"
    embedding_model: str = "intfloat/multilingual-e5-small"

    # === CONFIGURAZIONE RETRIEVAL ===
    retriever_k: int = 5
    retriever_fetch_k: int = 20
    search_type: str = "mmr"  # "similarity" o "mmr"

    # === CONFIGURAZIONE CHUNKING ===
    chunk_size: int = 1000
    chunk_overlap: int = 200
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 100

    # === CONFIGURAZIONE TIMEOUT E RETRY ===
    embedding_timeout: int = 60
    embedding_retry_attempts: int = 3
    embedding_retry_delay: int = 5

    # === CONFIGURAZIONE BATCH PROCESSING ===
    batch_size: int = 50

    # === CONFIGURAZIONE PERCORSI ===
    pdf_directory: str = "pdfs"
    vector_store_directory: str = "vector_store_cache"

    # === CONFIGURAZIONE STREAMLIT ===
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"

    # === CONFIGURAZIONE LOGGING ===
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"

    # === CONFIGURAZIONE MULTIMODALE ===
    enable_multimodal: bool = False  # Usato solo dal batch indexing, non a runtime Streamlit
    ocr_confidence_threshold: int = 60
    min_image_size: tuple = (100, 50)
    cache_multimodal_results: bool = True

    # === CONFIGURAZIONE FUNZIONALITÀ AVANZATE ===
    use_hyde: bool = True
    retriever_lambda_mult: float = 0.6

    # === CONFIGURAZIONE SICUREZZA ===
    google_api_key: Optional[str] = field(default=None, repr=False)
    openai_api_key: Optional[str] = field(default=None, repr=False)
    anthropic_api_key: Optional[str] = field(default=None, repr=False)
    huggingface_api_key: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> 'AppConfig':
        """
        Crea un'istanza di configurazione leggendo le variabili d'ambiente.

        Returns:
            AppConfig: Istanza configurata con valori da environment o default
        """
        return cls(
            # Modelli
            llm_provider=os.getenv("LLM_PROVIDER", cls.llm_provider),
            llm_model=os.getenv("LLM_MODEL_NAME", cls.llm_model),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", cls.embedding_provider),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),

            # Retrieval
            retriever_k=int(os.getenv("RETRIEVER_K", cls.retriever_k)),
            retriever_fetch_k=int(os.getenv("RETRIEVER_FETCH_K", cls.retriever_fetch_k)),
            search_type=os.getenv("SEARCH_TYPE", cls.search_type),

            # Chunking
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", cls.parent_chunk_size)),
            parent_chunk_overlap=int(os.getenv("PARENT_CHUNK_OVERLAP", cls.parent_chunk_overlap)),
            child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", cls.child_chunk_size)),
            child_chunk_overlap=int(os.getenv("CHILD_CHUNK_OVERLAP", cls.child_chunk_overlap)),

            # Timeout e retry
            embedding_timeout=int(os.getenv("EMBEDDING_TIMEOUT", cls.embedding_timeout)),
            embedding_retry_attempts=int(os.getenv("EMBEDDING_RETRY_ATTEMPTS", cls.embedding_retry_attempts)),
            embedding_retry_delay=int(os.getenv("EMBEDDING_RETRY_DELAY", cls.embedding_retry_delay)),

            # Batch processing
            batch_size=int(os.getenv("BATCH_SIZE", cls.batch_size)),

            # Percorsi
            pdf_directory=os.getenv("PDF_DIRECTORY_PATH", cls.pdf_directory),
            vector_store_directory=os.getenv("VECTOR_STORE_PATH", cls.vector_store_directory),

            # Streamlit
            streamlit_port=int(os.getenv("STREAMLIT_PORT", cls.streamlit_port)),
            streamlit_host=os.getenv("STREAMLIT_HOST", cls.streamlit_host),

            # Logging
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            log_format=os.getenv("LOG_FORMAT", cls.log_format),

            # Multimodale
            enable_multimodal=os.getenv("ENABLE_MULTIMODAL", "false").lower() in ("true", "1", "yes"),
            ocr_confidence_threshold=int(os.getenv("OCR_CONFIDENCE_THRESHOLD", cls.ocr_confidence_threshold)),
            min_image_size=tuple(map(int, os.getenv("MIN_IMAGE_SIZE", "100,50").split(","))),
            cache_multimodal_results=os.getenv("CACHE_MULTIMODAL_RESULTS", "true").lower() in ("true", "1", "yes"),

            # Funzionalità avanzate
            use_hyde=os.getenv("USE_HYDE", "true").lower() in ("true", "1", "yes"),
            retriever_lambda_mult=float(os.getenv("RETRIEVER_LAMBDA_MULT", cls.retriever_lambda_mult)),

            # Sicurezza
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),  # Supporta entrambi i nomi
            huggingface_api_key=os.getenv("HUGGINGFACE_API_KEY")
        )

    @property
    def metadata_file_path(self) -> str:
        """Path del file metadata per la cache."""
        return os.path.join(self.vector_store_directory, "metadata.json")

    @property
    def faiss_index_path(self) -> str:
        """Path dell'indice FAISS."""
        return os.path.join(self.vector_store_directory, "faiss_index")

    @property
    def faiss_core_index_file(self) -> str:
        """Path del file core dell'indice FAISS."""
        return os.path.join(self.faiss_index_path, "index.faiss")

    @property
    def docstore_path(self) -> str:
        """Path del docstore pickle."""
        return os.path.join(self.vector_store_directory, "docstore.pkl")

    def validate(self) -> None:
        """
        Valida la configurazione e lancia eccezioni se ci sono problemi.

        Raises:
            ValueError: Se la configurazione non è valida
            FileNotFoundError: Se directory obbligatorie non esistono
        """
        # Validazione provider
        valid_llm_providers = ["google", "openai", "anthropic"]
        if self.llm_provider not in valid_llm_providers:
            raise ValueError(f"llm_provider deve essere uno di: {valid_llm_providers}")

        valid_embedding_providers = ["google", "openai", "huggingface"]
        if self.embedding_provider not in valid_embedding_providers:
            raise ValueError(f"embedding_provider deve essere uno di: {valid_embedding_providers}")

        # Validazione API key basata sul provider
        if self.llm_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY richiesta per provider 'google'")
        elif self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY richiesta per provider 'openai'")
        elif self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY richiesta per provider 'anthropic'")

        # Validazione API key per embeddings
        if self.embedding_provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY richiesta per embedding provider 'google'")
        elif self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY richiesta per embedding provider 'openai'")
        # HuggingFace embeddings non richiedono API key (modelli locali)
        # elif self.embedding_provider == "huggingface" and not self.huggingface_api_key:
        #     raise ValueError("HUGGINGFACE_API_KEY richiesta per embedding provider 'huggingface'")

        # Validazione valori numerici
        if self.retriever_k <= 0:
            raise ValueError("retriever_k deve essere maggiore di 0")

        if self.retriever_fetch_k < self.retriever_k:
            raise ValueError("retriever_fetch_k deve essere >= retriever_k")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size deve essere maggiore di 0")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap deve essere minore di chunk_size")

        # Validazione search_type
        if self.search_type not in ["similarity", "mmr"]:
            raise ValueError("search_type deve essere 'similarity' o 'mmr'")

        # Validazione percorsi (creazione se non esistono)
        pdf_path = Path(self.pdf_directory)
        if not pdf_path.exists():
            pdf_path.mkdir(parents=True, exist_ok=True)

        vector_store_path = Path(self.vector_store_directory)
        if not vector_store_path.exists():
            vector_store_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """
        Converte la configurazione in dizionario (per serializzazione).
        Esclude informazioni sensibili come API keys.

        Returns:
            dict: Configurazione come dizionario
        """
        config_dict = {}
        sensitive_keys = ['google_api_key', 'openai_api_key', 'anthropic_api_key', 'huggingface_api_key']
        for key, value in self.__dict__.items():
            if key not in sensitive_keys:  # Escludi informazioni sensibili
                config_dict[key] = value
        return config_dict

    def __str__(self) -> str:
        """String representation senza informazioni sensibili."""
        safe_dict = self.to_dict()
        return f"AppConfig({safe_dict})"


# Istanza globale di default
# Può essere importata direttamente: from config import config
config = AppConfig.from_env()
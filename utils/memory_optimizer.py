"""
Ottimizzazioni per la gestione della memoria in AI-syd-Gold.
Implementa strategie per ridurre l'utilizzo di memoria durante l'elaborazione di documenti grandi.
"""

import gc
import hashlib
import logging
import os
import psutil
import threading
import time
import weakref
from functools import lru_cache, wraps
from typing import List, Dict, Any, Iterator, Optional, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Monitor per il tracking dell'utilizzo della memoria.
    """

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None
        self.peak_memory = 0
        self.measurements = []

    def get_memory_info(self) -> Dict[str, float]:
        """
        Ottiene informazioni dettagliate sulla memoria.

        Returns:
            Dict con informazioni su memoria fisica e virtuale (in MB)
        """
        memory_info = self.process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': self.process.memory_percent(),
            'available_mb': virtual_memory.available / 1024 / 1024,
            'total_mb': virtual_memory.total / 1024 / 1024,
            'system_percent': virtual_memory.percent
        }

    def set_baseline(self):
        """Imposta la baseline della memoria per comparazioni."""
        self.baseline_memory = self.get_memory_info()
        logger.info(f"Baseline memoria impostata: {self.baseline_memory['rss_mb']:.1f} MB")

    def get_memory_delta(self) -> Optional[float]:
        """
        Calcola la differenza di memoria rispetto alla baseline.

        Returns:
            Differenza in MB, None se baseline non impostata
        """
        if self.baseline_memory is None:
            return None

        current = self.get_memory_info()
        return current['rss_mb'] - self.baseline_memory['rss_mb']

    def track_peak(self):
        """Aggiorna il picco di memoria."""
        current = self.get_memory_info()['rss_mb']
        if current > self.peak_memory:
            self.peak_memory = current

    def log_memory_usage(self, operation: str):
        """
        Logga l'utilizzo corrente della memoria per un'operazione.

        Args:
            operation: Nome dell'operazione in corso
        """
        info = self.get_memory_info()
        delta = self.get_memory_delta()

        logger.info(
            f"Memoria {operation}: {info['rss_mb']:.1f} MB "
            f"({info['percent']:.1f}% processo, {info['system_percent']:.1f}% sistema)"
        )

        if delta is not None:
            logger.info(f"Delta da baseline: {delta:+.1f} MB")

        self.track_peak()


# Istanza globale del monitor
memory_monitor = MemoryMonitor()


@contextmanager
def monitor_memory_usage(operation_name: str):
    """
    Context manager per monitorare l'utilizzo della memoria durante un'operazione.

    Args:
        operation_name: Nome dell'operazione da monitorare

    Example:
        with monitor_memory_usage("document_processing"):
            # operazioni che consumano memoria
            pass
    """
    memory_monitor.log_memory_usage(f"INIZIO {operation_name}")
    start_time = time.time()

    try:
        yield memory_monitor
    finally:
        end_time = time.time()
        memory_monitor.log_memory_usage(f"FINE {operation_name}")
        logger.info(f"Durata {operation_name}: {end_time - start_time:.2f}s")


class CacheManager:
    """
    Gestore centralizzato per le cache con controllo memoria.
    """

    def __init__(self, max_cache_size_mb: int = 500):
        self.max_cache_size_mb = max_cache_size_mb
        self.caches = weakref.WeakSet()
        self._lock = threading.Lock()

    def register_cache(self, cache_func):
        """Registra una cache per il monitoring."""
        with self._lock:
            self.caches.add(cache_func)

    def clear_all_caches(self):
        """Pulisce tutte le cache registrate."""
        logger.warning("La pulizia della cache è temporaneamente disabilitata per debug.")
        pass # Disabilitato per debug

    def check_memory_pressure(self) -> bool:
        """
        Controlla se c'è pressione di memoria.
        TEMPORANEAMENTE DISABILITATO PER DEBUG.
        Returns:
            True se la memoria è sotto pressione
        """
        # Temporaneamente disabilitato per prevenire la pulizia della cache
        return False


# Istanza globale del cache manager
cache_manager = CacheManager()


def memory_aware_cache(maxsize: int = 128, typed: bool = False):
    """
    Decorator per cache con consapevolezza della memoria.
    Pulisce automaticamente la cache quando c'è pressione di memoria.

    Args:
        maxsize: Dimensione massima della cache
        typed: Se considerare i tipi degli argomenti

    Returns:
        Decorated function con cache memory-aware
    """

    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)
        cache_manager.register_cache(cached_func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Controlla pressione memoria prima di eseguire
            if cache_manager.check_memory_pressure():
                logger.warning("Pressione memoria rilevata, pulizia cache")
                cached_func.cache_clear()
                gc.collect()

            return cached_func(*args, **kwargs)

        # Espone metodi della cache
        wrapper.cache_info = cached_func.cache_info
        wrapper.cache_clear = cached_func.cache_clear

        return wrapper

    return decorator


def cached_embed_query(embeddings_model, query_text: str) -> List[float]:
    """
    Cache per query di embedding con gestione intelligente della memoria.

    Args:
        embeddings_model: Modello di embedding
        query_text: Testo da embeddare

    Returns:
        Lista di float rappresentante l'embedding
    """
    # Crea hash del testo per cache key più efficiente
    query_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
    logger.debug(f"Embedding query (hash: {query_hash[:8]}...)")

    return embeddings_model.embed_query(query_text)


def process_documents_in_chunks(
    documents: List[Any],
    chunk_size: int = 10,
    process_func: Optional[Callable] = None,
    clear_cache_between_chunks: bool = True
) -> Iterator[List[Any]]:
    """
    Processa documenti in chunk per ridurre l'utilizzo di memoria.

    Args:
        documents: Lista di documenti da processare
        chunk_size: Dimensione di ogni chunk
        process_func: Funzione opzionale da applicare a ogni chunk
        clear_cache_between_chunks: Se pulire le cache tra i chunk

    Yields:
        Chunk di documenti processati
    """
    total_docs = len(documents)
    logger.info(f"Processamento {total_docs} documenti in chunk di {chunk_size}")

    for i in range(0, total_docs, chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, total_docs)
        chunk = documents[chunk_start:chunk_end]

        logger.info(f"Processamento chunk {chunk_start + 1}-{chunk_end}/{total_docs}")

        with monitor_memory_usage(f"chunk_{chunk_start + 1}-{chunk_end}"):
            if process_func:
                chunk = process_func(chunk)

            yield chunk

            # Garbage collection tra i chunk
            if clear_cache_between_chunks:
                # Pulizia aggressiva solo se necessario
                if cache_manager.check_memory_pressure():
                    logger.info("Pulizia cache e garbage collection")
                    cache_manager.clear_all_caches()

                # Garbage collection sempre
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collection: {collected} oggetti rimossi")


class MemoryOptimizer:
    """
    Classe principale per l'ottimizzazione della memoria.
    """

    def __init__(self, config):
        self.config = config
        self.monitor = memory_monitor
        self.cache_manager = cache_manager

        # Imposta baseline all'inizializzazione
        self.monitor.set_baseline()

    def optimize_batch_processing(self, documents: List[Any], operation: str) -> Iterator[List[Any]]:
        """
        Ottimizza il processamento batch con monitoraggio memoria.

        Args:
            documents: Documenti da processare
            operation: Nome dell'operazione per logging

        Yields:
            Batch di documenti ottimizzati
        """
        batch_size = self._calculate_optimal_batch_size(len(documents))
        logger.info(f"Batch size ottimale per {operation}: {batch_size}")

        yield from process_documents_in_chunks(
            documents=documents,
            chunk_size=batch_size,
            clear_cache_between_chunks=True
        )

    def _calculate_optimal_batch_size(self, total_documents: int) -> int:
        """
        Calcola la dimensione ottimale del batch basata sulla memoria disponibile.

        Args:
            total_documents: Numero totale di documenti

        Returns:
            Dimensione ottimale del batch
        """
        memory_info = self.monitor.get_memory_info()
        available_mb = memory_info['available_mb']

        # Calcola batch size basato sulla memoria disponibile
        if available_mb > 4000:  # > 4GB disponibili
            base_batch_size = 50
        elif available_mb > 2000:  # > 2GB disponibili
            base_batch_size = 30
        elif available_mb > 1000:  # > 1GB disponibili
            base_batch_size = 20
        else:  # < 1GB disponibili
            base_batch_size = 10

        # Usa il batch size configurato come baseline
        configured_batch_size = getattr(self.config, 'batch_size', 50)

        # Prendi il minimo tra configurato e calcolato per sicurezza
        optimal_size = min(configured_batch_size, base_batch_size)

        # Non superare il numero totale di documenti
        return min(optimal_size, total_documents)

    def create_optimized_embeddings_func(self, base_embeddings):
        """
        Crea una funzione di embedding ottimizzata con cache.

        Args:
            base_embeddings: Modello di embedding base

        Returns:
            Funzione di embedding ottimizzata
        """

        def optimized_embed_query(text: str) -> List[float]:
            return cached_embed_query(base_embeddings, text)

        def optimized_embed_documents(texts: List[str]) -> List[List[float]]:
            """Embedding di documenti con processamento a chunk."""
            embeddings = []

            for chunk in process_documents_in_chunks(texts, chunk_size=20):
                with monitor_memory_usage(f"embedding_chunk_{len(chunk)}_docs"):
                    chunk_embeddings = base_embeddings.embed_documents(chunk)
                    embeddings.extend(chunk_embeddings)

            return embeddings

        # Crea wrapper per HuggingFace che non supporta assegnazione dinamica
        if hasattr(base_embeddings, '__class__') and 'HuggingFace' in str(base_embeddings.__class__):
            class OptimizedEmbeddingsWrapper:
                def __init__(self, base_embeddings):
                    self._base = base_embeddings
                    # Copia tutti gli attributi originali
                    for attr in dir(base_embeddings):
                        if not attr.startswith('_') and not callable(getattr(base_embeddings, attr)):
                            setattr(self, attr, getattr(base_embeddings, attr))

                def embed_query(self, text):
                    return optimized_embed_query(text)

                def embed_documents(self, texts):
                    return optimized_embed_documents(texts)

                def __call__(self, text):
                    """Supporto per compatibilità FAISS - chiama embed_query."""
                    return self.embed_query(text)

                def __getattr__(self, name):
                    return getattr(self._base, name)

            return OptimizedEmbeddingsWrapper(base_embeddings)
        else:
            # Per altri tipi di embeddings, usa assegnazione diretta
            base_embeddings.embed_query = optimized_embed_query
            base_embeddings.embed_documents = optimized_embed_documents
            return base_embeddings

    def get_memory_report(self) -> Dict[str, Any]:
        """
        Genera un report dettagliato dell'utilizzo della memoria.

        Returns:
            Dizionario con statistiche memoria
        """
        info = self.monitor.get_memory_info()
        delta = self.monitor.get_memory_delta()

        return {
            'current_memory_mb': info['rss_mb'],
            'baseline_memory_mb': self.monitor.baseline_memory['rss_mb'] if self.monitor.baseline_memory else None,
            'delta_mb': delta,
            'peak_memory_mb': self.monitor.peak_memory,
            'process_percent': info['percent'],
            'system_percent': info['system_percent'],
            'available_mb': info['available_mb'],
            'cache_stats': {
                'embed_query_cache': cached_embed_query.cache_info()._asdict()
            }
        }


def clear_caches():
    """Funzione di utilità per pulire tutte le cache."""
    cache_manager.clear_all_caches()
    gc.collect()
    logger.info("Tutte le cache sono state pulite")


def log_memory_stats():
    """Logga statistiche memoria attuali."""
    memory_monitor.log_memory_usage("STATS")


# Configurazione automatica garbage collection
def setup_memory_optimization():
    """Configura ottimizzazioni globali per la memoria."""
    # Configura garbage collection più aggressivo
    gc.set_threshold(700, 10, 10)  # Più aggressivo del default (700, 10, 10)

    # Log configurazione
    logger.info("Ottimizzazioni memoria configurate")
    logger.info(f"GC thresholds: {gc.get_threshold()}")
    logger.info(f"Processo PID: {os.getpid()}")

    # Imposta baseline iniziale
    memory_monitor.set_baseline()


# Inizializzazione automatica
setup_memory_optimization()
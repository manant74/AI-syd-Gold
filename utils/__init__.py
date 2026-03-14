"""
Utilities package per AI-syd-Gold.
Contiene moduli di supporto per ottimizzazioni e funzionalità aggiuntive.
"""

from .memory_optimizer import (
    MemoryOptimizer,
    make_cached_embedder,
    process_documents_in_chunks,
    monitor_memory_usage,
    clear_caches
)

__all__ = [
    "MemoryOptimizer",
    "make_cached_embedder",
    "process_documents_in_chunks",
    "monitor_memory_usage",
    "clear_caches"
]
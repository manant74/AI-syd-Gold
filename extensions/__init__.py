"""Extensions package for AI-syd-Gold multimodal capabilities."""
# Import lazy: non carichiamo multimodal.py qui perché le sue dipendenze (cv2, pytesseract, PIL, numpy)
# sono pesanti e servono solo durante il batch processing da riga di comando, non a runtime Streamlit.

__all__ = ['MultimodalDocumentProcessor', 'ImageExtractionResult', 'ProcessingStats']

def __getattr__(name):
    if name in __all__:
        from .multimodal import MultimodalDocumentProcessor, ImageExtractionResult, ProcessingStats
        return locals()[name]
    raise AttributeError(f"module 'extensions' has no attribute {name!r}")
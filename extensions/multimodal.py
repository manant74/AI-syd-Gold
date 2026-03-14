"""
Multimodal Document Processor for AI-syd-Gold
Handles extraction of text from images, tables, and technical diagrams in PDF documents.
"""

import os
import io
import logging
import hashlib
import platform

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Import lazy delle librerie pesanti — vengono caricati solo al primo uso
cv2 = None
fitz = None
pytesseract = None
Image = None
ImageEnhance = None
ImageFilter = None
np = None

def _ensure_heavy_imports():
    """Carica le librerie pesanti la prima volta che servono."""
    global cv2, fitz, pytesseract, Image, ImageEnhance, ImageFilter, np
    if cv2 is not None:
        return  # già caricate

    import cv2 as _cv2
    import fitz as _fitz
    import pytesseract as _pytesseract
    from PIL import Image as _Image, ImageEnhance as _ImageEnhance, ImageFilter as _ImageFilter
    import numpy as _np

    cv2 = _cv2
    fitz = _fitz
    pytesseract = _pytesseract
    Image = _Image
    ImageEnhance = _ImageEnhance
    ImageFilter = _ImageFilter
    np = _np

    # Configurazione Tesseract PATH su Windows
    if platform.system() == 'Windows':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                logger.info(f"Tesseract trovato in: {path}")
                break
        else:
            logger.warning("Tesseract non trovato nei percorsi standard")

@dataclass
class ImageExtractionResult:
    """Risultato dell'estrazione da un'immagine."""
    text: str
    confidence: float
    image_type: str  # 'table', 'diagram', 'formula', 'text'
    metadata: Dict[str, Any]

@dataclass
class ProcessingStats:
    """Statistiche del processamento multimodale."""
    total_pages: int = 0
    images_found: int = 0
    images_processed: int = 0
    text_extracted_chars: int = 0
    processing_errors: int = 0

class MultimodalDocumentProcessor:
    """
    Processore avanzato per documenti PDF che estrae contenuto da:
    - Immagini con testo (OCR)
    - Tabelle tecniche
    - Diagrammi e schemi
    - Formule matematiche
    """

    def __init__(self, config=None):
        self.config = config or {}

        # Configurazione OCR
        self.ocr_config = '--oem 3 --psm 6'

        # Configurazione per diversi tipi di contenuto
        self.table_config = '--oem 3 --psm 6'  # Per tabelle
        self.formula_config = '--oem 3 --psm 8'  # Per formule matematiche

        # Soglie di confidenza (overridabili via config)
        self.min_confidence = getattr(config, 'ocr_confidence_threshold', 60)
        self.min_text_length = 10

        # Cache per evitare riprocessamento
        self._cache_dir = Path("cache/multimodal")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("MultimodalDocumentProcessor inizializzato")

    def process_pdf_document(self, pdf_path: str) -> Tuple[List[Document], ProcessingStats]:
        """
        Processa un PDF utilizzando un approccio ibrido: prima tenta l'estrazione
        di testo nativo, e solo se fallisce esegue l'OCR sulla pagina.

        Args:
            pdf_path: Percorso al file PDF

        Returns:
            Tuple di (documenti, statistiche)
        """
        _ensure_heavy_imports()
        logger.info(f"Inizio processamento ibrido: {pdf_path}")
        stats = ProcessingStats()
        extracted_documents = []

        try:
            pdf_document = fitz.open(pdf_path)
            stats.total_pages = pdf_document.page_count

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                
                # 1. Tentativo di estrazione testo nativo
                native_text = page.get_text().strip()

                # Euristica per decidere se il testo è sufficiente
                if len(native_text) > 100: # Considera la pagina come testuale
                    logger.info(f"Pagina {page_num}: Estratto testo nativo ({len(native_text)} caratteri).")
                    doc = Document(
                        page_content=native_text,
                        metadata={
                            'source': pdf_path,
                            'page': page_num,
                            'content_type': 'text',
                            'extraction_method': 'native'
                        }
                    )
                    extracted_documents.append(doc)
                    stats.text_extracted_chars += len(native_text)
                
                # 2. Fallback a OCR se non c'è testo nativo
                else:
                    logger.info(f"Pagina {page_num}: Testo nativo insufficiente. Esecuzione OCR...")
                    try:
                        result = self._ocr_scanned_page(page, pdf_path, page_num)
                        if result and result.text:
                            doc = self._create_document_from_image(
                                result, pdf_path, page_num, {} # img_data non è più rilevante qui
                            )
                            extracted_documents.append(doc)
                            stats.images_processed += 1
                            stats.text_extracted_chars += len(result.text)
                        else:
                            logger.warning(f"Pagina {page_num}: OCR non ha prodotto risultati validi.")
                            stats.processing_errors += 1
                    except Exception as e:
                        logger.error(f"Errore OCR pagina {page_num}: {e}")
                        stats.processing_errors += 1

            pdf_document.close()

        except Exception as e:
            logger.error(f"Errore critico processamento PDF {pdf_path}: {e}")
            stats.processing_errors += 1

        logger.info(f"Processamento completato: {len(extracted_documents)} documenti creati.")
        return extracted_documents, stats

    def _ocr_scanned_page(self, page, pdf_path: str, page_num: int) -> Optional[ImageExtractionResult]:
        """Esegue OCR su un'intera pagina, trattandola come un'immagine."""
        
        _ensure_heavy_imports()
        # Controllo cache per l'intera pagina
        cache_key = f"{Path(pdf_path).stem}_page{page_num}_full_ocr"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info(f"Pagina {page_num}: Trovato risultato OCR in cache.")
            return cached_result

        # Renderizza pagina ad alta risoluzione
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))

        # Usa il preprocessing esistente
        processed_image = self._preprocess_image(image)
        
        # Esegui OCR
        ocr_result = pytesseract.image_to_data(
            processed_image,
            config=self.ocr_config,
            output_type=pytesseract.Output.DICT
        )

        extracted_text, confidence = self._analyze_ocr_results(ocr_result)
        logger.info(f"OCR pagina {page_num} - Testo: '{extracted_text[:100]}...' Confidenza: {confidence:.1f}%")

        if confidence >= self.min_confidence and len(extracted_text) >= self.min_text_length:
            result = ImageExtractionResult(
                text=extracted_text,
                confidence=confidence,
                image_type='full_page_scan', # Nuovo tipo
                metadata={
                    'page_num': page_num,
                    'source_pdf': pdf_path
                }
            )
            self._cache_result(cache_key, result)
            return result
        
        return None

    def _extract_images_from_page(self, page, page_num: int) -> List[Dict]:
        """Estrae tutte le immagini da una pagina PDF."""
        images = []

        try:
            # Ottieni lista immagini dalla pagina
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    # Estrai dati immagine
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)

                    # Ottieni bbox in modo sicuro
                    try:
                        bbox = page.get_image_bbox(img)
                    except:
                        # Fallback: usa dimensioni complete della pagina
                        bbox = fitz.Rect(0, 0, base_image["width"], base_image["height"])

                    image_data = {
                        'data': base_image["image"],
                        'ext': base_image["ext"],
                        'width': base_image["width"],
                        'height': base_image["height"],
                        'bbox': bbox,
                        'page_num': page_num,
                        'image_index': img_index,
                        'xref': xref
                    }

                    # Filtra immagini troppo piccole (probabilmente decorative)
                    if image_data['width'] > 100 and image_data['height'] > 50:
                        images.append(image_data)

                except Exception as e:
                    logger.warning(f"Errore estrazione immagine {img_index} pagina {page_num}: {e}")

        except Exception as e:
            logger.warning(f"Errore estrazione immagini pagina {page_num}: {e}")

        return images

    def _process_image(self, img_data: Dict, pdf_path: str, page_num: int) -> Optional[ImageExtractionResult]:
        """Processa una singola immagine per estrarre testo."""

        _ensure_heavy_imports()
        # Controllo cache
        cache_key = self._generate_cache_key(img_data, pdf_path)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result

        try:
            # Converti dati in immagine PIL
            image = Image.open(io.BytesIO(img_data['data']))

            # Preprocessing per migliorare OCR
            processed_image = self._preprocess_image(image)

            # Determina il tipo di contenuto
            content_type = self._classify_image_content(processed_image, img_data)

            # Scegli configurazione OCR appropriata
            ocr_config = self._get_ocr_config_for_type(content_type)

            # Estrai testo con OCR
            ocr_result = pytesseract.image_to_data(
                processed_image,
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )

            # Analizza risultati OCR
            extracted_text, confidence = self._analyze_ocr_results(ocr_result)

            logger.info(f"OCR risultato - Testo: '{extracted_text[:100]}...' Confidenza: {confidence:.1f}%")

            if confidence >= self.min_confidence and len(extracted_text) >= self.min_text_length:
                result = ImageExtractionResult(
                    text=extracted_text,
                    confidence=confidence,
                    image_type=content_type,
                    metadata={
                        'page_num': page_num,
                        'bbox': img_data['bbox'],
                        'dimensions': (img_data['width'], img_data['height']),
                        'source_pdf': pdf_path
                    }
                )

                # Salva in cache
                self._cache_result(cache_key, result)
                return result

        except Exception as e:
            logger.warning(f"Errore processamento immagine: {e}")

        return None

    def _preprocess_image(self, image) -> "Image.Image":
        """Preprocessing dell'immagine per migliorare accuratezza OCR."""

        # Converti in RGB se necessario
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Ridimensiona se troppo piccola
        width, height = image.size
        if width < 300 or height < 200:
            scale_factor = max(300/width, 200/height)
            new_size = (int(width * scale_factor), int(height * scale_factor))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Migliora contrasto e nitidezza
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        # Riduci rumore
        image = image.filter(ImageFilter.MedianFilter(size=3))

        return image

    def _classify_image_content(self, image, img_data: Dict) -> str:
        """Classifica il tipo di contenuto dell'immagine."""

        # Converti in array numpy per analisi
        img_array = np.array(image)

        # Analisi dimensioni e aspect ratio
        width, height = image.size
        aspect_ratio = width / height

        # Euristica per classificazione
        if aspect_ratio > 2.0 and height < 100:
            # Probabile formula o testo orizzontale
            return 'formula'
        elif self._has_table_structure(img_array):
            # Possibile tabella
            return 'table'
        elif self._has_diagram_features(img_array):
            # Possibile diagramma tecnico
            return 'diagram'
        else:
            # Testo generico
            return 'text'

    def _has_table_structure(self, img_array: Any) -> bool:
        """Rileva se l'immagine contiene strutture tabulari."""

        # Converti in scala di grigi
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Rileva linee orizzontali e verticali
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        # Conta linee significative
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=50)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=50)

        return (h_lines is not None and len(h_lines) >= 2 and
                v_lines is not None and len(v_lines) >= 2)

    def _has_diagram_features(self, img_array: Any) -> bool:
        """Rileva caratteristiche tipiche di diagrammi tecnici."""

        # Converti in scala di grigi
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # Rileva contorni
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Conta forme geometriche
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30)
        geometric_shapes = len(contours) if contours else 0

        return geometric_shapes > 3 or (circles is not None and len(circles[0]) > 0)

    def _get_ocr_config_for_type(self, content_type: str) -> str:
        """Restituisce configurazione OCR ottimale per tipo di contenuto."""

        configs = {
            'table': self.table_config,
            'formula': self.formula_config,
            'diagram': self.ocr_config,
            'text': self.ocr_config
        }

        return configs.get(content_type, self.ocr_config)

    def _analyze_ocr_results(self, ocr_result: Dict) -> Tuple[str, float]:
        """Analizza risultati OCR e calcola confidenza."""

        words = []
        confidences = []

        for i, word in enumerate(ocr_result['text']):
            confidence = ocr_result['conf'][i]

            if confidence > 0 and word.strip():
                words.append(word.strip())
                confidences.append(confidence)

        if not words:
            return "", 0.0

        # Unisci parole e calcola confidenza media
        extracted_text = ' '.join(words)
        avg_confidence = sum(confidences) / len(confidences)

        # Post-processing del testo
        extracted_text = self._postprocess_text(extracted_text)

        return extracted_text, avg_confidence

    def _postprocess_text(self, text: str) -> str:
        """Post-processing del testo estratto."""

        # Rimuovi caratteri non necessari
        text = text.replace('\n\n', '\n').replace('  ', ' ')

        # Correggi comuni errori OCR per termini tecnici
        technical_corrections = {
            'O-ring': 'O-ring',
            'cuscinett0': 'cuscinetto',
            'diametr0': 'diametro',
            'rpm': 'rpm',
            'mm': 'mm',
            'µm': 'µm'
        }

        for wrong, correct in technical_corrections.items():
            text = text.replace(wrong, correct)

        return text.strip()

    def _create_document_from_image(self, result: ImageExtractionResult,
                                   pdf_path: str, page_num: int, img_data: Dict) -> Document:
        """Crea un Document LangChain dal risultato dell'estrazione."""

        # Metadati completi
        metadata = {
            'source': pdf_path,
            'page': page_num,
            'content_type': 'multimodal_image',
            'image_type': result.image_type,
            'confidence': result.confidence,
            'extraction_method': 'OCR',
            'image_dimensions': f"{img_data.get('width', 'N/A')}x{img_data.get('height', 'N/A')}",
            'bbox': str(img_data.get('bbox', 'N/A')),
            'chunk_type': f"image_{result.image_type}"
        }

        # Prefisso per identificare contenuto multimodale
        content = f"[IMMAGINE-{result.image_type.upper()}] {result.text}"

        return Document(
            page_content=content,
            metadata=metadata
        )

    def _generate_cache_key(self, img_data: Dict, pdf_path: str) -> str:
        """Genera chiave cache per immagine."""

        content_hash = hashlib.md5(img_data['data']).hexdigest()
        return f"{Path(pdf_path).stem}_p{img_data['page_num']}_i{img_data['image_index']}_{content_hash[:8]}"

    def _get_cached_result(self, cache_key: str) -> Optional[ImageExtractionResult]:
        """Recupera risultato dalla cache."""

        cache_file = self._cache_dir / f"{cache_key}.pickle"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Errore lettura cache {cache_key}: {e}")

        return None

    def _cache_result(self, cache_key: str, result: ImageExtractionResult):
        """Salva risultato in cache."""

        cache_file = self._cache_dir / f"{cache_key}.pickle"
        try:
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Errore scrittura cache {cache_key}: {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Restituisce statistiche del processamento multimodale."""

        cache_files = list(self._cache_dir.glob("*.pickle"))

        return {
            'cache_entries': len(cache_files),
            'cache_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            'supported_formats': ['PDF with embedded images'],
            'ocr_engine': 'Tesseract',
            'content_types': ['table', 'diagram', 'formula', 'text']
        }

    def clear_cache(self):
        """Pulisce la cache del processamento multimodale."""

        try:
            import shutil
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cache multimodale pulita")
        except Exception as e:
            logger.error(f"Errore pulizia cache: {e}")


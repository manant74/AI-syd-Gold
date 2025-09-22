"""
Configurazione globale per i test pytest.
Include fixtures condivise e configurazione test environment.
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from pathlib import Path

# Import del modulo principale
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import app
from config import AppConfig


@pytest.fixture(scope="session")
def test_env():
    """Fixture per configurare l'ambiente di test."""
    # Salva le variabili d'ambiente originali
    original_env = {
        'PDF_DIRECTORY_PATH': os.getenv('PDF_DIRECTORY_PATH'),
        'VECTOR_STORE_PATH': os.getenv('VECTOR_STORE_PATH'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
        'LLM_MODEL_NAME': os.getenv('LLM_MODEL_NAME')
    }

    # Crea directory temporanee per i test
    temp_dir = tempfile.mkdtemp(prefix="bearx_test_")
    pdf_dir = os.path.join(temp_dir, "pdfs")
    cache_dir = os.path.join(temp_dir, "cache")

    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Imposta variabili d'ambiente per i test
    os.environ['PDF_DIRECTORY_PATH'] = pdf_dir
    os.environ['VECTOR_STORE_PATH'] = cache_dir
    os.environ['LLM_MODEL_NAME'] = 'gemini-1.5-flash'

    # Patch della configurazione nel modulo app
    test_config = AppConfig(
        pdf_directory=pdf_dir,
        vector_store_directory=cache_dir,
        google_api_key="test_api_key"
    )

    with patch.object(app, 'config', test_config):
        yield {
            'temp_dir': temp_dir,
            'pdf_dir': pdf_dir,
            'cache_dir': cache_dir,
            'original_env': original_env,
            'config': test_config
        }

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temp directory {temp_dir}: {e}")

    # Ripristina variabili d'ambiente originali
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture
def sample_pdf_content():
    """Fixture con contenuto PDF di esempio per i test."""
    return [
        {
            'page_content': """
            Bearing Technology Manual

            Chapter 1: Introduction to Ball Bearings

            Ball bearings are rolling-element bearings that use balls to maintain
            separation between the bearing races. The purpose of a ball bearing is
            to reduce rotational friction and support radial and axial loads.

            Key specifications:
            - Inner diameter: 10-500mm
            - Operating temperature: -40°C to +120°C
            - Load capacity: Up to 50kN
            - Speed rating: Up to 20,000 rpm

            Material: Steel grade 100Cr6 (AISI 52100)
            Hardness: 58-65 HRC
            """,
            'metadata': {'source': 'test_bearing_manual.pdf', 'page': 0}
        },
        {
            'page_content': """
            Chapter 2: Lubrication Systems

            Proper lubrication is critical for bearing performance and longevity.

            Lubrication types:
            1. Grease lubrication
               - Temperature range: -30°C to +110°C
               - Relubrication interval: 1000-8000 hours

            2. Oil lubrication
               - Temperature range: -40°C to +200°C
               - Viscosity: ISO VG 32-150

            Formula for bearing life calculation:
            L10 = (C/P)^3 × 10^6 revolutions

            Where:
            - L10 = Basic rating life
            - C = Dynamic load rating
            - P = Equivalent dynamic load
            """,
            'metadata': {'source': 'test_bearing_manual.pdf', 'page': 1}
        }
    ]


@pytest.fixture
def mock_embeddings():
    """Mock per GoogleGenerativeAIEmbeddings."""
    mock = Mock()
    mock.embed_query.return_value = [0.1] * 1536  # Simula embedding vector
    mock.embed_documents.return_value = [[0.1] * 1536] * 10  # Lista di embedding
    return mock


@pytest.fixture
def mock_llm():
    """Mock per ChatGoogleGenerativeAI."""
    mock = Mock()
    mock.invoke.return_value = "Mocked LLM response for testing"
    return mock


@pytest.fixture
def sample_documents():
    """Fixture con documenti di esempio in formato LangChain."""
    from langchain.schema import Document

    return [
        Document(
            page_content="""
            Ball bearings are essential components in mechanical systems.
            They reduce friction and support loads in rotating machinery.
            Common applications include motors, pumps, and machine tools.

            Technical specifications:
            - Bore diameter: 10-200mm
            - Load rating: 5-50kN
            - Speed limit: 15,000 rpm
            """,
            metadata={'source': 'bearings_guide.pdf', 'page': 0}
        ),
        Document(
            page_content="""
            Maintenance procedures for industrial bearings:

            1. Visual inspection every 500 hours
            2. Vibration monitoring monthly
            3. Temperature monitoring continuously
            4. Grease replenishment every 2000 hours

            Warning signs of bearing failure:
            - Excessive noise
            - Vibration increase
            - Temperature rise
            - Grease discoloration
            """,
            metadata={'source': 'maintenance_manual.pdf', 'page': 5}
        )
    ]


@pytest.fixture
def test_queries():
    """Fixture con query di test tipiche."""
    return [
        "Cosa sono i cuscinetti a sfere?",
        "Come si calcola la durata di un cuscinetto?",
        "Quale lubrificazione è migliore per alte temperature?",
        "Quali sono i segni di guasto di un cuscinetto?",
        "Specifiche tecniche cuscinetti radiali",
        "Procedura di montaggio cuscinetti",
        "Materiali per cuscinetti industriali",
        "Calcolo carico dinamico equivalente"
    ]


@pytest.fixture
def mock_vector_store():
    """Mock per FAISS vector store."""
    mock = Mock()
    mock.similarity_search.return_value = []
    mock.similarity_search_with_score.return_value = []
    mock.save_local.return_value = None
    mock.load_local.return_value = mock
    return mock


# Marker personalizzati per categorizzare i test
def pytest_configure(config):
    """Configurazione markers personalizzati."""
    config.addinivalue_line(
        "markers", "unit: marca i test unitari veloci"
    )
    config.addinivalue_line(
        "markers", "integration: marca i test di integrazione"
    )
    config.addinivalue_line(
        "markers", "performance: marca i test di performance"
    )
    config.addinivalue_line(
        "markers", "slow: marca i test che richiedono molto tempo"
    )
    config.addinivalue_line(
        "markers", "api: marca i test che richiedono API esterne"
    )


# Hook per skip automatico dei test che richiedono API key
def pytest_runtest_setup(item):
    """Skip automatico test che richiedono API se non disponibile."""
    if item.get_closest_marker("api"):
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not available - skipping API test")
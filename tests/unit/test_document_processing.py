"""
Test unitari per le funzioni di processamento documenti.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import hashlib

import app


class TestDocumentValidation:
    """Test per le funzioni di validazione documenti."""

    def test_get_pdf_hash_valid_file(self, test_env):
        """Test calcolo hash per file PDF valido."""
        # Crea un file di test
        test_file = os.path.join(test_env['temp_dir'], 'test.pdf')
        test_content = b"Test PDF content"

        with open(test_file, 'wb') as f:
            f.write(test_content)

        # Calcola hash atteso
        expected_hash = hashlib.md5(test_content).hexdigest()

        # Test della funzione
        result_hash = app._get_pdf_hash(test_file)

        assert result_hash == expected_hash

    def test_get_pdf_hash_nonexistent_file(self):
        """Test calcolo hash per file inesistente."""
        result = app._get_pdf_hash("/path/to/nonexistent/file.pdf")
        assert result is None

    def test_get_pdf_metadata_empty_directory(self, test_env):
        """Test metadati per directory vuota."""
        result = app._get_pdf_metadata(test_env['pdf_dir'])
        assert result == {}

    def test_get_pdf_metadata_with_pdfs(self, test_env):
        """Test metadati per directory con PDF."""
        # Crea file PDF di test
        pdf_files = ['doc1.pdf', 'doc2.pdf', 'not_a_pdf.txt']
        test_contents = [b"PDF content 1", b"PDF content 2", b"Text content"]

        for filename, content in zip(pdf_files, test_contents):
            filepath = os.path.join(test_env['pdf_dir'], filename)
            with open(filepath, 'wb') as f:
                f.write(content)

        result = app._get_pdf_metadata(test_env['pdf_dir'])

        # Solo i file PDF dovrebbero essere inclusi
        assert len(result) == 2
        assert 'doc1.pdf' in result
        assert 'doc2.pdf' in result
        assert 'not_a_pdf.txt' not in result

        # Verifica che gli hash siano corretti
        expected_hash1 = hashlib.md5(b"PDF content 1").hexdigest()
        expected_hash2 = hashlib.md5(b"PDF content 2").hexdigest()

        assert result['doc1.pdf'] == expected_hash1
        assert result['doc2.pdf'] == expected_hash2

    def test_get_pdf_metadata_nonexistent_directory(self):
        """Test metadati per directory inesistente."""
        result = app._get_pdf_metadata("/path/to/nonexistent/directory")
        assert result == {}


class TestEnvironmentValidation:
    """Test per la validazione dell'ambiente."""

    def test_validate_environment_success(self, test_env):
        """Test validazione ambiente con configurazione corretta."""
        # Utilizza la configurazione di test già patchata nel test_env
        # Non dovrebbe lanciare eccezioni
        app.validate_environment()

    def test_validate_environment_creates_missing_dirs(self, test_env):
        """Test che la validazione crea le directory mancanti."""
        missing_dir = os.path.join(test_env['temp_dir'], 'missing')

        # Crea una configurazione di test con directory mancante
        from config import AppConfig
        test_config = AppConfig(
            pdf_directory=missing_dir,
            vector_store_directory=test_env['cache_dir'],
            google_api_key="test_key"
        )

        with patch.object(app, 'config', test_config):
            # La validazione dovrebbe creare le directory automaticamente
            app.validate_environment()
            # Verifica che la directory sia stata creata
            assert os.path.exists(missing_dir)

    def test_validate_environment_missing_api_key(self):
        """Test validazione con API key mancante."""
        from config import AppConfig

        bad_config = AppConfig(
            pdf_directory="test_dir",
            vector_store_directory="test_cache",
            google_api_key=None  # API key mancante
        )

        with patch.object(app, 'config', bad_config):
            with pytest.raises(ValueError, match="GOOGLE_API_KEY non configurata"):
                app.validate_environment()


class TestDocumentLoading:
    """Test per il caricamento documenti."""

    @patch('app.PyPDFDirectoryLoader')
    def test_load_and_validate_documents_success(self, mock_loader, test_env, sample_documents):
        """Test caricamento documenti con successo."""
        # Configura il mock
        mock_instance = Mock()
        mock_instance.load.return_value = sample_documents
        mock_loader.return_value = mock_instance

        # Utilizza la configurazione già patchata nel test_env
        result = app.load_and_validate_documents()

        assert result == sample_documents
        mock_loader.assert_called_once_with(test_env['pdf_dir'], glob="**/*.pdf")

    @patch('app.PyPDFDirectoryLoader')
    def test_load_and_validate_documents_empty(self, mock_loader, test_env):
        """Test caricamento documenti vuoto."""
        # Configura il mock per restituire lista vuota
        mock_instance = Mock()
        mock_instance.load.return_value = []
        mock_loader.return_value = mock_instance

        # Utilizza la configurazione già patchata nel test_env
        result = app.load_and_validate_documents()

        assert result is None

    @patch('app.PyPDFDirectoryLoader')
    @patch('app.debug_pdf_content')
    def test_load_and_validate_documents_insufficient_content(self, mock_debug, mock_loader, test_env):
        """Test caricamento documenti con contenuto insufficiente."""
        # Documenti con contenuto scarso
        poor_documents = [
            Mock(page_content="", metadata={'source': 'empty.pdf'}),
            Mock(page_content="a", metadata={'source': 'minimal.pdf'})
        ]

        mock_instance = Mock()
        mock_instance.load.return_value = poor_documents
        mock_loader.return_value = mock_instance

        # debug_pdf_content restituisce False per contenuto insufficiente
        mock_debug.return_value = False

        # Utilizza la configurazione già patchata nel test_env
        with pytest.raises(ValueError, match="Nessun contenuto significativo trovato"):
            app.load_and_validate_documents()

    @patch('app.PyPDFDirectoryLoader')
    def test_load_and_validate_documents_exception(self, mock_loader, test_env):
        """Test gestione eccezioni durante caricamento."""
        mock_loader.side_effect = Exception("PDF loading error")

        # Utilizza la configurazione già patchata nel test_env
        with pytest.raises(Exception, match="PDF loading error"):
            app.load_and_validate_documents()


class TestContentDebugging:
    """Test per le funzioni di debug contenuto."""

    def test_debug_pdf_content_empty_list(self):
        """Test debug con lista documenti vuota."""
        result = app.debug_pdf_content([])
        assert result is False

    def test_debug_pdf_content_meaningful_documents(self, sample_documents):
        """Test debug con documenti significativi."""
        # Patch print per evitare output durante i test
        with patch('builtins.print'):
            result = app.debug_pdf_content(sample_documents)

        assert result is True

    def test_debug_pdf_content_poor_documents(self):
        """Test debug con documenti di scarsa qualità."""
        poor_docs = [
            Mock(page_content="a", metadata={'source': 'poor1.pdf'}),
            Mock(page_content="", metadata={'source': 'poor2.pdf'})
        ]

        with patch('builtins.print'):
            result = app.debug_pdf_content(poor_docs)

        assert result is False

    def test_debug_text_chunks_empty(self):
        """Test debug chunks con lista vuota."""
        with patch('builtins.print'):
            result = app.debug_text_chunks([])

        assert result is False

    def test_debug_text_chunks_meaningful(self, sample_documents):
        """Test debug chunks con contenuto significativo."""
        with patch('builtins.print'):
            result = app.debug_text_chunks(sample_documents)

        assert result is True


class TestEmbeddings:
    """Test per l'inizializzazione embeddings."""

    @patch('app.GoogleGenerativeAIEmbeddings')
    def test_initialize_embeddings_success(self, mock_embeddings_class):
        """Test inizializzazione embeddings con successo."""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536
        mock_embeddings_class.return_value = mock_embeddings

        result = app.initialize_embeddings()

        # Il risultato è ora un oggetto ottimizzato, non il mock originale
        assert result is not None
        mock_embeddings_class.assert_called_once_with(model="models/embedding-001")
        mock_embeddings.embed_query.assert_called_once_with("test")

    @patch('app.GoogleGenerativeAIEmbeddings')
    def test_initialize_embeddings_failure(self, mock_embeddings_class):
        """Test fallimento inizializzazione embeddings."""
        mock_embeddings_class.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="Errore nell'inizializzazione"):
            app.initialize_embeddings()

    @patch('app.GoogleGenerativeAIEmbeddings')
    def test_initialize_embeddings_empty_response(self, mock_embeddings_class):
        """Test inizializzazione con risposta vuota."""
        mock_embeddings = Mock()
        mock_embeddings.embed_query.return_value = None
        mock_embeddings_class.return_value = mock_embeddings

        with pytest.raises(RuntimeError, match="Test embedding fallito"):
            app.initialize_embeddings()
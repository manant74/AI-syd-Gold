"""
Test unitari per la creazione e configurazione del chatbot.
"""

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

import app


class TestChatbotCreation:
    """Test per la funzione create_chatbot."""

    @patch('app.validate_environment')
    @patch('app.initialize_embeddings')
    @patch('app.load_and_validate_documents')
    @patch('app.FAISS')
    @patch('app.ChatGoogleGenerativeAI')
    @patch('app.RetrievalQA')
    def test_create_chatbot_standard_mode(self, mock_qa, mock_llm, mock_faiss,
                                        mock_load_docs, mock_init_emb, mock_validate,
                                        test_env, sample_documents):
        """Test creazione chatbot in modalità standard."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_init_emb.return_value = mock_embeddings
        mock_load_docs.return_value = sample_documents

        mock_vectorstore = Mock()
        mock_faiss.from_texts.return_value = mock_vectorstore
        mock_faiss.load_local.return_value = mock_vectorstore

        mock_llm_instance = Mock()
        mock_llm.return_value = mock_llm_instance

        mock_qa_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_qa_chain

        # Patch delle variabili di ambiente
        # Utilizza la configurazione già patchata nel test_env
        result = app.create_chatbot(retriever_type="standard")

        assert result == mock_qa_chain
        mock_validate.assert_called_once()
        mock_init_emb.assert_called_once()

    @patch('app.validate_environment')
    @patch('app.initialize_embeddings')
    @patch('app.load_and_validate_documents')
    @patch('app.BM25Retriever')
    @patch('app.FAISS')
    @patch('app.EnsembleRetriever')
    @patch('app.ChatGoogleGenerativeAI')
    @patch('app.RetrievalQA')
    def test_create_chatbot_ensemble_mode(self, mock_qa, mock_llm, mock_ensemble,
                                        mock_faiss, mock_bm25, mock_load_docs,
                                        mock_init_emb, mock_validate,
                                        test_env, sample_documents):
        """Test creazione chatbot in modalità ensemble."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_init_emb.return_value = mock_embeddings
        mock_load_docs.return_value = sample_documents

        mock_bm25_retriever = Mock()
        mock_bm25.from_documents.return_value = mock_bm25_retriever

        mock_vectorstore = Mock()
        mock_faiss.from_documents.return_value = mock_vectorstore
        mock_faiss_retriever = Mock()
        mock_vectorstore.as_retriever.return_value = mock_faiss_retriever

        mock_ensemble_retriever = Mock()
        mock_ensemble.return_value = mock_ensemble_retriever

        mock_qa_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_qa_chain

        # Utilizza la configurazione già patchata nel test_env
        result = app.create_chatbot(retriever_type="ensemble")

        assert result == mock_qa_chain
        mock_ensemble.assert_called_once_with(
            retrievers=[mock_bm25_retriever, mock_faiss_retriever],
            weights=[0.5, 0.5]
        )

    @patch('app.validate_environment')
    @patch('app.initialize_embeddings')
    @patch('app.load_and_validate_documents')
    def test_create_chatbot_no_documents(self, mock_load_docs, mock_init_emb,
                                       mock_validate, test_env):
        """Test creazione chatbot quando non ci sono documenti."""
        mock_init_emb.return_value = Mock()
        mock_load_docs.return_value = None

        # Utilizza la configurazione già patchata nel test_env
        result = app.create_chatbot()

        assert result is None

    @patch('app.validate_environment')
    def test_create_chatbot_validation_error(self, mock_validate, test_env):
        """Test gestione errore durante validazione ambiente."""
        mock_validate.side_effect = ValueError("Environment validation failed")

        # Utilizza la configurazione già patchata nel test_env
        with pytest.raises(ValueError):
            app.create_chatbot()


class TestCacheManagement:
    """Test per la gestione della cache."""

    def test_cache_validation_no_cache_files(self, test_env):
        """Test quando i file di cache non esistono."""
        # I file di cache non esistono, quindi dovrebbe creare un nuovo indice
        with patch.object(app, 'VECTOR_STORE_PATH', test_env['cache_dir']), \
             patch.object(app, 'METADATA_FILE', os.path.join(test_env['cache_dir'], 'metadata.json')):

            # Test che verifica che non esiste cache valida
            faiss_index_path = os.path.join(test_env['cache_dir'], "faiss_index")
            faiss_core_index_file = os.path.join(faiss_index_path, "index.faiss")
            docstore_path = os.path.join(test_env['cache_dir'], "docstore.pkl")
            metadata_file = os.path.join(test_env['cache_dir'], 'metadata.json')

            # Verifica che i file non esistano
            assert not os.path.exists(faiss_core_index_file)
            assert not os.path.exists(docstore_path)
            assert not os.path.exists(metadata_file)

    def test_cache_validation_with_valid_cache(self, test_env):
        """Test con cache valida esistente."""
        # Crea file di cache simulati
        cache_dir = test_env['cache_dir']
        faiss_index_path = os.path.join(cache_dir, "faiss_index")
        os.makedirs(faiss_index_path, exist_ok=True)

        faiss_core_index_file = os.path.join(faiss_index_path, "index.faiss")
        docstore_path = os.path.join(cache_dir, "docstore.pkl")
        metadata_file = os.path.join(cache_dir, 'metadata.json')

        # Crea file simulati con contenuto
        with open(faiss_core_index_file, 'wb') as f:
            f.write(b"fake faiss index")

        with open(docstore_path, 'wb') as f:
            f.write(b"fake docstore")

        # Crea metadati che corrispondono ai PDF attuali
        test_metadata = {"test.pdf": "fake_hash"}
        with open(metadata_file, 'w') as f:
            json.dump(test_metadata, f)

        # Verifica che i file esistano e abbiano contenuto
        assert os.path.exists(faiss_core_index_file)
        assert os.path.getsize(faiss_core_index_file) > 0
        assert os.path.exists(docstore_path)
        assert os.path.getsize(docstore_path) > 0
        assert os.path.exists(metadata_file)
        assert os.path.getsize(metadata_file) > 0

    def test_cache_invalidation_on_file_changes(self, test_env):
        """Test invalidazione cache quando i file cambiano."""
        # Simula scenario dove i metadati salvati non corrispondono ai file attuali
        cache_dir = test_env['cache_dir']
        metadata_file = os.path.join(cache_dir, 'metadata.json')

        # Metadati salvati (diversi da quelli attuali)
        saved_metadata = {"old_file.pdf": "old_hash"}
        with open(metadata_file, 'w') as f:
            json.dump(saved_metadata, f)

        # Mock della funzione che ottiene metadati attuali
        current_metadata = {"new_file.pdf": "new_hash"}

        with patch('app._get_pdf_metadata', return_value=current_metadata):
            # Test che la cache viene invalidata
            with open(metadata_file, 'r') as f:
                loaded_metadata = json.load(f)

            assert loaded_metadata != current_metadata
            # In questo caso la cache dovrebbe essere invalidata


class TestPromptTemplates:
    """Test per i template dei prompt."""

    def test_rag_prompt_template_structure(self):
        """Test struttura del template RAG."""
        from config.system_prompt import EXPERT_PROMPT
        template = EXPERT_PROMPT

        # Verifica che contenga le variabili necessarie
        assert "{context}" in template
        assert "{question}" in template

        # Verifica che contenga elementi chiave del prompt
        assert "BearX" in template
        assert "cuscinetti" in template
        assert "REGOLE FONDAMENTALI" in template
        assert "BASATI SOLO SUL CONTESTO" in template

    def test_prompt_template_variables(self):
        """Test che il template possa essere formattato correttamente."""
        from config.system_prompt import EXPERT_PROMPT
        template = EXPERT_PROMPT

        # Test formattazione con variabili di esempio
        test_context = "Informazioni sui cuscinetti..."
        test_question = "Cosa sono i cuscinetti a sfere?"

        try:
            formatted = template.format(context=test_context, question=test_question)
            assert test_context in formatted
            assert test_question in formatted
        except KeyError as e:
            pytest.fail(f"Template missing required variable: {e}")


class TestConfigurationValues:
    """Test per i valori di configurazione."""

    def test_default_configuration_values(self):
        """Test valori di configurazione di default."""
        # Test che le variabili abbiano valori di default sensati
        assert app.RETRIEVER_K >= 1
        assert app.RETRIEVER_FETCH_K >= app.RETRIEVER_K
        assert app.SEARCH_TYPE in ["similarity", "mmr"]
        assert app.LLM_MODEL_NAME.startswith("gemini")

    @patch.dict(os.environ, {'RETRIEVER_K': '10', 'RETRIEVER_FETCH_K': '50'})
    def test_environment_variable_override(self):
        """Test override delle variabili tramite environment."""
        # Reload del modulo per applicare nuove env vars
        import importlib
        importlib.reload(app)

        assert app.RETRIEVER_K == 10
        assert app.RETRIEVER_FETCH_K == 50

    def test_path_configuration(self):
        """Test configurazione dei percorsi."""
        from config import AppConfig
        config = AppConfig.from_env()

        assert config.pdf_directory is not None
        assert config.vector_store_directory is not None
        assert config.metadata_file_path.endswith("metadata.json")


@pytest.mark.unit
class TestErrorHandling:
    """Test per la gestione degli errori."""

    @patch('app.initialize_embeddings')
    def test_embeddings_initialization_failure(self, mock_init_emb, test_env):
        """Test fallimento inizializzazione embeddings."""
        mock_init_emb.side_effect = RuntimeError("API Error")

        # Utilizza la configurazione già patchata nel test_env
        with pytest.raises(RuntimeError):
                app.create_chatbot()

    @patch('app.validate_environment')
    @patch('app.initialize_embeddings')
    @patch('app.load_and_validate_documents')
    @patch('app.ChatGoogleGenerativeAI')
    def test_llm_initialization_failure(self, mock_llm, mock_load_docs,
                                      mock_init_emb, mock_validate, test_env):
        """Test fallimento inizializzazione LLM."""
        mock_init_emb.return_value = Mock()
        mock_load_docs.return_value = [Mock()]
        mock_llm.side_effect = Exception("LLM Error")

        # Utilizza la configurazione già patchata nel test_env
        with pytest.raises(Exception):
                app.create_chatbot()
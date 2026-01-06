import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.db_manager import DatabaseManager
except ImportError:
    # Allow test to be collected even if module is missing, 
    # but fail the test case that needs it
    DatabaseManager = None

class TestDatabaseManager:
    @pytest.fixture
    def mock_chroma_client(self):
        with patch('src.utils.db_manager.chromadb.PersistentClient') as mock:
            yield mock

    def test_import(self):
        """Fail if DatabaseManager cannot be imported"""
        assert DatabaseManager is not None, "Could not import DatabaseManager from src.utils.db_manager"

    def test_initialization(self, mock_chroma_client):
        """Test that DatabaseManager initializes the ChromaDB client correctly."""
        if DatabaseManager is None:
            pytest.fail("DatabaseManager not imported")
            
        db_path = "./test_db"
        manager = DatabaseManager(db_path=db_path)
        
        # Check if PersistentClient was called with correct path
        mock_chroma_client.assert_called_once_with(path=db_path)
        
        # Check if collection was obtained
        # We assume the constructor calls get_or_create_collection
        manager.client.get_or_create_collection.assert_called()

    def test_default_collection_name(self, mock_chroma_client):
        """Test that the collection is created with the default name."""
        if DatabaseManager is None:
            pytest.fail("DatabaseManager not imported")

        manager = DatabaseManager()
        collection_name = "tracking_events"
        
        # Verify get_or_create_collection was called with correct name
        manager.client.get_or_create_collection.assert_called_with(name=collection_name)

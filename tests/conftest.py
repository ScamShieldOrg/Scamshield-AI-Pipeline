import os
import sys
from pathlib import Path

os.environ["PHISHTANK_AUTO_DOWNLOAD"] = "false"
os.environ["RAG_ENABLED"] = "false"
os.environ["GEMINI_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["SAFE_BROWSING_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import pytest
from unittest.mock import patch

ROOT_DIR = str(Path(__file__).resolve().parents[1])
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


@pytest.fixture(autouse=True)
def mock_db_persistence_in_tests():
    """Mock database persistence during unit tests so offline test runner does not require live Postgres."""
    with patch("api.app.save_scan") as mock_save_scan, \
         patch("api.visobert_app.save_prediction") as mock_save_pred:
        yield

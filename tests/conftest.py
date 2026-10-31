import os

os.environ["PHISHTANK_AUTO_DOWNLOAD"] = "false"

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_db_persistence_in_tests():
    """Mock database persistence during unit tests so offline test runner does not require live Postgres."""
    with patch("api.app.save_scan") as mock_save_scan, \
         patch("api.visobert_app.save_prediction") as mock_save_pred:
        yield

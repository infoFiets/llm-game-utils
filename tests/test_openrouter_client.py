"""Tests for OpenRouterClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from llm_game_utils.clients import OpenRouterClient, LLMResponse


class TestOpenRouterClient:
    """Test suite for OpenRouterClient."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Set up mock environment variables."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")

    @pytest.fixture
    def client(self, mock_env):
        """Create a test client instance."""
        return OpenRouterClient()

    def test_initialization_with_api_key(self):
        """Test client initialization with API key."""
        client = OpenRouterClient(api_key="test-key")
        assert client.api_key == "test-key"

    def test_initialization_from_env(self, mock_env):
        """Test client initialization from environment variable."""
        client = OpenRouterClient()
        assert client.api_key == "test-api-key"

    def test_initialization_without_api_key_raises_error(self, monkeypatch):
        """Test that initialization fails without API key."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterClient()

    def test_add_model_config(self, client):
        """Test adding model configuration."""
        client.add_model_config(
            model_id="test/model",
            name="Test Model",
            input_cost=0.01,
            output_cost=0.03
        )

        assert "test/model" in client.model_configs
        assert client.model_configs["test/model"]["name"] == "Test Model"
        assert client.model_configs["test/model"]["pricing"]["input"] == 0.01
        assert client.model_configs["test/model"]["pricing"]["output"] == 0.03

    def test_calculate_cost(self, client):
        """Test cost calculation."""
        client.add_model_config(
            model_id="test/model",
            name="Test Model",
            input_cost=0.01,
            output_cost=0.03
        )

        cost = client.calculate_cost("test/model", 1000, 500)
        # (1000/1000 * 0.01) + (500/1000 * 0.03) = 0.01 + 0.015 = 0.025
        assert cost == 0.025

    def test_calculate_cost_no_config(self, client):
        """Test cost calculation for unconfigured model."""
        cost = client.calculate_cost("unknown/model", 1000, 500)
        assert cost == 0.0

    def test_get_available_models(self, client):
        """Test getting available models."""
        client.add_model_config("model1", "Model 1", 0.01, 0.03)
        client.add_model_config("model2", "Model 2", 0.02, 0.04)

        models = client.get_available_models()
        assert len(models) == 2
        assert "model1" in models
        assert "model2" in models

    def test_get_model_info(self, client):
        """Test getting model info."""
        client.add_model_config(
            model_id="test/model",
            name="Test Model",
            input_cost=0.01,
            output_cost=0.03
        )

        info = client.get_model_info("test/model")
        assert info is not None
        assert info["id"] == "test/model"
        assert info["name"] == "Test Model"
        assert info["pricing"]["input"] == 0.01

    def test_get_model_info_not_found(self, client):
        """Test getting info for unknown model."""
        info = client.get_model_info("unknown/model")
        assert info is None

    @patch('llm_game_utils.clients.openrouter_client.httpx.Client')
    def test_query_success(self, mock_client_class, client):
        """Test successful query."""
        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        mock_response.raise_for_status = Mock()

        # Mock the client instance
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_response
        mock_client_class.return_value = mock_client_instance

        # Re-create client to use mock
        client = OpenRouterClient(api_key="test-key")
        client.add_model_config("test/model", "Test", 0.01, 0.03)

        response = client.query(
            model_id="test/model",
            prompt="Test prompt"
        )

        assert isinstance(response, LLMResponse)
        assert response.response == "Test response"
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_tokens == 30


# Example of how to run tests:
# pytest tests/test_openrouter_client.py -v

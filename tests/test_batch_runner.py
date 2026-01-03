"""Tests for batch runner functionality."""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from llm_game_utils.clients.batch_runner import BatchRunner
from llm_game_utils.clients.base_client import LLMResponse


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self, responses=None, should_fail=None):
        """Initialize with predefined responses or failure conditions.

        Args:
            responses: Dict mapping model_id to response text
            should_fail: Set of model_ids that should raise exceptions
        """
        self.responses = responses or {}
        self.should_fail = should_fail or set()
        self.query_calls = []

    def query(self, model_id, prompt, **kwargs):
        """Mock query method."""
        self.query_calls.append({
            'model_id': model_id,
            'prompt': prompt,
            **kwargs
        })

        if model_id in self.should_fail:
            raise Exception(f"Mock error for {model_id}")

        response_text = self.responses.get(model_id, f"Response from {model_id}")

        return LLMResponse(
            model_id=model_id,
            model_name=model_id,
            prompt=prompt,
            response=response_text,
            timestamp=datetime.now(),
            response_time=0.5,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.01,
            metadata={}
        )


class TestBatchRunnerInitialization:
    """Test batch runner initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        client = MockLLMClient()
        runner = BatchRunner(client)
        assert runner.client == client


class TestQueryAllModels:
    """Test querying multiple models."""

    def test_query_single_model(self):
        """Test querying a single model."""
        client = MockLLMClient({'model-1': 'Response 1'})
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1'],
            prompt="Test prompt"
        )

        assert 'model-1' in results
        assert results['model-1']['success'] is True
        assert results['model-1']['response'] == 'Response 1'
        assert results['model-1']['cost'] == 0.01

    def test_query_multiple_models_sequential(self):
        """Test querying multiple models sequentially."""
        client = MockLLMClient({
            'model-1': 'Response 1',
            'model-2': 'Response 2',
            'model-3': 'Response 3'
        })
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1', 'model-2', 'model-3'],
            prompt="Test prompt",
            parallel=False
        )

        assert len(results) == 3
        assert all(results[m]['success'] for m in results)
        assert results['model-1']['response'] == 'Response 1'
        assert results['model-2']['response'] == 'Response 2'
        assert results['model-3']['response'] == 'Response 3'

    def test_query_with_parameters(self):
        """Test querying with specific parameters."""
        client = MockLLMClient({'model-1': 'Response'})
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1'],
            prompt="Test",
            system_prompt="System",
            temperature=0.5,
            max_tokens=100
        )

        # Check that parameters were passed to client
        call = client.query_calls[0]
        assert call['prompt'] == "Test"
        assert call['system_prompt'] == "System"
        assert call['temperature'] == 0.5
        assert call['max_tokens'] == 100

    def test_query_parallel(self):
        """Test parallel query execution."""
        client = MockLLMClient({
            'model-1': 'Response 1',
            'model-2': 'Response 2'
        })
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1', 'model-2'],
            prompt="Test",
            parallel=True,
            max_workers=2
        )

        assert len(results) == 2
        assert all(results[m]['success'] for m in results)

    def test_error_handling_one_model_fails(self):
        """Test that one model failing doesn't stop others."""
        client = MockLLMClient(
            responses={'model-1': 'Success'},
            should_fail={'model-2'}
        )
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1', 'model-2'],
            prompt="Test",
            parallel=False
        )

        assert len(results) == 2
        assert results['model-1']['success'] is True
        assert results['model-2']['success'] is False
        assert 'error' in results['model-2']
        assert results['model-2']['error'] is not None

    def test_error_handling_parallel(self):
        """Test error handling in parallel mode."""
        client = MockLLMClient(
            responses={'model-1': 'Success'},
            should_fail={'model-2'}
        )
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1', 'model-2'],
            prompt="Test",
            parallel=True
        )

        assert len(results) == 2
        assert results['model-1']['success'] is True
        assert results['model-2']['success'] is False

    def test_result_structure(self):
        """Test that result structure is correct."""
        client = MockLLMClient({'model-1': 'Test response'})
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1'],
            prompt="Test"
        )

        result = results['model-1']
        assert 'response' in result
        assert 'cost' in result
        assert 'tokens' in result
        assert 'input_tokens' in result
        assert 'output_tokens' in result
        assert 'time' in result
        assert 'model_name' in result
        assert 'error' in result
        assert 'success' in result

    def test_failed_result_structure(self):
        """Test result structure for failed queries."""
        client = MockLLMClient(should_fail={'model-1'})
        runner = BatchRunner(client)

        results = runner.query_all_models(
            model_ids=['model-1'],
            prompt="Test"
        )

        result = results['model-1']
        assert result['success'] is False
        assert result['response'] == ""
        assert result['cost'] == 0.0
        assert result['tokens'] == 0
        assert result['error'] is not None


class TestCompareResponses:
    """Test response comparison formatting."""

    def test_compare_markdown_format(self):
        """Test markdown format output."""
        results = {
            'model-1': {
                'model_name': 'Model 1',
                'response': 'This is a test response',
                'cost': 0.01,
                'tokens': 150,
                'time': 1.5,
                'success': True
            }
        }

        runner = BatchRunner(MockLLMClient())
        comparison = runner.compare_responses(results, output_format='markdown')

        assert '# Model Comparison' in comparison
        assert 'Model 1' in comparison
        assert '0.0100' in comparison
        assert 'This is a test response' in comparison

    def test_compare_json_format(self):
        """Test JSON format output."""
        import json

        results = {
            'model-1': {
                'response': 'Test',
                'cost': 0.01,
                'tokens': 100,
                'time': 1.0,
                'success': True
            }
        }

        runner = BatchRunner(MockLLMClient())
        comparison = runner.compare_responses(results, output_format='json')

        # Should be valid JSON
        parsed = json.loads(comparison)
        assert 'model-1' in parsed

    def test_compare_table_format(self):
        """Test table format output."""
        results = {
            'model-1': {
                'model_name': 'Model 1',
                'response': 'Test response',
                'cost': 0.01,
                'tokens': 100,
                'time': 1.0,
                'success': True
            }
        }

        runner = BatchRunner(MockLLMClient())
        comparison = runner.compare_responses(results, output_format='table')

        assert 'Model' in comparison
        assert 'Status' in comparison
        assert 'Tokens' in comparison
        assert 'Cost' in comparison

    def test_compare_multiple_models(self):
        """Test comparison with multiple models."""
        results = {
            'model-1': {
                'model_name': 'Model 1',
                'response': 'Response 1',
                'cost': 0.01,
                'tokens': 100,
                'time': 1.0,
                'success': True
            },
            'model-2': {
                'model_name': 'Model 2',
                'response': 'Response 2',
                'cost': 0.02,
                'tokens': 200,
                'time': 2.0,
                'success': True
            }
        }

        runner = BatchRunner(MockLLMClient())
        comparison = runner.compare_responses(results, output_format='markdown')

        assert 'Model 1' in comparison
        assert 'Model 2' in comparison
        assert 'Response 1' in comparison
        assert 'Response 2' in comparison

    def test_compare_with_errors(self):
        """Test comparison includes error information."""
        results = {
            'model-1': {
                'model_name': 'Model 1',
                'response': '',
                'cost': 0.0,
                'tokens': 0,
                'time': 0.5,
                'error': 'API Error',
                'success': False
            }
        }

        runner = BatchRunner(MockLLMClient())
        comparison = runner.compare_responses(results, output_format='markdown')

        assert 'Error' in comparison or 'error' in comparison or 'API Error' in comparison

    def test_compare_shows_statistics(self):
        """Test that comparison includes overall statistics."""
        results = {
            'model-1': {
                'model_name': 'Model 1',
                'response': 'Test',
                'cost': 0.01,
                'tokens': 100,
                'time': 1.0,
                'success': True
            },
            'model-2': {
                'model_name': 'Model 2',
                'response': 'Test',
                'cost': 0.02,
                'tokens': 150,
                'time': 1.5,
                'success': True
            }
        }

        runner = BatchRunner(MockLLMClient())
        comparison = runner.compare_responses(results, output_format='markdown')

        # Should show total cost
        assert '0.03' in comparison or 'Total Cost' in comparison
        # Should show total tokens
        assert '250' in comparison or 'Total Tokens' in comparison

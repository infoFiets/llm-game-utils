"""Tests for JSON parsing utilities."""

import pytest
from llm_game_utils.prompts.json_parser import (
    extract_json_from_response,
    validate_json_schema,
    extract_and_validate,
    retry_until_valid_json
)
from llm_game_utils.exceptions import InvalidJSONResponseError


class TestExtractJSONFromResponse:
    """Test JSON extraction from various response formats."""

    def test_direct_json(self):
        """Test extracting from direct JSON response."""
        response = '{"name": "Alice", "age": 30}'
        result = extract_json_from_response(response)
        assert result == {"name": "Alice", "age": 30}

    def test_json_with_whitespace(self):
        """Test extracting JSON with extra whitespace."""
        response = '  \n  {"name": "Bob", "score": 100}  \n  '
        result = extract_json_from_response(response)
        assert result == {"name": "Bob", "score": 100}

    def test_json_in_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        response = """Here is the data:
```json
{
  "player": "Charlie",
  "points": 50
}
```
Hope this helps!"""
        result = extract_json_from_response(response)
        assert result == {"player": "Charlie", "points": 50}

    def test_json_in_generic_code_block(self):
        """Test extracting JSON from generic markdown block."""
        response = """The result is:
```
{"status": "success", "value": 42}
```"""
        result = extract_json_from_response(response)
        assert result == {"status": "success", "value": 42}

    def test_json_embedded_in_text(self):
        """Test extracting JSON embedded in explanatory text."""
        response = 'The player data is {"name": "Diana", "level": 5} which shows progress.'
        result = extract_json_from_response(response)
        assert result == {"name": "Diana", "level": 5}

    def test_json_array(self):
        """Test extracting JSON array."""
        response = 'Here are the scores: [10, 20, 30, 40]'
        result = extract_json_from_response(response)
        assert result == [10, 20, 30, 40]

    def test_nested_json(self):
        """Test extracting nested JSON."""
        response = """```json
{
  "player": {
    "name": "Eve",
    "stats": {
      "hp": 100,
      "mp": 50
    }
  }
}
```"""
        result = extract_json_from_response(response)
        assert result == {
            "player": {
                "name": "Eve",
                "stats": {"hp": 100, "mp": 50}
            }
        }

    def test_multiple_json_objects(self):
        """Test extracting first valid JSON when multiple present."""
        response = """
First: {"a": 1}
Second: {"b": 2}
"""
        result = extract_json_from_response(response)
        # Should get the first one
        assert result in [{"a": 1}, {"b": 2}]

    def test_invalid_json(self):
        """Test that invalid JSON returns None."""
        response = "This is just text with no JSON"
        result = extract_json_from_response(response)
        assert result is None

    def test_malformed_json(self):
        """Test that malformed JSON returns None."""
        response = '{"name": "Alice", "age": }'  # Missing value
        result = extract_json_from_response(response)
        assert result is None

    def test_empty_string(self):
        """Test that empty string returns None."""
        result = extract_json_from_response("")
        assert result is None

    def test_none_input(self):
        """Test that None input returns None."""
        result = extract_json_from_response(None)
        assert result is None


class TestValidateJSONSchema:
    """Test JSON schema validation."""

    def test_valid_schema(self):
        """Test validation with all required fields present."""
        data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
        assert validate_json_schema(data, ["name", "age"])

    def test_all_required_fields(self):
        """Test validation when all required fields match exactly."""
        data = {"x": 1, "y": 2}
        assert validate_json_schema(data, ["x", "y"])

    def test_extra_fields(self):
        """Test validation with extra fields is still valid."""
        data = {"required": "yes", "extra": "also here"}
        assert validate_json_schema(data, ["required"])

    def test_missing_field(self):
        """Test validation fails when required field is missing."""
        data = {"name": "Bob"}
        assert not validate_json_schema(data, ["name", "age"])

    def test_field_with_none_value(self):
        """Test that fields with None value are considered present."""
        data = {"name": "Charlie", "age": None}
        assert validate_json_schema(data, ["name", "age"])

    def test_empty_required_fields(self):
        """Test validation with no required fields."""
        data = {"anything": "goes"}
        assert validate_json_schema(data, [])

    def test_non_dict_input(self):
        """Test validation fails for non-dict input."""
        assert not validate_json_schema("not a dict", ["field"])
        assert not validate_json_schema([1, 2, 3], ["field"])
        assert not validate_json_schema(None, ["field"])


class TestExtractAndValidate:
    """Test combined extraction and validation."""

    def test_valid_extraction_and_validation(self):
        """Test successful extraction and validation."""
        response = '{"name": "Alice", "score": 100}'
        result = extract_and_validate(response, ["name", "score"])
        assert result == {"name": "Alice", "score": 100}

    def test_extraction_success_validation_fails(self):
        """Test extraction works but validation fails."""
        response = '{"name": "Bob"}'
        result = extract_and_validate(response, ["name", "score"])
        assert result is None

    def test_extraction_fails(self):
        """Test when extraction fails."""
        response = "No JSON here"
        result = extract_and_validate(response, ["name"])
        assert result is None

    def test_complex_markdown_response(self):
        """Test with complex markdown response."""
        response = """
I've analyzed the data and here's the result:

```json
{
  "player_name": "Eve",
  "resources": {
    "wood": 5,
    "stone": 3
  },
  "action": "build"
}
```

This shows the player has enough resources.
"""
        result = extract_and_validate(response, ["player_name", "resources", "action"])
        assert result is not None
        assert result["player_name"] == "Eve"
        assert result["resources"]["wood"] == 5


class MockClient:
    """Mock client for testing retry_until_valid_json."""

    def __init__(self, responses):
        """Initialize with list of responses to return."""
        self.responses = responses
        self.call_count = 0

    def query(self, **kwargs):
        """Return mock response."""
        if self.call_count < len(self.responses):
            response_text = self.responses[self.call_count]
            self.call_count += 1

            # Return object with response attribute
            class MockResponse:
                def __init__(self, text):
                    self.response = text

            return MockResponse(response_text)
        raise Exception("No more responses")


class TestRetryUntilValidJSON:
    """Test retry logic for getting valid JSON."""

    def test_first_attempt_success(self):
        """Test success on first attempt."""
        client = MockClient(['{"name": "Alice", "age": 30}'])
        result = retry_until_valid_json(
            client=client,
            prompt="Get user data",
            required_fields=["name", "age"],
            model_id="test-model"
        )
        assert result == {"name": "Alice", "age": 30}
        assert client.call_count == 1

    def test_second_attempt_success(self):
        """Test success on second attempt."""
        client = MockClient([
            "This is not JSON",
            '{"name": "Bob", "age": 25}'
        ])
        result = retry_until_valid_json(
            client=client,
            prompt="Get user data",
            required_fields=["name", "age"],
            max_retries=3,
            model_id="test-model"
        )
        assert result == {"name": "Bob", "age": 25}
        assert client.call_count == 2

    def test_third_attempt_success(self):
        """Test success on third attempt."""
        client = MockClient([
            "Not JSON",
            '{"name": "Charlie"}',  # Missing field
            '{"name": "Charlie", "age": 35}'
        ])
        result = retry_until_valid_json(
            client=client,
            prompt="Get user data",
            required_fields=["name", "age"],
            max_retries=3,
            model_id="test-model"
        )
        assert result == {"name": "Charlie", "age": 35}
        assert client.call_count == 3

    def test_max_retries_exceeded(self):
        """Test exception when max retries exceeded."""
        client = MockClient([
            "Not JSON 1",
            "Not JSON 2",
            "Not JSON 3"
        ])
        with pytest.raises(InvalidJSONResponseError) as exc_info:
            retry_until_valid_json(
                client=client,
                prompt="Get user data",
                required_fields=["name", "age"],
                max_retries=3,
                model_id="test-model"
            )

        assert exc_info.value.attempts == 3
        assert "Not JSON 3" in exc_info.value.response

    def test_missing_required_fields(self):
        """Test retry when JSON is valid but missing fields."""
        client = MockClient([
            '{"name": "Diana"}',  # Missing age
            '{"age": 28}',  # Missing name
            '{"name": "Diana", "age": 28}'  # Both present
        ])
        result = retry_until_valid_json(
            client=client,
            prompt="Get user data",
            required_fields=["name", "age"],
            max_retries=3,
            model_id="test-model"
        )
        assert result == {"name": "Diana", "age": 28}

    def test_markdown_wrapped_json(self):
        """Test with markdown-wrapped JSON."""
        client = MockClient([
            """Here's the data:
```json
{
  "player": "Eve",
  "score": 1000
}
```"""
        ])
        result = retry_until_valid_json(
            client=client,
            prompt="Get player data",
            required_fields=["player", "score"],
            model_id="test-model"
        )
        assert result == {"player": "Eve", "score": 1000}

"""Tests for PromptFormatter."""

import json
import pytest

from llm_game_utils.prompts import PromptFormatter


class TestPromptFormatter:
    """Test suite for PromptFormatter."""

    @pytest.fixture
    def formatter(self):
        """Create a formatter instance."""
        return PromptFormatter()

    def test_format_game_state(self, formatter):
        """Test formatting game state."""
        prompt = formatter.format_game_state(
            game_name="TestGame",
            current_state={"score": 10, "level": 2},
            available_actions=["action1", "action2"],
            additional_context="Test context"
        )

        assert "TestGame" in prompt
        assert "Test context" in prompt
        assert "action1" in prompt
        assert "action2" in prompt
        assert "score" in prompt
        assert "10" in prompt

    def test_format_game_state_no_json(self, formatter):
        """Test formatting game state without JSON."""
        prompt = formatter.format_game_state(
            game_name="TestGame",
            current_state={"score": 10},
            available_actions=["action1"],
            include_json=False
        )

        assert "TestGame" in prompt
        assert "score: 10" in prompt

    def test_format_with_template(self, formatter):
        """Test formatting with template."""
        prompt = formatter.format_with_template(
            "Hello {name}, you have {points} points",
            name="Player1",
            points=100
        )

        assert prompt == "Hello Player1, you have 100 points"

    def test_format_player_turn(self, formatter):
        """Test formatting player turn."""
        prompt = formatter.format_player_turn(
            player_name="Player1",
            game_name="Chess",
            turn_number=5,
            game_state="White has advantage",
            instruction="Make your move"
        )

        assert "Player1" in prompt
        assert "Chess" in prompt
        assert "Turn #5" in prompt
        assert "White has advantage" in prompt
        assert "Make your move" in prompt

    def test_format_rules_reminder(self, formatter):
        """Test formatting rules reminder."""
        prompt = formatter.format_rules_reminder(
            game_name="TestGame",
            rules=["Rule 1", "Rule 2"],
            current_situation="You are in a difficult position"
        )

        assert "TestGame" in prompt
        assert "Rule 1" in prompt
        assert "Rule 2" in prompt
        assert "difficult position" in prompt

    def test_format_multi_choice(self, formatter):
        """Test formatting multiple choice."""
        prompt = formatter.format_multi_choice(
            question="What do you want to do?",
            choices=["Option A", "Option B", "Option C"],
            context="You have 3 choices"
        )

        assert "What do you want to do?" in prompt
        assert "You have 3 choices" in prompt
        assert "1. Option A" in prompt
        assert "2. Option B" in prompt
        assert "3. Option C" in prompt

    def test_format_json_response_request(self, formatter):
        """Test formatting JSON response request."""
        schema = {"action": "string", "reasoning": "string"}

        prompt = formatter.format_json_response_request(
            instruction="Choose an action",
            schema=schema,
            context="Current game state is XYZ"
        )

        assert "Choose an action" in prompt
        assert "Current game state is XYZ" in prompt
        assert "action" in prompt
        assert "reasoning" in prompt
        assert "```json" in prompt

    def test_format_conversation_history(self, formatter):
        """Test formatting conversation history."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        prompt = formatter.format_conversation_history(
            messages=messages,
            new_prompt="What's next?"
        )

        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "What's next?" in prompt

    def test_extract_json_from_response_with_markdown(self, formatter):
        """Test extracting JSON from markdown response."""
        response = '''Here's my response:
        ```json
        {"action": "move", "value": 5}
        ```
        That's my choice.'''

        result = formatter.extract_json_from_response(response)

        assert result is not None
        assert result["action"] == "move"
        assert result["value"] == 5

    def test_extract_json_from_plain_json(self, formatter):
        """Test extracting JSON from plain response."""
        response = '{"action": "move", "value": 5}'

        result = formatter.extract_json_from_response(response)

        assert result is not None
        assert result["action"] == "move"
        assert result["value"] == 5

    def test_extract_json_from_invalid_response(self, formatter):
        """Test extracting JSON from invalid response."""
        response = "This is not JSON at all"

        result = formatter.extract_json_from_response(response)

        assert result is None


# Example of how to run tests:
# pytest tests/test_prompt_formatter.py -v

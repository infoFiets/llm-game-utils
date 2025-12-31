"""Tests for GameResultLogger."""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from llm_game_utils.logging import GameResultLogger
from llm_game_utils.clients import LLMResponse


class TestGameResultLogger:
    """Test suite for GameResultLogger."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create a temporary output directory."""
        return tmp_path / "test_logs"

    @pytest.fixture
    def logger(self, temp_output_dir):
        """Create a test logger instance."""
        return GameResultLogger(output_dir=temp_output_dir)

    def test_initialization(self, logger, temp_output_dir):
        """Test logger initialization."""
        assert logger.output_dir == temp_output_dir
        assert temp_output_dir.exists()
        assert isinstance(logger.sessions, dict)

    def test_start_session(self, logger):
        """Test starting a new session."""
        session_id = logger.start_session(
            game_name="TestGame",
            players=["Player1", "Player2"],
            game_config={"setting": "value"}
        )

        assert session_id in logger.sessions
        session = logger.sessions[session_id]
        assert session["game_name"] == "TestGame"
        assert session["players"] == ["Player1", "Player2"]
        assert session["game_config"] == {"setting": "value"}
        assert session["start_time"] is not None
        assert session["end_time"] is None
        assert session["moves"] == []
        assert session["llm_responses"] == []

    def test_start_session_custom_id(self, logger):
        """Test starting a session with custom ID."""
        session_id = logger.start_session(
            game_name="TestGame",
            players=["P1"],
            session_id="custom-session-id"
        )

        assert session_id == "custom-session-id"
        assert "custom-session-id" in logger.sessions

    def test_start_session_with_invalid_filename_chars(self, logger):
        """Test that game names with invalid filename characters are sanitized."""
        # Game name with characters invalid in filenames
        session_id = logger.start_session(
            game_name="Catan: Seafarers / Cities & Knights",
            players=["P1"]
        )

        # Session ID should not contain invalid characters
        assert ":" not in session_id
        assert "/" not in session_id
        assert "\\" not in session_id
        assert session_id in logger.sessions

        # Should be able to save without errors
        logger.save_session(session_id)

    def test_file_encoding_with_unicode(self, logger, temp_output_dir):
        """Test that unicode characters are properly saved and loaded."""
        session_id = logger.start_session(
            game_name="Pokémon Battle",
            players=["Ash 🔥", "Misty 💧"]
        )

        # Save session
        logger.save_session(session_id)

        # Clear and reload
        logger.sessions.clear()
        loaded = logger.load_session(session_id)

        # Verify unicode was preserved
        assert loaded["game_name"] == "Pokémon Battle"
        assert loaded["players"] == ["Ash 🔥", "Misty 💧"]

    def test_log_move(self, logger):
        """Test logging a move."""
        session_id = logger.start_session("TestGame", ["P1"])

        logger.log_move(
            session_id=session_id,
            player="P1",
            move_data={"action": "test"},
            turn_number=1
        )

        session = logger.sessions[session_id]
        assert len(session["moves"]) == 1
        assert session["moves"][0]["player"] == "P1"
        assert session["moves"][0]["move_data"] == {"action": "test"}
        assert session["moves"][0]["turn_number"] == 1

    def test_log_move_invalid_session(self, logger):
        """Test logging move to invalid session."""
        with pytest.raises(KeyError):
            logger.log_move(
                session_id="invalid",
                player="P1",
                move_data={}
            )

    def test_log_llm_response(self, logger):
        """Test logging an LLM response."""
        session_id = logger.start_session("TestGame", ["P1"])

        # Create mock LLM response
        llm_response = LLMResponse(
            model_id="test/model",
            model_name="Test Model",
            prompt="Test prompt",
            response="Test response",
            timestamp=datetime.now(),
            response_time=1.5,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost=0.001,
            metadata={}
        )

        logger.log_llm_response(
            session_id=session_id,
            llm_response=llm_response,
            context={"turn": 1}
        )

        session = logger.sessions[session_id]
        assert len(session["llm_responses"]) == 1
        assert session["llm_responses"][0]["model_id"] == "test/model"
        assert session["llm_responses"][0]["response"] == "Test response"
        assert session["llm_responses"][0]["context"] == {"turn": 1}

    def test_add_metadata(self, logger):
        """Test adding metadata to session."""
        session_id = logger.start_session("TestGame", ["P1"])

        logger.add_metadata(session_id, "custom_key", "custom_value")

        session = logger.sessions[session_id]
        assert session["metadata"]["custom_key"] == "custom_value"

    def test_end_session(self, logger, temp_output_dir):
        """Test ending a session."""
        session_id = logger.start_session("TestGame", ["P1", "P2"])

        result = logger.end_session(
            session_id=session_id,
            winner="P1",
            final_scores={"P1": 10, "P2": 8}
        )

        assert result["end_time"] is not None
        assert result["winner"] == "P1"
        assert result["final_scores"] == {"P1": 10, "P2": 8}

        # Check that file was saved
        saved_file = temp_output_dir / f"{session_id}.json"
        assert saved_file.exists()

    def test_save_and_load_session(self, logger, temp_output_dir):
        """Test saving and loading a session."""
        session_id = logger.start_session("TestGame", ["P1"])
        logger.log_move(session_id, "P1", {"action": "test"})

        # Save session
        saved_path = logger.save_session(session_id)
        assert saved_path.exists()

        # Clear sessions
        logger.sessions.clear()

        # Load session
        loaded_session = logger.load_session(session_id)
        assert loaded_session["game_name"] == "TestGame"
        assert len(loaded_session["moves"]) == 1

    def test_get_session_summary(self, logger):
        """Test getting session summary."""
        session_id = logger.start_session("TestGame", ["P1"])

        # Add some data
        logger.log_move(session_id, "P1", {"action": "test"})

        llm_response = LLMResponse(
            model_id="test/model",
            model_name="Test",
            prompt="Test",
            response="Test",
            timestamp=datetime.now(),
            response_time=1.0,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost=0.001,
            metadata={}
        )
        logger.log_llm_response(session_id, llm_response)

        summary = logger.get_session_summary(session_id)

        assert summary["session_id"] == session_id
        assert summary["game_name"] == "TestGame"
        assert summary["total_moves"] == 1
        assert summary["total_llm_calls"] == 1
        assert summary["total_cost"] == 0.001
        assert summary["total_tokens"] == 30


# Example of how to run tests:
# pytest tests/test_result_logger.py -v

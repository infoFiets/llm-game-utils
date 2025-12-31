"""Result logging utilities for game sessions and LLM interactions."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..clients.base_client import LLMResponse


logger = logging.getLogger(__name__)


class GameResultLogger:
    """Logger for tracking game results and LLM interactions.

    This class provides utilities for logging game sessions, player moves,
    LLM responses, and final results to structured JSON files.

    Example:
        ```python
        logger = GameResultLogger(output_dir="game_logs")
        session_id = logger.start_session(game_name="Catan", players=["GPT-4", "Claude"])

        # Log a move
        logger.log_move(
            session_id=session_id,
            player="GPT-4",
            move_data={"action": "build_road", "location": "edge_12"}
        )

        # Log an LLM response
        logger.log_llm_response(session_id, llm_response)

        # End session
        logger.end_session(session_id, winner="GPT-4", final_scores={"GPT-4": 10, "Claude": 8})
        ```
    """

    def __init__(self, output_dir: Union[str, Path] = "game_logs"):
        """Initialize the result logger.

        Args:
            output_dir: Directory to save log files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # In-memory session storage
        self.sessions: Dict[str, Dict[str, Any]] = {}

        logger.info(f"GameResultLogger initialized with output_dir: {self.output_dir}")

    def start_session(
        self,
        game_name: str,
        players: List[str],
        game_config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Start a new game session.

        Args:
            game_name: Name of the game (e.g., "Catan", "Cards Against Humanity")
            players: List of player names/identifiers
            game_config: Optional game configuration parameters
            session_id: Optional custom session ID, auto-generated if None

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"{game_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.sessions[session_id] = {
            "session_id": session_id,
            "game_name": game_name,
            "players": players,
            "game_config": game_config or {},
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "moves": [],
            "llm_responses": [],
            "winner": None,
            "final_scores": {},
            "metadata": {}
        }

        logger.info(f"Started session {session_id} for {game_name} with players: {players}")
        return session_id

    def log_move(
        self,
        session_id: str,
        player: str,
        move_data: Dict[str, Any],
        turn_number: Optional[int] = None
    ) -> None:
        """Log a player move.

        Args:
            session_id: Session identifier
            player: Player who made the move
            move_data: Dictionary containing move details
            turn_number: Optional turn number

        Raises:
            KeyError: If session_id not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")

        move_entry = {
            "timestamp": datetime.now().isoformat(),
            "player": player,
            "turn_number": turn_number,
            "move_data": move_data
        }

        self.sessions[session_id]["moves"].append(move_entry)
        logger.debug(f"Logged move for {player} in session {session_id}")

    def log_llm_response(
        self,
        session_id: str,
        llm_response: LLMResponse,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an LLM response.

        Args:
            session_id: Session identifier
            llm_response: LLMResponse object from client
            context: Optional context information (game state, turn number, etc.)

        Raises:
            KeyError: If session_id not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")

        response_entry = {
            "timestamp": llm_response.timestamp.isoformat(),
            "model_id": llm_response.model_id,
            "model_name": llm_response.model_name,
            "prompt": llm_response.prompt,
            "response": llm_response.response,
            "response_time": llm_response.response_time,
            "input_tokens": llm_response.input_tokens,
            "output_tokens": llm_response.output_tokens,
            "total_tokens": llm_response.total_tokens,
            "cost": llm_response.cost,
            "context": context or {}
        }

        self.sessions[session_id]["llm_responses"].append(response_entry)
        logger.debug(f"Logged LLM response from {llm_response.model_name} in session {session_id}")

    def add_metadata(
        self,
        session_id: str,
        key: str,
        value: Any
    ) -> None:
        """Add metadata to a session.

        Args:
            session_id: Session identifier
            key: Metadata key
            value: Metadata value

        Raises:
            KeyError: If session_id not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")

        self.sessions[session_id]["metadata"][key] = value
        logger.debug(f"Added metadata {key} to session {session_id}")

    def end_session(
        self,
        session_id: str,
        winner: Optional[str] = None,
        final_scores: Optional[Dict[str, Any]] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """End a game session.

        Args:
            session_id: Session identifier
            winner: Optional winner identifier
            final_scores: Optional dictionary of final scores
            save: Whether to save the session to disk

        Returns:
            Complete session data

        Raises:
            KeyError: If session_id not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")

        self.sessions[session_id]["end_time"] = datetime.now().isoformat()
        self.sessions[session_id]["winner"] = winner
        self.sessions[session_id]["final_scores"] = final_scores or {}

        session_data = self.sessions[session_id]

        if save:
            self.save_session(session_id)

        logger.info(f"Ended session {session_id}. Winner: {winner}")
        return session_data

    def save_session(self, session_id: str) -> Path:
        """Save session data to a JSON file.

        Args:
            session_id: Session identifier

        Returns:
            Path to saved file

        Raises:
            KeyError: If session_id not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")

        output_file = self.output_dir / f"{session_id}.json"

        with open(output_file, 'w') as f:
            json.dump(self.sessions[session_id], f, indent=2)

        logger.info(f"Saved session {session_id} to {output_file}")
        return output_file

    def load_session(self, session_id: str) -> Dict[str, Any]:
        """Load session data from a JSON file.

        Args:
            session_id: Session identifier

        Returns:
            Session data dictionary

        Raises:
            FileNotFoundError: If session file not found
        """
        input_file = self.output_dir / f"{session_id}.json"

        if not input_file.exists():
            raise FileNotFoundError(f"Session file not found: {input_file}")

        with open(input_file, 'r') as f:
            session_data = json.load(f)

        self.sessions[session_id] = session_data
        logger.info(f"Loaded session {session_id} from {input_file}")
        return session_data

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of a session.

        Args:
            session_id: Session identifier

        Returns:
            Summary dictionary with key metrics

        Raises:
            KeyError: If session_id not found
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        # Calculate total costs and tokens
        total_cost = sum(r["cost"] for r in session["llm_responses"])
        total_tokens = sum(r["total_tokens"] for r in session["llm_responses"])
        total_response_time = sum(r["response_time"] for r in session["llm_responses"])

        return {
            "session_id": session_id,
            "game_name": session["game_name"],
            "players": session["players"],
            "winner": session["winner"],
            "start_time": session["start_time"],
            "end_time": session["end_time"],
            "total_moves": len(session["moves"]),
            "total_llm_calls": len(session["llm_responses"]),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "total_response_time": total_response_time,
            "final_scores": session["final_scores"]
        }

    def list_sessions(self) -> List[str]:
        """List all session files in the output directory.

        Returns:
            List of session IDs
        """
        session_files = list(self.output_dir.glob("*.json"))
        return [f.stem for f in session_files]

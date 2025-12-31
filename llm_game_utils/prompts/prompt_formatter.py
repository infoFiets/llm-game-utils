"""Prompt formatting utilities for game contexts."""

from typing import Any, Dict, List, Optional


class PromptFormatter:
    """Utility class for formatting prompts for LLM game interactions.

    This class provides helpers for constructing clear, structured prompts
    that include game state, rules, and action options.

    Example:
        ```python
        formatter = PromptFormatter()

        # Format a game state prompt
        prompt = formatter.format_game_state(
            game_name="Catan",
            current_state={"resources": {"wood": 2, "brick": 1}},
            available_actions=["build_road", "build_settlement", "end_turn"],
            additional_context="It's your turn. You rolled a 7."
        )

        # Format with custom template
        custom_prompt = formatter.format_with_template(
            template="Game: {game}\\nState: {state}\\nWhat do you do?",
            game="Chess",
            state="You're in check"
        )
        ```
    """

    @staticmethod
    def format_game_state(
        game_name: str,
        current_state: Dict[str, Any],
        available_actions: List[str],
        additional_context: Optional[str] = None,
        include_json: bool = True
    ) -> str:
        """Format a game state into a clear prompt.

        Args:
            game_name: Name of the game
            current_state: Dictionary representing current game state
            available_actions: List of available action names
            additional_context: Optional additional context or instructions
            include_json: Whether to include JSON-formatted state

        Returns:
            Formatted prompt string
        """
        prompt_parts = [f"Game: {game_name}"]

        if additional_context:
            prompt_parts.append(f"\n{additional_context}")

        prompt_parts.append("\nCurrent Game State:")
        if include_json:
            import json
            prompt_parts.append(json.dumps(current_state, indent=2))
        else:
            for key, value in current_state.items():
                prompt_parts.append(f"  {key}: {value}")

        prompt_parts.append("\nAvailable Actions:")
        for i, action in enumerate(available_actions, 1):
            prompt_parts.append(f"  {i}. {action}")

        prompt_parts.append("\nWhat action do you choose? Please respond with your choice and reasoning.")

        return "\n".join(prompt_parts)

    @staticmethod
    def format_with_template(template: str, **kwargs) -> str:
        """Format a prompt using a template string.

        Args:
            template: Template string with {placeholders}
            **kwargs: Values to fill in placeholders

        Returns:
            Formatted prompt string
        """
        return template.format(**kwargs)

    @staticmethod
    def format_player_turn(
        player_name: str,
        game_name: str,
        turn_number: int,
        game_state: str,
        instruction: str
    ) -> str:
        """Format a player turn prompt.

        Args:
            player_name: Name of the player
            game_name: Name of the game
            turn_number: Current turn number
            game_state: Description of current game state
            instruction: Specific instruction for this turn

        Returns:
            Formatted prompt string
        """
        return f"""You are {player_name} playing {game_name}.

Turn #{turn_number}

{game_state}

{instruction}

Please provide your response in a clear, structured format."""

    @staticmethod
    def format_rules_reminder(
        game_name: str,
        rules: List[str],
        current_situation: str
    ) -> str:
        """Format a rules reminder prompt.

        Args:
            game_name: Name of the game
            rules: List of relevant rules
            current_situation: Description of current situation

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Game: {game_name}",
            "\nRelevant Rules:"
        ]

        for i, rule in enumerate(rules, 1):
            prompt_parts.append(f"  {i}. {rule}")

        prompt_parts.append(f"\nCurrent Situation:\n{current_situation}")
        prompt_parts.append("\nHow do you proceed according to the rules?")

        return "\n".join(prompt_parts)

    @staticmethod
    def format_multi_choice(
        question: str,
        choices: List[str],
        context: Optional[str] = None
    ) -> str:
        """Format a multiple choice question prompt.

        Args:
            question: The question to ask
            choices: List of choice options
            context: Optional context for the question

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        if context:
            prompt_parts.append(context)
            prompt_parts.append("")

        prompt_parts.append(question)
        prompt_parts.append("")

        for i, choice in enumerate(choices, 1):
            prompt_parts.append(f"{i}. {choice}")

        prompt_parts.append("\nPlease select your choice by number and explain your reasoning.")

        return "\n".join(prompt_parts)

    @staticmethod
    def format_json_response_request(
        instruction: str,
        schema: Dict[str, Any],
        context: Optional[str] = None
    ) -> str:
        """Format a prompt requesting JSON response.

        Args:
            instruction: What to do
            schema: Expected JSON schema/example
            context: Optional context

        Returns:
            Formatted prompt string
        """
        import json

        prompt_parts = []

        if context:
            prompt_parts.append(context)
            prompt_parts.append("")

        prompt_parts.append(instruction)
        prompt_parts.append("\nPlease respond in the following JSON format:")
        prompt_parts.append("```json")
        prompt_parts.append(json.dumps(schema, indent=2))
        prompt_parts.append("```")

        return "\n".join(prompt_parts)

    @staticmethod
    def format_conversation_history(
        messages: List[Dict[str, str]],
        new_prompt: str
    ) -> str:
        """Format conversation history with a new prompt.

        Args:
            messages: List of message dicts with 'role' and 'content'
            new_prompt: New prompt to add

        Returns:
            Formatted prompt string with conversation history
        """
        prompt_parts = ["Conversation History:"]

        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            prompt_parts.append(f"\n{role}: {content}")

        prompt_parts.append(f"\n\nCurrent Prompt:\n{new_prompt}")

        return "\n".join(prompt_parts)

    @staticmethod
    def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from a response that might contain markdown or extra text.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        import json
        import re

        # Try to find JSON in markdown code blocks
        json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse the entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find any JSON object in the response
        json_obj_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        match = re.search(json_obj_pattern, response)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return None

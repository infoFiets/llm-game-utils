"""JSON parsing utilities for LLM responses.

LLMs often return JSON wrapped in markdown code blocks or embedded in
explanatory text. This module provides robust extraction and validation
functions to handle various response formats.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..exceptions import InvalidJSONResponseError

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from an LLM response even if wrapped in markdown/text.

    Tries multiple extraction strategies to find valid JSON:
    1. Direct JSON parsing (response is pure JSON)
    2. Markdown code blocks (```json {...}```)
    3. Generic code blocks (```{...}```)
    4. Curly brace extraction (finds first {...} or [...])
    5. Multi-line patterns

    Args:
        response: The LLM response text that may contain JSON

    Returns:
        Parsed dict if valid JSON found, None otherwise

    Example:
        ```python
        # Handles various formats:
        extract_json_from_response('{"key": "value"}')
        extract_json_from_response('```json\\n{"key": "value"}\\n```')
        extract_json_from_response('Here is the data: {"key": "value"}')
        ```
    """
    if not response or not isinstance(response, str):
        return None

    response = response.strip()

    # Strategy 1: Try parsing the response directly as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Try extracting from markdown code blocks with json language tag
    json_code_block_pattern = r'```json\s*\n(.*?)\n```'
    matches = re.findall(json_code_block_pattern, response, re.DOTALL | re.IGNORECASE)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 3: Try extracting from generic markdown code blocks
    code_block_pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Strategy 4: Try finding JSON object or array with curly braces or brackets
    # Look for the first occurrence of { or [
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start_idx = response.find(start_char)
        if start_idx == -1:
            continue

        # Find matching closing brace/bracket
        depth = 0
        for i in range(start_idx, len(response)):
            if response[i] == start_char:
                depth += 1
            elif response[i] == end_char:
                depth -= 1
                if depth == 0:
                    potential_json = response[start_idx:i+1]
                    try:
                        return json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Try to continue searching
                        break

    # Strategy 5: Try extracting everything between first { and last }
    if '{' in response and '}' in response:
        start = response.find('{')
        end = response.rfind('}')
        if start < end:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                pass

    # If all strategies fail, return None
    logger.debug(f"Could not extract JSON from response: {response[:100]}...")
    return None


def validate_json_schema(
    data: Dict[str, Any],
    required_fields: List[str]
) -> bool:
    """Validate that JSON has all required fields.

    Args:
        data: Parsed JSON dictionary
        required_fields: List of field names that must exist in data

    Returns:
        True if all required fields are present (even if their values are None)

    Example:
        ```python
        data = {"name": "Alice", "age": 30}
        validate_json_schema(data, ["name", "age"])  # True
        validate_json_schema(data, ["name", "email"])  # False
        ```
    """
    if not isinstance(data, dict):
        return False

    for field in required_fields:
        if field not in data:
            logger.debug(f"Missing required field: {field}")
            return False

    return True


def extract_and_validate(
    response: str,
    required_fields: List[str]
) -> Optional[Dict[str, Any]]:
    """Extract JSON from response and validate it has required fields.

    Convenience function that combines extraction and validation.

    Args:
        response: LLM response text
        required_fields: List of field names that must be present

    Returns:
        Parsed and validated dict, or None if extraction/validation fails

    Example:
        ```python
        response = 'Here is the player data: {"name": "Alice", "score": 100}'
        data = extract_and_validate(response, ["name", "score"])
        if data:
            print(f"{data['name']} scored {data['score']}")
        ```
    """
    data = extract_json_from_response(response)

    if data is None:
        return None

    if not validate_json_schema(data, required_fields):
        return None

    return data


def retry_until_valid_json(
    client,
    prompt: str,
    required_fields: List[str],
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """Keep querying LLM until it returns valid JSON with required fields.

    This function will automatically retry if the LLM returns invalid JSON
    or JSON missing required fields. It appends instructions to the prompt
    on subsequent attempts.

    Args:
        client: OpenRouterClient instance (or any client with a query method)
        prompt: Base prompt to send to the LLM
        required_fields: List of fields that must be in the JSON response
        max_retries: Maximum number of attempts (default: 3)
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        **kwargs: Additional parameters passed to client.query()

    Returns:
        Valid JSON dict with all required fields, or None if max retries exceeded

    Raises:
        InvalidJSONResponseError: If valid JSON cannot be obtained after max retries

    Example:
        ```python
        client = OpenRouterClient()
        client.add_model_config("openai/gpt-4-turbo", "GPT-4", 0.01, 0.03)

        result = retry_until_valid_json(
            client=client,
            prompt="List the player's resources in Catan",
            required_fields=["wood", "brick", "wheat", "sheep", "ore"],
            max_retries=3,
            model_id="openai/gpt-4-turbo"
        )

        if result:
            print(f"Resources: {result}")
        ```
    """
    # Ensure the prompt asks for JSON
    json_instruction = (
        f"\n\nYou must respond with ONLY valid JSON containing these fields: "
        f"{', '.join(required_fields)}. "
        f"Do not include any explanatory text before or after the JSON."
    )

    # Add JSON instruction if not already present
    if "json" not in prompt.lower():
        prompt = prompt + json_instruction

    last_response = None

    for attempt in range(1, max_retries + 1):
        try:
            # Query the LLM
            response_obj = client.query(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )

            # Extract the response text
            # Handle both LLMResponse objects and dict responses
            if hasattr(response_obj, 'response'):
                response_text = response_obj.response
            elif isinstance(response_obj, dict):
                response_text = response_obj.get('response', str(response_obj))
            else:
                response_text = str(response_obj)

            last_response = response_text

            # Try to extract and validate JSON
            result = extract_and_validate(response_text, required_fields)

            if result is not None:
                logger.info(f"Valid JSON obtained on attempt {attempt}/{max_retries}")
                return result

            # If we failed, modify the prompt for the next attempt
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt}/{max_retries} failed to return valid JSON. "
                    f"Retrying..."
                )
                prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: Your previous response was not valid JSON or was "
                    f"missing required fields. Please respond with ONLY a JSON object "
                    f"containing exactly these fields: {', '.join(required_fields)}"
                )

        except Exception as e:
            logger.error(f"Error during attempt {attempt}/{max_retries}: {str(e)}")
            if attempt == max_retries:
                raise InvalidJSONResponseError(
                    response=str(last_response) if last_response else "",
                    attempts=attempt
                )

    # If we've exhausted all retries, raise an exception
    raise InvalidJSONResponseError(
        response=last_response if last_response else "",
        attempts=max_retries
    )

"""Custom exceptions for llm-game-utils.

This module defines specific exception types for better error handling
throughout the package. Using specific exceptions allows callers to
handle different error scenarios appropriately.
"""


class LLMGameUtilsError(Exception):
    """Base exception for all llm-game-utils errors.

    All custom exceptions in this package inherit from this base class,
    making it easy to catch any package-specific error.
    """
    pass


class APIError(LLMGameUtilsError):
    """Base exception for all API-related errors.

    Raised when there's a problem communicating with an LLM API provider.
    """
    pass


class InsufficientCreditsError(APIError):
    """Raised when the API account has insufficient credits.

    This typically means you need to add credits to your OpenRouter or
    other provider account before continuing.

    Example:
        ```python
        try:
            response = client.query(model_id, prompt)
        except InsufficientCreditsError:
            print("Please add credits to your account")
        ```
    """
    pass


class RateLimitError(APIError):
    """Raised when hitting API rate limits.

    This occurs when you've made too many requests in a short time period.
    The built-in rate limiter should prevent this, but it can still happen
    with concurrent requests or when using multiple clients.

    Example:
        ```python
        try:
            response = client.query(model_id, prompt)
        except RateLimitError:
            time.sleep(60)  # Wait a bit
            response = client.query(model_id, prompt)
        ```
    """
    pass


class InvalidModelError(APIError):
    """Raised when the specified model ID doesn't exist or isn't available.

    Check the provider's model list to ensure you're using a valid model ID.

    Attributes:
        model_id: The invalid model ID that was requested
    """

    def __init__(self, model_id: str, message: str = None):
        """Initialize with the invalid model ID.

        Args:
            model_id: The model ID that was not found
            message: Optional custom error message
        """
        self.model_id = model_id
        if message is None:
            message = f"Model '{model_id}' is not available or doesn't exist"
        super().__init__(message)


class BudgetExceededError(LLMGameUtilsError):
    """Raised when a budget limit is exceeded.

    This prevents runaway API costs by enforcing daily or session budget limits.

    Attributes:
        budget_type: Either 'daily' or 'session'
        limit: The budget limit that was exceeded
        current: The current spending that triggered the error

    Example:
        ```python
        try:
            response = client.query(model_id, prompt)
        except BudgetExceededError as e:
            print(f"{e.budget_type} budget of ${e.limit} exceeded")
            print(f"Current spending: ${e.current}")
        ```
    """

    def __init__(self, budget_type: str, limit: float, current: float):
        """Initialize with budget information.

        Args:
            budget_type: Type of budget exceeded ('daily' or 'session')
            limit: The budget limit in dollars
            current: Current spending in dollars
        """
        self.budget_type = budget_type
        self.limit = limit
        self.current = current
        message = (
            f"{budget_type.capitalize()} budget limit exceeded: "
            f"${current:.4f} >= ${limit:.4f}"
        )
        super().__init__(message)


class InvalidJSONResponseError(LLMGameUtilsError):
    """Raised when an LLM fails to return valid JSON after retries.

    This happens when prompting an LLM to return JSON but it consistently
    returns malformed JSON or non-JSON text.

    Attributes:
        response: The last response that failed to parse
        attempts: Number of retry attempts made

    Example:
        ```python
        try:
            result = retry_until_valid_json(
                client, prompt, ["field1", "field2"]
            )
        except InvalidJSONResponseError as e:
            print(f"Failed after {e.attempts} attempts")
            print(f"Last response: {e.response}")
        ```
    """

    def __init__(self, response: str, attempts: int):
        """Initialize with the failed response and attempt count.

        Args:
            response: The last response text that failed to parse
            attempts: Number of attempts made
        """
        self.response = response
        self.attempts = attempts
        message = (
            f"Failed to get valid JSON after {attempts} attempts. "
            f"Last response: {response[:100]}..."
        )
        super().__init__(message)


class CacheError(LLMGameUtilsError):
    """Raised when cache operations fail.

    This can occur due to disk I/O errors, permission issues, or
    corrupted cache files.

    Example:
        ```python
        try:
            cache.set(model_id, prompt, temperature, response)
        except CacheError:
            # Continue without caching
            pass
        ```
    """
    pass

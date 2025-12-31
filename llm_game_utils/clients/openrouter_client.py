"""OpenRouter API client with rate limiting and error handling."""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .base_client import BaseLLMClient, LLMResponse


logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_minute: int = 60):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_interval:
            sleep_time = self.min_interval - time_since_last_request
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


class OpenRouterClient(BaseLLMClient):
    """Client for interacting with OpenRouter API.

    OpenRouter provides unified access to multiple LLM providers through
    a single API. This client handles authentication, rate limiting,
    retries, and response parsing.

    Example:
        ```python
        client = OpenRouterClient(api_key="your-key")
        response = client.query(
            model_id="openai/gpt-4-turbo",
            prompt="What is the capital of France?",
            temperature=0.7
        )
        print(response.response)
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        rate_limit: int = 60,
        timeout: int = 120,
        app_name: str = "llm-game-utils",
        site_url: str = "https://github.com/infoFiets/llm-game-utils",
        model_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL
            rate_limit: Maximum requests per minute
            timeout: Request timeout in seconds
            app_name: Application name for OpenRouter headers
            site_url: Site URL for OpenRouter headers
            model_configs: Optional dict mapping model IDs to config dicts with
                          'name' and 'pricing' (input/output costs per 1K tokens)

        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        if api_key is None:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY not provided and not found in environment. "
                    "Either pass api_key parameter or set OPENROUTER_API_KEY env var."
                )

        self.api_key = api_key
        self.base_url = base_url
        self.app_name = app_name
        self.site_url = site_url

        # Set up HTTP client
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
            }
        )

        # Set up rate limiter
        self.rate_limiter = RateLimiter(rate_limit)

        # Model configurations (name and pricing)
        self.model_configs = model_configs or {}

        logger.info(f"OpenRouter client initialized for {app_name}")

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()

    def add_model_config(
        self,
        model_id: str,
        name: str,
        input_cost: float,
        output_cost: float
    ) -> None:
        """Add or update a model configuration.

        Args:
            model_id: Model identifier (e.g., "openai/gpt-4-turbo")
            name: Human-readable model name
            input_cost: Cost per 1K input tokens in dollars
            output_cost: Cost per 1K output tokens in dollars
        """
        self.model_configs[model_id] = {
            "name": name,
            "pricing": {
                "input": input_cost,
                "output": output_cost
            }
        }

    def calculate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """Calculate cost for a model based on token usage.

        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in dollars, or 0.0 if pricing not configured
        """
        if model_id not in self.model_configs:
            return 0.0

        pricing = self.model_configs[model_id]["pricing"]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        reraise=True
    )
    def _make_request(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a request to OpenRouter API with retries.

        Args:
            model_id: Model identifier
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            API response dictionary

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        # Wait for rate limiting
        self.rate_limiter.wait_if_needed()

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.debug(f"Making request to {model_id}")

        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error for {model_id}: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.TimeoutException as e:
            logger.error(f"Timeout error for {model_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {model_id}: {str(e)}")
            raise

    def query(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Query a model with a prompt.

        Args:
            model_id: Model identifier (e.g., "openai/gpt-4-turbo")
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (e.g., top_p, frequency_penalty)

        Returns:
            LLMResponse object with results and metadata

        Raises:
            httpx.HTTPError: If API request fails
        """
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Track timing
        start_time = time.time()

        # Make request
        response_data = self._make_request(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        end_time = time.time()
        response_time = end_time - start_time

        # Extract response content
        response_text = response_data["choices"][0]["message"]["content"]

        # Extract token usage
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        # Calculate cost
        cost = self.calculate_cost(model_id, input_tokens, output_tokens)

        # Get model name
        model_name = self.model_configs.get(model_id, {}).get("name", model_id)

        logger.info(
            f"Query completed: {model_name} | "
            f"{response_time:.2f}s | "
            f"{total_tokens} tokens | "
            f"${cost:.4f}"
        )

        return LLMResponse(
            model_id=model_id,
            model_name=model_name,
            prompt=prompt,
            response=response_text,
            timestamp=datetime.now(),
            response_time=response_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "raw_response": response_data,
                **kwargs
            }
        )

    def batch_query(
        self,
        model_ids: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Query multiple models with the same prompt.

        Args:
            model_ids: List of model identifiers
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            List of LLMResponse objects
        """
        responses = []

        for model_id in model_ids:
            try:
                response = self.query(
                    model_id=model_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                responses.append(response)

            except Exception as e:
                logger.error(f"Failed to query {model_id}: {str(e)}")
                # Continue with other models even if one fails
                continue

        return responses

    def get_available_models(self) -> List[str]:
        """Get list of available model IDs from configured models.

        Returns:
            List of model IDs
        """
        return list(self.model_configs.keys())

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model information or None if not found
        """
        if model_id not in self.model_configs:
            return None

        config = self.model_configs[model_id]
        return {
            "id": model_id,
            "name": config.get("name", model_id),
            "pricing": config.get("pricing", {})
        }

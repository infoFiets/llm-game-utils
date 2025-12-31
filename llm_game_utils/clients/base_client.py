"""Base client interface for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM model.

    Attributes:
        model_id: Unique identifier for the model
        model_name: Human-readable model name
        prompt: The user prompt sent to the model
        response: The generated text response
        timestamp: When the response was generated
        response_time: Time taken to generate response (seconds)
        input_tokens: Number of input tokens processed
        output_tokens: Number of output tokens generated
        total_tokens: Total tokens used (input + output)
        cost: Estimated cost in dollars
        metadata: Additional metadata (temperature, system_prompt, etc.)
    """
    model_id: str
    model_name: str
    prompt: str
    response: str
    timestamp: datetime
    response_time: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    metadata: Dict[str, Any]


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients.

    Implement this class to create clients for different LLM providers
    (OpenRouter, OpenAI, Anthropic, etc.).
    """

    @abstractmethod
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
            model_id: Model identifier
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse object with results and metadata

        Raises:
            ValueError: If model not found or invalid parameters
            Exception: If API request fails
        """
        pass

    @abstractmethod
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
            **kwargs: Additional provider-specific parameters

        Returns:
            List of LLMResponse objects
        """
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs.

        Returns:
            List of model identifiers
        """
        pass

    @abstractmethod
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary with model information or None if not found
        """
        pass

"""Batch query multiple models for comparison and testing.

This module provides the BatchRunner class to query multiple LLM models
with the same prompt, useful for comparing responses, testing different
models, or running A/B tests.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any

from .base_client import LLMResponse

logger = logging.getLogger(__name__)


class BatchRunner:
    """Query multiple models with the same prompt for comparison.

    Useful for:
    - Comparing model responses
    - Testing which model works best for a task
    - Running the same prompt across multiple models
    - A/B testing different model configurations

    Example:
        ```python
        client = OpenRouterClient()
        runner = BatchRunner(client)

        # Query multiple models
        results = runner.query_all_models(
            model_ids=[
                "openai/gpt-4-turbo",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-pro"
            ],
            prompt="What is the capital of France?",
            parallel=False  # Sequential to avoid rate limits
        )

        # Compare responses
        comparison = runner.compare_responses(results, output_format="markdown")
        print(comparison)
        ```
    """

    def __init__(self, openrouter_client):
        """Initialize BatchRunner with an OpenRouterClient.

        Args:
            openrouter_client: An instance of OpenRouterClient (or compatible client)
        """
        self.client = openrouter_client
        logger.debug("BatchRunner initialized")

    def query_all_models(
        self,
        model_ids: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        parallel: bool = False,
        max_workers: int = 3,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Query multiple models with the same prompt.

        Args:
            model_ids: List of model identifiers to query
            prompt: The prompt to send to all models
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            parallel: If True, use threading for parallel execution (faster but
                     more API load). If False, query sequentially.
            max_workers: Maximum number of parallel workers (only used if parallel=True)
            **kwargs: Additional parameters passed to client.query()

        Returns:
            Dictionary mapping model_id to results:
            {
                "model_id": {
                    "response": str,          # The generated text
                    "cost": float,            # Cost in dollars
                    "tokens": int,            # Total tokens used
                    "input_tokens": int,      # Input tokens
                    "output_tokens": int,     # Output tokens
                    "time": float,            # Response time in seconds
                    "error": Optional[str],   # Error message if failed
                    "success": bool           # Whether query succeeded
                },
                ...
            }

        Example:
            ```python
            results = runner.query_all_models(
                model_ids=["openai/gpt-4-turbo", "anthropic/claude-3.5-sonnet"],
                prompt="Explain quantum computing",
                temperature=0.5,
                parallel=False
            )

            for model_id, result in results.items():
                if result["success"]:
                    print(f"{model_id}: {result['response'][:100]}...")
                else:
                    print(f"{model_id} failed: {result['error']}")
            ```
        """
        results = {}

        if parallel:
            logger.info(f"Querying {len(model_ids)} models in parallel (max {max_workers} workers)")
            results = self._query_parallel(
                model_ids=model_ids,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_workers=max_workers,
                **kwargs
            )
        else:
            logger.info(f"Querying {len(model_ids)} models sequentially")
            results = self._query_sequential(
                model_ids=model_ids,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

        return results

    def _query_sequential(
        self,
        model_ids: List[str],
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Query models one at a time (safer for rate limits)."""
        results = {}

        for i, model_id in enumerate(model_ids, 1):
            logger.info(f"Querying model {i}/{len(model_ids)}: {model_id}")
            result = self._query_single_model(
                model_id=model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            results[model_id] = result

        return results

    def _query_parallel(
        self,
        model_ids: List[str],
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        max_workers: int,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """Query models in parallel using threading."""
        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_model = {
                executor.submit(
                    self._query_single_model,
                    model_id=model_id,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ): model_id
                for model_id in model_ids
            }

            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    results[model_id] = result
                except Exception as e:
                    logger.error(f"Parallel query failed for {model_id}: {e}")
                    results[model_id] = {
                        "response": "",
                        "cost": 0.0,
                        "tokens": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "time": 0.0,
                        "model_name": model_id,
                        "error": str(e),
                        "success": False
                    }

        return results

    def _query_single_model(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> Dict[str, Any]:
        """Query a single model and return formatted results."""
        start_time = time.time()

        try:
            response: LLMResponse = self.client.query(
                model_id=model_id,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            elapsed_time = time.time() - start_time

            return {
                "response": response.response,
                "cost": response.cost,
                "tokens": response.total_tokens,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "time": elapsed_time,
                "model_name": response.model_name,
                "error": None,
                "success": True
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Query failed for {model_id}: {str(e)}")

            return {
                "response": "",
                "cost": 0.0,
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "time": elapsed_time,
                "model_name": model_id,
                "error": str(e),
                "success": False
            }

    def compare_responses(
        self,
        results: Dict[str, Dict[str, Any]],
        output_format: str = "markdown"
    ) -> str:
        """Generate a formatted comparison of model responses.

        Args:
            results: Output from query_all_models()
            output_format: Output format - "markdown", "json", or "table"

        Returns:
            Formatted comparison string

        Example:
            ```python
            results = runner.query_all_models(...)
            comparison = runner.compare_responses(results, "markdown")
            print(comparison)
            ```
        """
        if output_format == "json":
            import json
            return json.dumps(results, indent=2)

        elif output_format == "table":
            return self._format_as_table(results)

        else:  # markdown (default)
            return self._format_as_markdown(results)

    def _format_as_markdown(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Format results as markdown."""
        lines = ["# Model Comparison\n"]

        # Summary table
        lines.append("## Summary\n")
        lines.append("| Model | Success | Tokens | Cost | Time |")
        lines.append("|-------|---------|--------|------|------|")

        for model_id, result in results.items():
            model_name = result.get("model_name", model_id)
            success = "✓" if result["success"] else "✗"
            tokens = result["tokens"]
            cost = f"${result['cost']:.4f}"
            time_str = f"{result['time']:.2f}s"
            lines.append(f"| {model_name} | {success} | {tokens} | {cost} | {time_str} |")

        # Detailed responses
        lines.append("\n## Detailed Responses\n")

        for model_id, result in results.items():
            model_name = result.get("model_name", model_id)
            lines.append(f"### {model_name}\n")

            if result["success"]:
                lines.append(f"**Cost:** ${result['cost']:.4f} | "
                           f"**Tokens:** {result['tokens']} | "
                           f"**Time:** {result['time']:.2f}s\n")
                lines.append("**Response:**\n")
                lines.append(f"```\n{result['response']}\n```\n")
            else:
                lines.append(f"**Error:** {result['error']}\n")

        # Overall stats
        successful = [r for r in results.values() if r["success"]]
        if successful:
            total_cost = sum(r["cost"] for r in successful)
            total_tokens = sum(r["tokens"] for r in successful)
            avg_time = sum(r["time"] for r in successful) / len(successful)

            lines.append("## Overall Statistics\n")
            lines.append(f"- **Total Cost:** ${total_cost:.4f}")
            lines.append(f"- **Total Tokens:** {total_tokens}")
            lines.append(f"- **Average Response Time:** {avg_time:.2f}s")
            lines.append(f"- **Success Rate:** {len(successful)}/{len(results)}")

        return "\n".join(lines)

    def _format_as_table(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Format results as a simple text table."""
        # Calculate column widths
        max_model_len = max(len(result.get("model_name", mid)) for mid, result in results.items())
        max_model_len = max(max_model_len, len("Model"))

        lines = []
        header = f"{'Model':<{max_model_len}} | Status | Tokens | Cost      | Time    | Response"
        lines.append(header)
        lines.append("-" * len(header))

        for model_id, result in results.items():
            model_name = result.get("model_name", model_id)
            status = "OK" if result["success"] else "FAIL"
            tokens = f"{result['tokens']:>6}"
            cost = f"${result['cost']:>7.4f}"
            time_str = f"{result['time']:>6.2f}s"

            if result["success"]:
                response_preview = result['response'][:50].replace('\n', ' ')
                if len(result['response']) > 50:
                    response_preview += "..."
            else:
                response_preview = f"Error: {result['error'][:40]}"

            line = f"{model_name:<{max_model_len}} | {status:>6} | {tokens} | {cost} | {time_str} | {response_preview}"
            lines.append(line)

        # Summary
        successful = [r for r in results.values() if r["success"]]
        if successful:
            total_cost = sum(r["cost"] for r in successful)
            total_tokens = sum(r["tokens"] for r in successful)

            lines.append("")
            lines.append(f"Total Cost: ${total_cost:.4f} | Total Tokens: {total_tokens} | "
                        f"Success: {len(successful)}/{len(results)}")

        return "\n".join(lines)

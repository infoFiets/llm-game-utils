"""Advanced examples demonstrating v0.2.0 features.

This example demonstrates:
- Budget tracking
- Response caching
- JSON response parsing
- Batch model comparison
- Enhanced error handling

To run this example:
1. Create a .env file with your OPENROUTER_API_KEY
2. Run: python examples_advanced.py
"""

from llm_game_utils import (
    OpenRouterClient,
    BudgetTracker,
    ResponseCache,
    BatchRunner
)
from llm_game_utils.prompts import json_parser
from llm_game_utils.exceptions import (
    BudgetExceededError,
    InvalidJSONResponseError
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def example_budget_tracking():
    """Example 1: Budget Tracking"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Budget Tracking")
    print("="*60)

    # Create a budget tracker with low limits for demonstration
    budget = BudgetTracker(
        daily_budget=5.0,
        session_budget=0.50,
        warning_threshold=0.7
    )

    # Initialize client with budget tracking
    client = OpenRouterClient(budget_tracker=budget)
    client.add_model_config(
        model_id="openai/gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        input_cost=0.0005,
        output_cost=0.0015
    )

    print("\nInitial budget status:")
    print(budget.get_status_report())

    try:
        # Make a query - budget is automatically checked
        print("\nMaking first query...")
        response = client.query(
            model_id="openai/gpt-3.5-turbo",
            prompt="Say 'Hello, world!' in 3 words or less.",
            temperature=0.5,
            max_tokens=20
        )
        print(f"Response: {response.response}")
        print(f"Cost: ${response.cost:.6f}")

        print("\nBudget after first query:")
        print(budget.get_status_report())

        # Try to exceed budget with many queries
        print("\nAttempting to exceed session budget...")
        for i in range(50):  # Intentionally try many queries
            response = client.query(
                model_id="openai/gpt-3.5-turbo",
                prompt=f"Count to {i}",
                max_tokens=50
            )

    except BudgetExceededError as e:
        print(f"\n✓ Budget protection worked! {e}")
        print(f"  Budget type: {e.budget_type}")
        print(f"  Limit: ${e.limit:.2f}")
        print(f"  Would have spent: ${e.current:.4f}")

    print("\nFinal budget status:")
    print(budget.get_status_report())


def example_response_caching():
    """Example 2: Response Caching"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Response Caching")
    print("="*60)

    # Create a cache
    cache = ResponseCache(
        cache_dir=".example_cache",
        ttl_hours=1  # Short TTL for demonstration
    )

    # Initialize client with caching
    client = OpenRouterClient(cache=cache)
    client.add_model_config(
        model_id="openai/gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        input_cost=0.0005,
        output_cost=0.0015
    )

    prompt = "What is 2 + 2?"

    print("\nFirst call (will hit API)...")
    response1 = client.query(
        model_id="openai/gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.7
    )
    print(f"Response: {response1.response}")
    print(f"Cost: ${response1.cost:.6f}")

    print("\nSecond call (will use cache)...")
    response2 = client.query(
        model_id="openai/gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.7
    )
    print(f"Response: {response2.response}")
    print(f"Cost: ${response2.cost:.6f} (from cache, no API cost!)")

    print("\nThird call (bypass cache)...")
    response3 = client.query(
        model_id="openai/gpt-3.5-turbo",
        prompt=prompt,
        temperature=0.7,
        use_cache=False  # Force fresh API call
    )
    print(f"Response: {response3.response}")
    print(f"Cost: ${response3.cost:.6f}")

    # Show cache statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Cache size: {stats['cache_size_mb']:.2f} MB")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")

    # Clean up
    print(f"\nClearing cache... deleted {cache.clear()} entries")


def example_json_parsing():
    """Example 3: JSON Response Parsing"""
    print("\n" + "="*60)
    print("EXAMPLE 3: JSON Response Parsing")
    print("="*60)

    client = OpenRouterClient()
    client.add_model_config(
        model_id="openai/gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        input_cost=0.0005,
        output_cost=0.0015
    )

    # Example 1: Extract JSON from various formats
    print("\nExtracting JSON from different response formats:")

    responses = [
        '{"score": 100}',
        '```json\n{"score": 200}\n```',
        'The result is {"score": 300} which is good.',
    ]

    for i, resp in enumerate(responses, 1):
        data = json_parser.extract_json_from_response(resp)
        print(f"  Format {i}: {resp[:30]:30s} -> {data}")

    # Example 2: Retry until valid JSON
    print("\nRetrying until LLM returns valid JSON...")

    try:
        result = json_parser.retry_until_valid_json(
            client=client,
            prompt="Return a JSON object with fields 'player' (string) and 'score' (number) for a game.",
            required_fields=["player", "score"],
            max_retries=3,
            model_id="openai/gpt-3.5-turbo",
            temperature=0.5
        )

        print(f"✓ Got valid JSON: {result}")
        print(f"  Player: {result['player']}")
        print(f"  Score: {result['score']}")

    except InvalidJSONResponseError as e:
        print(f"✗ Failed to get valid JSON after {e.attempts} attempts")
        print(f"  Last response: {e.response[:100]}...")


def example_batch_comparison():
    """Example 4: Batch Model Comparison"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Model Comparison")
    print("="*60)

    client = OpenRouterClient()

    # Add multiple models
    models = [
        ("openai/gpt-3.5-turbo", "GPT-3.5 Turbo", 0.0005, 0.0015),
        ("anthropic/claude-3-haiku", "Claude 3 Haiku", 0.00025, 0.00125),
    ]

    for model_id, name, input_cost, output_cost in models:
        client.add_model_config(model_id, name, input_cost, output_cost)

    # Create batch runner
    runner = BatchRunner(client)

    print("\nQuerying multiple models with the same prompt...")
    prompt = "Explain quantum entanglement in one sentence."

    results = runner.query_all_models(
        model_ids=[m[0] for m in models],
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        parallel=False  # Sequential to avoid rate limits
    )

    # Show individual results
    print("\nResults:")
    for model_id, result in results.items():
        print(f"\n  {result.get('model_name', model_id)}:")
        if result['success']:
            print(f"    Response: {result['response'][:80]}...")
            print(f"    Cost: ${result['cost']:.6f}")
            print(f"    Time: {result['time']:.2f}s")
            print(f"    Tokens: {result['tokens']}")
        else:
            print(f"    Error: {result['error']}")

    # Generate comparison
    print("\n" + "-"*60)
    print("Formatted Comparison:")
    print("-"*60)
    comparison = runner.compare_responses(results, output_format="table")
    print(comparison)


def example_error_handling():
    """Example 5: Enhanced Error Handling"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Enhanced Error Handling")
    print("="*60)

    from llm_game_utils.exceptions import InvalidModelError

    client = OpenRouterClient()

    # Try an invalid model
    print("\nAttempting to use invalid model...")
    try:
        client.query(
            model_id="invalid/model-that-doesnt-exist",
            prompt="Hello"
        )
    except InvalidModelError as e:
        print(f"✓ Caught InvalidModelError: {e}")
        print(f"  Model ID: {e.model_id}")

    # Budget exceeded (already demonstrated)
    print("\nBudget exceeded handling (see Example 1)")

    # JSON parsing error (already demonstrated)
    print("JSON parsing error handling (see Example 3)")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("LLM Game Utils v0.2.0 - Advanced Features Examples")
    print("="*60)
    print("\nThese examples demonstrate the new features in v0.2.0:")
    print("  1. Budget Tracking")
    print("  2. Response Caching")
    print("  3. JSON Response Parsing")
    print("  4. Batch Model Comparison")
    print("  5. Enhanced Error Handling")

    try:
        example_budget_tracking()
        example_response_caching()
        example_json_parsing()
        example_batch_comparison()
        example_error_handling()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure OPENROUTER_API_KEY is set in your .env file")


if __name__ == "__main__":
    main()

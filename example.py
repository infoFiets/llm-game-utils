"""Example usage of llm-game-utils package.

This example demonstrates how to use the OpenRouterClient, GameResultLogger,
and PromptFormatter together in a simple game context.

To run this example:
1. Create a .env file with your OPENROUTER_API_KEY
2. Run: python example.py
"""

from llm_game_utils import OpenRouterClient, GameResultLogger, PromptFormatter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Run example game using llm-game-utils."""
    print("=== LLM Game Utils Example ===\n")

    # Initialize components
    print("1. Initializing OpenRouter client...")
    client = OpenRouterClient(
        app_name="Example Game",
        site_url="https://github.com/infoFiets/llm-game-utils"
    )

    # Configure models
    print("2. Configuring models...")
    client.add_model_config(
        model_id="openai/gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        input_cost=0.0005,
        output_cost=0.0015
    )

    client.add_model_config(
        model_id="anthropic/claude-3-haiku",
        name="Claude 3 Haiku",
        input_cost=0.00025,
        output_cost=0.00125
    )

    # Initialize logger and formatter
    print("3. Initializing logger and formatter...")
    logger = GameResultLogger(output_dir="example_logs")
    formatter = PromptFormatter()

    # Start a game session
    print("4. Starting game session...")
    session_id = logger.start_session(
        game_name="Rock Paper Scissors",
        players=["GPT-3.5", "Claude"],
        game_config={"rounds": 3}
    )
    print(f"   Session ID: {session_id}\n")

    # Play a round
    print("5. Playing Round 1...")

    # Player 1 (GPT-3.5) turn
    print("   - GPT-3.5's turn...")
    game_state = {
        "round": 1,
        "score": {"GPT-3.5": 0, "Claude": 0}
    }

    prompt = formatter.format_game_state(
        game_name="Rock Paper Scissors",
        current_state=game_state,
        available_actions=["rock", "paper", "scissors"],
        additional_context="Choose your move for round 1."
    )

    # Get LLM response
    try:
        llm_response = client.query(
            model_id="openai/gpt-3.5-turbo",
            prompt=prompt,
            system_prompt="You are playing Rock Paper Scissors. Respond with just your choice: rock, paper, or scissors.",
            temperature=0.8,
            max_tokens=50
        )

        print(f"     Response: {llm_response.response}")
        print(f"     Tokens: {llm_response.total_tokens}")
        print(f"     Cost: ${llm_response.cost:.6f}")
        print(f"     Time: {llm_response.response_time:.2f}s")

        # Log the response
        logger.log_llm_response(
            session_id=session_id,
            llm_response=llm_response,
            context={"round": 1, "player": "GPT-3.5"}
        )

        # Log the move
        logger.log_move(
            session_id=session_id,
            player="GPT-3.5",
            move_data={"choice": "rock"},  # In real game, parse from response
            turn_number=1
        )

    except Exception as e:
        print(f"     Error: {e}")
        print("     (Make sure OPENROUTER_API_KEY is set in .env)")

    # End the session
    print("\n6. Ending session...")
    logger.end_session(
        session_id=session_id,
        winner="GPT-3.5",
        final_scores={"GPT-3.5": 2, "Claude": 1}
    )

    # Get summary
    print("\n7. Session Summary:")
    summary = logger.get_session_summary(session_id)
    print(f"   Game: {summary['game_name']}")
    print(f"   Players: {summary['players']}")
    print(f"   Winner: {summary['winner']}")
    print(f"   Total Moves: {summary['total_moves']}")
    print(f"   Total LLM Calls: {summary['total_llm_calls']}")
    print(f"   Total Cost: ${summary['total_cost']:.6f}")
    print(f"   Total Tokens: {summary['total_tokens']}")
    print(f"   Final Scores: {summary['final_scores']}")

    print(f"\n8. Session saved to: example_logs/{session_id}.json")
    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()

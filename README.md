# LLM Game Utils

A lightweight Python package providing shared utilities for building LLM-powered game projects. This package extracts reusable components for LLM API interaction, result logging, and prompt formatting that can be used across multiple game projects.

## Features

- **LLM Client Infrastructure**: Ready-to-use OpenRouter client with rate limiting, retry logic, and error handling
- **Result Logging**: Track game sessions, player moves, and LLM interactions
- **Prompt Formatting**: Utilities for creating structured, game-context prompts
- **Extensible Architecture**: Abstract base classes for implementing custom LLM clients
- **Minimal Dependencies**: Only essential dependencies for maximum compatibility

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/infoFiets/llm-game-utils.git
```

Or clone and install locally:

```bash
git clone https://github.com/infoFiets/llm-game-utils.git
cd llm-game-utils
pip install -e .
```

## Quick Start

### 1. Set up your API key

Create a `.env` file in your project:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

### 2. Basic usage

```python
from llm_game_utils import OpenRouterClient, GameResultLogger, PromptFormatter

# Initialize the client
client = OpenRouterClient()

# Add model configurations with pricing
client.add_model_config(
    model_id="openai/gpt-4-turbo",
    name="GPT-4 Turbo",
    input_cost=0.01,   # per 1K tokens
    output_cost=0.03   # per 1K tokens
)

# Query a model
response = client.query(
    model_id="openai/gpt-4-turbo",
    prompt="You are playing Catan. You have 2 wood and 1 brick. What should you build?",
    system_prompt="You are an expert Catan player.",
    temperature=0.7
)

print(f"Response: {response.response}")
print(f"Cost: ${response.cost:.4f}")
print(f"Tokens: {response.total_tokens}")
```

## Complete Game Example

Here's a complete example of using llm-game-utils in a game context:

```python
from llm_game_utils import OpenRouterClient, GameResultLogger, PromptFormatter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components
client = OpenRouterClient(
    app_name="Catan AI Game",
    site_url="https://github.com/infoFiets/catan-ai"
)

# Configure models
client.add_model_config(
    model_id="openai/gpt-4-turbo",
    name="GPT-4 Turbo",
    input_cost=0.01,
    output_cost=0.03
)

client.add_model_config(
    model_id="anthropic/claude-3.5-sonnet",
    name="Claude 3.5 Sonnet",
    input_cost=0.003,
    output_cost=0.015
)

# Initialize logger
logger = GameResultLogger(output_dir="game_logs")
formatter = PromptFormatter()

# Start a game session
session_id = logger.start_session(
    game_name="Catan",
    players=["GPT-4", "Claude"],
    game_config={"max_turns": 100, "victory_points": 10}
)

# Simulate a game turn
game_state = {
    "player": "GPT-4",
    "resources": {"wood": 2, "brick": 1, "wheat": 0},
    "victory_points": 3
}

# Format prompt using PromptFormatter
prompt = formatter.format_game_state(
    game_name="Catan",
    current_state=game_state,
    available_actions=["build_road", "build_settlement", "trade", "end_turn"],
    additional_context="You just rolled a 6. You gained 2 wood."
)

# Get LLM response
llm_response = client.query(
    model_id="openai/gpt-4-turbo",
    prompt=prompt,
    system_prompt="You are an expert Catan player. Always explain your reasoning.",
    temperature=0.7
)

# Log the LLM response
logger.log_llm_response(
    session_id=session_id,
    llm_response=llm_response,
    context={"turn": 1, "game_phase": "main"}
)

# Parse and execute the move
# ... (your game logic here)

# Log the move
logger.log_move(
    session_id=session_id,
    player="GPT-4",
    move_data={"action": "build_road", "location": "edge_12"},
    turn_number=1
)

# End the session
final_data = logger.end_session(
    session_id=session_id,
    winner="GPT-4",
    final_scores={"GPT-4": 10, "Claude": 8}
)

# Get summary
summary = logger.get_session_summary(session_id)
print(f"Game completed! Winner: {summary['winner']}")
print(f"Total cost: ${summary['total_cost']:.4f}")
print(f"Total LLM calls: {summary['total_llm_calls']}")
```

## API Documentation

### OpenRouterClient

The main client for interacting with OpenRouter API.

#### Initialization

```python
client = OpenRouterClient(
    api_key=None,              # Optional, reads from OPENROUTER_API_KEY env var
    base_url="https://openrouter.ai/api/v1",
    rate_limit=60,             # requests per minute
    timeout=120,               # seconds
    app_name="llm-game-utils",
    site_url="https://github.com/infoFiets/llm-game-utils"
)
```

#### Key Methods

##### `query(model_id, prompt, system_prompt=None, temperature=0.7, max_tokens=None, **kwargs)`

Query a single model.

**Parameters:**
- `model_id` (str): Model identifier (e.g., "openai/gpt-4-turbo")
- `prompt` (str): User prompt
- `system_prompt` (str, optional): System prompt
- `temperature` (float): Sampling temperature (0.0-1.0)
- `max_tokens` (int, optional): Maximum tokens to generate
- `**kwargs`: Additional parameters (top_p, frequency_penalty, etc.)

**Returns:** `LLMResponse` object

##### `batch_query(model_ids, prompt, system_prompt=None, temperature=0.7, max_tokens=None, **kwargs)`

Query multiple models with the same prompt.

**Parameters:**
- `model_ids` (List[str]): List of model identifiers
- Other parameters same as `query()`

**Returns:** `List[LLMResponse]`

##### `add_model_config(model_id, name, input_cost, output_cost)`

Add or update model configuration with pricing.

**Parameters:**
- `model_id` (str): Model identifier
- `name` (str): Human-readable name
- `input_cost` (float): Cost per 1K input tokens
- `output_cost` (float): Cost per 1K output tokens

### GameResultLogger

Logger for tracking game sessions and LLM interactions.

#### Initialization

```python
logger = GameResultLogger(output_dir="game_logs")
```

#### Key Methods

##### `start_session(game_name, players, game_config=None, session_id=None)`

Start a new game session.

**Returns:** `str` - Session ID

##### `log_move(session_id, player, move_data, turn_number=None)`

Log a player move.

##### `log_llm_response(session_id, llm_response, context=None)`

Log an LLM response.

##### `end_session(session_id, winner=None, final_scores=None, save=True)`

End a game session.

**Returns:** `Dict[str, Any]` - Complete session data

##### `get_session_summary(session_id)`

Get summary statistics for a session.

**Returns:** `Dict[str, Any]` - Summary with costs, tokens, moves, etc.

### PromptFormatter

Utilities for formatting game prompts.

#### Key Methods

##### `format_game_state(game_name, current_state, available_actions, additional_context=None, include_json=True)`

Format a structured game state prompt.

##### `format_player_turn(player_name, game_name, turn_number, game_state, instruction)`

Format a player turn prompt.

##### `format_multi_choice(question, choices, context=None)`

Format a multiple choice question.

##### `format_json_response_request(instruction, schema, context=None)`

Format a prompt requesting JSON response.

##### `extract_json_from_response(response)`

Extract JSON from an LLM response (handles markdown code blocks).

**Returns:** `Dict[str, Any]` or `None`

## Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Configuration

You can configure models in two ways:

1. **Programmatically** (recommended for games):

```python
client.add_model_config(
    model_id="openai/gpt-4-turbo",
    name="GPT-4 Turbo",
    input_cost=0.01,
    output_cost=0.03
)
```

2. **Via constructor**:

```python
model_configs = {
    "openai/gpt-4-turbo": {
        "name": "GPT-4 Turbo",
        "pricing": {"input": 0.01, "output": 0.03}
    }
}

client = OpenRouterClient(model_configs=model_configs)
```

## Common Model IDs

Here are some popular OpenRouter model IDs:

- `openai/gpt-4-turbo` - GPT-4 Turbo
- `openai/gpt-3.5-turbo` - GPT-3.5 Turbo
- `anthropic/claude-3.5-sonnet` - Claude 3.5 Sonnet
- `anthropic/claude-3-haiku` - Claude 3 Haiku
- `google/gemini-pro` - Gemini Pro
- `meta-llama/llama-3-70b-instruct` - Llama 3 70B
- `mistralai/mistral-large-latest` - Mistral Large

See [OpenRouter's model list](https://openrouter.ai/models) for current pricing and availability.

## Architecture

### LLMResponse Dataclass

All LLM queries return an `LLMResponse` object with:

```python
@dataclass
class LLMResponse:
    model_id: str              # Model identifier
    model_name: str            # Human-readable name
    prompt: str                # Original prompt
    response: str              # Generated text
    timestamp: datetime        # When generated
    response_time: float       # Time in seconds
    input_tokens: int          # Input tokens used
    output_tokens: int         # Output tokens generated
    total_tokens: int          # Total tokens
    cost: float                # Estimated cost in dollars
    metadata: Dict[str, Any]   # Additional data
```

### Extending with Custom Clients

Implement the `BaseLLMClient` abstract class to create clients for other providers:

```python
from llm_game_utils.clients import BaseLLMClient, LLMResponse

class MyCustomClient(BaseLLMClient):
    def query(self, model_id, prompt, system_prompt=None, temperature=0.7, max_tokens=None, **kwargs):
        # Your implementation
        pass

    def batch_query(self, model_ids, prompt, system_prompt=None, temperature=0.7, max_tokens=None, **kwargs):
        # Your implementation
        pass

    def get_available_models(self):
        # Your implementation
        pass

    def get_model_info(self, model_id):
        # Your implementation
        pass
```

## Dependencies

- `httpx` - Modern HTTP client
- `tenacity` - Retry logic
- `python-dotenv` - Environment variable management

## Development

### Installing for Development

```bash
git clone https://github.com/infoFiets/llm-game-utils.git
cd llm-game-utils
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black llm_game_utils/
isort llm_game_utils/
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [llm-benchmark-hub](https://github.com/infoFiets/llm-benchmark-hub) - LLM benchmarking platform (parent project)
- Your game projects using this library!

## Support

For bugs and feature requests, please open an issue on [GitHub](https://github.com/infoFiets/llm-game-utils/issues).

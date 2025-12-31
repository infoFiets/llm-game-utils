# API Key Setup Guide

## Overview

This document explains how to manage your OpenRouter API key when using `llm-game-utils`.

## Important: Where Does the API Key Go?

### ✅ DO: Put API key in your GAME projects

The API key should be stored in the **projects that USE this library**, NOT in this library itself.

**Example structure:**
```
your-catan-game/          ← Put .env file HERE
├── .env                  ← Contains OPENROUTER_API_KEY=...
├── .gitignore            ← Should include .env
├── game.py
├── requirements.txt      ← Includes llm-game-utils
└── ...

llm-game-utils/           ← This library (NO API key here!)
├── llm_game_utils/
├── tests/
├── .env.example          ← Just a template
└── ...
```

### ❌ DON'T: Put API key in this repository

- **Never** commit `.env` files with real API keys
- **Never** put API keys in code
- **Never** put API keys in GitHub Secrets for this library repo

## Step-by-Step Setup

### 1. Get Your API Key

1. Go to [OpenRouter](https://openrouter.ai/keys)
2. Sign up or log in
3. Create a new API key
4. Copy the key (you'll only see it once!)

### 2. Set Up Your Game Project

When you create a new game project that uses `llm-game-utils`:

**Create a `.env` file in your game project:**
```bash
# In your game project directory (NOT in llm-game-utils!)
cd your-game-project/
touch .env
```

**Add your API key to the `.env` file:**
```bash
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
```

**Update `.gitignore` to exclude `.env`:**
```gitignore
# Environment variables
.env
.env.local
.env.*.local
```

### 3. Use in Your Code

The `llm-game-utils` library will automatically read the API key from your `.env` file:

```python
from llm_game_utils import OpenRouterClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Client will automatically use OPENROUTER_API_KEY from .env
client = OpenRouterClient()

# Or explicitly pass the key
client = OpenRouterClient(api_key="your-key")
```

## GitHub Secrets (For CI/CD)

If you're running tests or deployments in GitHub Actions for your **game project**, you can use GitHub Secrets:

### Setting up GitHub Secrets (in your game repo, not this one)

1. Go to your game repository on GitHub
2. Settings → Secrets and variables → Actions
3. New repository secret:
   - Name: `OPENROUTER_API_KEY`
   - Value: your API key

### Using in GitHub Actions

```yaml
# .github/workflows/test.yml (in your game project)
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install git+https://github.com/infoFiets/llm-game-utils.git
          pip install -r requirements.txt

      - name: Run tests
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: pytest
```

## Security Best Practices

### ✅ DO:
- Use `.env` files for local development
- Add `.env` to `.gitignore`
- Use GitHub Secrets for CI/CD
- Rotate API keys periodically
- Use separate API keys for development and production
- Monitor your API usage on OpenRouter

### ❌ DON'T:
- Commit API keys to Git
- Share API keys in Slack/Discord/Email
- Hardcode API keys in source code
- Use the same API key across many projects
- Put API keys in public repositories

## Multiple Projects

If you have multiple game projects using `llm-game-utils`:

```
projects/
├── catan-ai/
│   ├── .env              ← API key for Catan project
│   └── ...
├── cards-against-humanity-ai/
│   ├── .env              ← API key for CAH project
│   └── ...
└── chess-ai/
    ├── .env              ← API key for Chess project
    └── ...
```

Each project has its own `.env` file. You can use the same API key or different ones.

## Troubleshooting

### Error: "OPENROUTER_API_KEY not provided and not found in environment"

**Solution:**
1. Make sure you have a `.env` file in your project
2. Make sure it contains `OPENROUTER_API_KEY=your-key`
3. Make sure you call `load_dotenv()` before creating the client

### API key not being loaded

**Check:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
print(os.getenv("OPENROUTER_API_KEY"))  # Should print your key
```

If it prints `None`, your `.env` file isn't being loaded correctly.

## Example Project Structure

Here's a complete example of a game project using `llm-game-utils`:

```
my-awesome-game/
├── .env                    # OPENROUTER_API_KEY=...
├── .gitignore              # Includes .env
├── requirements.txt        # Includes llm-game-utils
├── game.py                 # Your game logic
├── README.md
└── tests/
    └── test_game.py

# requirements.txt
llm-game-utils @ git+https://github.com/infoFiets/llm-game-utils.git
python-dotenv>=1.0.0
# ... other dependencies

# .gitignore
.env
.env.local
.env.*.local
__pycache__/
*.pyc
game_logs/
```

## Questions?

If you have questions about API key management, please open an issue on the [llm-game-utils repository](https://github.com/infoFiets/llm-game-utils/issues).

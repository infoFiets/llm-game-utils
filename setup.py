"""Setup configuration for llm-game-utils."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="llm-game-utils",
    version="0.1.0",
    description="Shared utilities for LLM game projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tom",
    author_email="",
    url="https://github.com/infoFiets/llm-game-utils",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "httpx>=0.25.0",
        "tenacity>=8.2.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.12.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Games/Entertainment :: Board Games",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="llm game ai openrouter gpt claude",
    project_urls={
        "Bug Reports": "https://github.com/infoFiets/llm-game-utils/issues",
        "Source": "https://github.com/infoFiets/llm-game-utils",
    },
)

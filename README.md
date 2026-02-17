# Agent demo

A small **demo** of an LLM agent with tool use: single Python file, OpenRouter API, and a few tools (time, math, book meeting). Built to show the basics of function calling and the agent loop in one place.

## What it does

- **Chat loop** with an assistant (via OpenRouter).
- **Tools**: get current time, evaluate math expressions, “book” a meeting (name + time).
- **Optional TTS** on macOS using `say` (toggle with `TTS_ENABLED` in `agent.py`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set your OpenRouter API key:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Get a key at [openrouter.ai](https://openrouter.ai).

## Run

```bash
python agent.py
```

Then type messages; Jimmy can use the tools (e.g. “What time is it?”, “What’s 2 + 3 * 4?”, “Book a meeting with Alice tomorrow at 3pm”) and you’ll see tool calls and results in the terminal.

## Project layout

- `agent.py` — agent loop, tool definitions, and tool runner (all in one file for the demo).
- `requirements.txt` — `openai` (used against OpenRouter’s OpenAI-compatible API).

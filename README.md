# Agent demo

A small **demo** of an LLM agent, shown in four steps. Each file is a complete chat loop. Open the next file when you are ready to add the next piece. Uses the OpenAI API.

## Steps

1. **`01_chat.py`** — chat loop only. No tools.
2. **`02_math.py`** — one tool: evaluate a math expression.
3. **`03_tools.py`** — time, math, and book a meeting.
4. **`04_tts.py`** — the same tools, plus macOS text-to-speech (`say`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Put your OpenAI API key in `.env`:

```
OPENAI_API_KEY=your_key_here
```

Get a key at [platform.openai.com](https://platform.openai.com/api-keys). Each script loads that file on startup.

## Run

```bash
python 01_chat.py
python 02_math.py
python 03_tools.py
python 04_tts.py
```

Try “What’s 2 + 3 * 4?” once math is in. With the full tool set, try “What time is it?” and “Book a meeting with Alice”. `04_tts.py` speaks the reply on macOS. Toggle that with `TTS_ENABLED`.

## Project layout

- `01_chat.py` through `04_tts.py` — one runnable stage each.
- `requirements.txt` — `openai` and `python-dotenv`.
- `.env` — `OPENAI_API_KEY` (not committed).

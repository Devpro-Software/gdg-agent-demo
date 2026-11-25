# Agent demo

A small **demo** of an LLM agent, shown in five steps. Each file is a complete chat loop. Open the next file when you are ready to add the next piece. Uses the OpenAI API.

## Steps

1. **`01_chat.py`** — chat loop only. No tools.
2. **`02_math.py`** — one tool: evaluate a math expression.
3. **`03_tools.py`** — time, math, and book a meeting.
4. **`04_tts.py`** — the same tools, plus macOS text-to-speech (`say`, System Voice / Siri).
5. **`05_stt.py`** — the same agent, with local push-to-talk speech-to-text (`mlx-whisper`) in front of the loop.

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

### Extra setup for `05_stt.py` (local STT)

Step 5 needs an **Apple Silicon Mac** (`mlx-whisper` uses MLX). Install these with Homebrew **before** `pip install`:

```bash
brew install portaudio ffmpeg
```

- **PortAudio** — required by `sounddevice` for mic capture. Without it, the pip install of `sounddevice` can fail or recording will not work.
- **ffmpeg** — required by `mlx-whisper` to decode audio. Without it, transcription fails after you record.

Also:

- Allow **microphone access** for Terminal (or iTerm / your IDE) the first time you record. macOS System Settings → Privacy & Security → Microphone.
- The first run downloads the Whisper model (`mlx-community/whisper-tiny`) from Hugging Face into the local cache. Before a talk, run `python 05_stt.py` once, press Enter, say a short phrase, then quit — so the download is not on stage.

Steps 01–04 do not need PortAudio, ffmpeg, or a mic.

## Run

```bash
python 01_chat.py
python 02_math.py
python 03_tools.py
python 04_tts.py
python 05_stt.py
```

Try “What’s 2 + 3 * 4?” once math is in. With the full tool set, try “What time is it?” and “Book a meeting with Alice”. `04_tts.py` speaks the reply on macOS. Toggle that with `TTS_ENABLED`.

In `05_stt.py`, press **Enter** to start recording, speak, then **Enter** again to stop and transcribe. Use **`t`** then Enter to type instead, or **`q`** to quit. The transcript is printed as `You: …` before the agent runs.

## Project layout

- `01_chat.py` through `05_stt.py` — one runnable stage each.
- `requirements.txt` — `openai`, `python-dotenv`, and (for step 5) `mlx-whisper`, `sounddevice`, `numpy`.
- `.env` — `OPENAI_API_KEY` (not committed).

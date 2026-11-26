# Agent demo

Super Secret OpenAI API Key: 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

A small **demo** of an LLM agent, shown in six steps. Each file is a complete chat loop. Open the next file when you are ready to add the next piece. Uses the OpenAI API (and a local MLX model in step 6).

## Steps

1. **`01_chat.py`** — chat loop only. No tools.
2. **`02_math.py`** — one tool: evaluate a math expression.
3. **`03_tools.py`** — time, math, and book a meeting.
4. **`04_tts.py`** — the same tools, plus macOS text-to-speech (`say`, System Voice / Siri).
5. **`05_pipeline.py`** — the same agent, with local push-to-talk speech-to-text (`mlx-whisper`) in front of the loop.
6. **`06_local.py`** — fully local: push-to-talk STT, on-device LLM (`mlx-lm`), and macOS TTS. No OpenAI key.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Put your OpenAI API key in `.env` (needed for steps 01–05):

```
OPENAI_API_KEY=your_key_here
```

Get a key at [platform.openai.com](https://platform.openai.com/api-keys). Each cloud script loads that file on startup.

### Extra setup for `05_pipeline.py` (local STT)

Step 5 needs an **Apple Silicon Mac** (`mlx-whisper` uses MLX). Install these with Homebrew **before** `pip install`:

```bash
brew install portaudio ffmpeg
```

- **PortAudio** — required by `sounddevice` for mic capture. Without it, the pip install of `sounddevice` can fail or recording will not work.
- **ffmpeg** — required by `mlx-whisper` to decode audio. Without it, transcription fails after you record.

Also:

- Allow **microphone access** for Terminal (or iTerm / your IDE) the first time you record. macOS System Settings → Privacy & Security → Microphone.
- The first run downloads the Whisper model (`mlx-community/whisper-tiny`) from Hugging Face into the local cache. Before a talk, run `python 05_pipeline.py` once, press Enter, say a short phrase, then quit — so the download is not on stage.

Steps 01–04 do not need PortAudio, ffmpeg, or a mic.

### Extra setup for `06_local.py` (local LLM)

Step 6 needs an **Apple Silicon Mac** and runs the model on the GPU via MLX. No OpenAI key. `06_local.py` starts `mlx_lm.server` itself (and stops it on exit). If a server is already on port 8080, it reuses that.

```bash
python 06_local.py
```

Same mic / PortAudio / ffmpeg needs as step 5. The first run downloads the LLM (~0.5–1 GB) and may download Whisper if you have not run step 5 yet. Warm that before a talk. Server logs are silenced; you only see the chat. Push-to-talk works like `05_pipeline.py` (`Enter` / `t` / `q`). The model is small on purpose — fast on an M4 with 16 GB, clearly weaker than gpt-4o-mini.

**Note:** bare `say` with Siri System Voice can stay silent inside **tmux**. Run TTS demos outside tmux, or use a classic voice with `say -v`.

## Run

```bash
python 01_chat.py
python 02_math.py
python 03_tools.py
python 04_tts.py
python 05_pipeline.py
python 06_local.py
```

Try “What’s 2 + 3 * 4?” once math is in. With the full tool set, try “What time is it?” and “Book a meeting with Alice”. `04_tts.py` speaks the reply on macOS. Toggle that with `TTS_ENABLED`.

In `05_pipeline.py`, press **Enter** to start recording, speak, then **Enter** again to stop and transcribe. Use **`t`** then Enter to type instead, or **`q`** to quit. The transcript is printed as `You: …` before the agent runs.

## Project layout

- `01_chat.py` through `06_local.py` — one runnable stage each.
- `requirements.txt` — `openai`, `python-dotenv`, and (for steps 5–6) `mlx-whisper`, `mlx-lm`, `sounddevice`, `numpy`.
- `.env` — `OPENAI_API_KEY` (not committed).

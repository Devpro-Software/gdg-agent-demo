"""Chat loop with all tools, macOS TTS, and local push-to-talk STT (mlx-whisper)."""

import json
import os
import re
import subprocess
import tempfile
import threading
import wave
from datetime import datetime

import mlx_whisper
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gpt-4o-mini"
TTS_ENABLED = True
STT_MODEL = "mlx-community/whisper-tiny"
SAMPLE_RATE = 16000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time in ISO format",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Book a meeting with a given name and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the meeting"},
                    "time": {"type": "string", "description": "The current time. Use the get_current_time tool to get the current time."},
                },
                "required": ["name", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eval_math",
            "description": "Evaluate a Python arithmetic expression. Use only digits and + - * / ** ( ). Use * for multiplication, never the letter x.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python arithmetic, e.g. 2 + 3 * 4 or 10 * 10"},
                },
                "required": ["expression"],
            },
        },
    },
]


def math_expression(expression: str) -> str:
    """Turn a casual math string into a Python arithmetic expression."""
    expression = expression.strip()
    expression = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    expression = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", expression)
    expression = re.sub(r"\s+", "", expression)
    if not expression or not re.fullmatch(r"[0-9+\-*/().]+", expression):
        raise ValueError(expression)
    return expression


def run_tool(name: str, arguments: dict) -> str:
    """Run a tool and return the result"""
    if name == "get_current_time":
        d = datetime.now()
        return d.isoformat()
    if name == "eval_math":
        raw = arguments.get("expression", "0")
        try:
            expression = math_expression(raw)
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception:
            return "Could not evaluate that expression. Use digits and + - * / ** ( )."
    if name == "book_meeting":
        return f"Meeting booked with {arguments.get('name', 'Unknown')} at {arguments.get('time', 'Unknown')}"
    return json.dumps({"error": f"Unknown tool: {name}"})


def record_push_to_talk() -> np.ndarray:
    """Record from the mic until the user presses Enter again."""
    chunks = []
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        chunks.append(indata.copy())

    def wait_for_enter():
        input()
        stop.set()

    print("Recording… press Enter to stop.")
    waiter = threading.Thread(target=wait_for_enter, daemon=True)
    waiter.start()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        stop.wait()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).reshape(-1)


def write_wav(path: str, audio: np.ndarray) -> None:
    """Write mono float32 audio as a 16-bit WAV file."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def transcribe(audio: np.ndarray) -> str:
    """Transcribe local audio with mlx-whisper."""
    if audio.size == 0:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = tmp.name
    try:
        write_wav(path, audio)
        result = mlx_whisper.transcribe(path, path_or_hf_repo=STT_MODEL)
        return (result.get("text") or "").strip()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def get_user_text() -> str | None:
    """Push-to-talk, type fallback, or quit. None means quit."""
    while True:
        try:
            choice = input("Press Enter to talk (t to type, q to quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return None
        if choice in ("q", "quit"):
            print("Bye.")
            return None
        if choice in ("t", "type"):
            try:
                typed = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                return None
            if typed:
                return typed
            continue
        if choice == "":
            audio = record_push_to_talk()
            print("Transcribing…")
            text = transcribe(audio)
            if not text:
                print("(no speech detected)")
                continue
            print("You:", text)
            return text
        print("Unknown choice. Press Enter, t, or q.")


def chat_loop():
    """Chat loop"""
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    messages = [{
        "role": "system",
        "content": "Your name is Jimmy. You are a helpful assistant that can use tools to get the current time, evaluate math expressions, and book meetings. Only output text, no special characters or parentheses.",
    }]

    while True:
        user = get_user_text()
        if user is None:
            break

        messages.append({"role": "user", "content": user})

        count = 0
        while True:
            count += 1
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            if len(resp.choices) == 0:
                print(f"[{count}] No response from model. Error: {resp.error}")
                break

            choice = resp.choices[0]
            msg = choice.message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments or "{}")
                    print(f"[{count}] Running tool {name} with arguments {args}")
                    result = run_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            if msg.content:
                print("Bot:", msg.content)
            messages.append({"role": "assistant", "content": msg.content or ""})
            if TTS_ENABLED and msg.content:
                with open(os.devnull, "w", encoding="utf-8") as devnull:
                    subprocess.call(["say", msg.content], stdout=devnull, stderr=devnull)
            break


if __name__ == "__main__":
    chat_loop()

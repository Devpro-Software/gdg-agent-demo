"""Local model chat with push-to-talk STT and macOS TTS. No OpenAI key."""

import atexit
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave

import mlx_whisper
import numpy as np
import sounddevice as sd
from openai import OpenAI

HOST = "127.0.0.1"
PORT = 8080
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL = "mlx-community/Qwen3-0.6B-4bit"
TTS_ENABLED = True
STT_MODEL = "mlx-community/whisper-tiny"
SAMPLE_RATE = 16000

_server = None


def server_ready() -> bool:
    """True when the local OpenAI-compatible server accepts requests."""
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/v1/models", timeout=1) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def stop_server() -> None:
    """Stop the mlx-lm server subprocess if we started it."""
    global _server
    if _server is None:
        return
    if _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server.kill()
            _server.wait(timeout=5)
    _server = None


def start_server() -> None:
    """Start mlx_lm.server if nothing is already listening."""
    global _server
    if server_ready():
        print(f"Using existing server at {BASE_URL}")
        return

    print(f"Starting local server for {MODEL}…")
    print("(First run downloads the model. That can take a minute.)")
    _server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            MODEL,
            "--host",
            HOST,
            "--port",
            str(PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_server)

    deadline = time.time() + 300
    while time.time() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(
                "mlx_lm.server exited before it became ready. "
                "Is mlx-lm installed? Try: pip install mlx-lm"
            )
        if server_ready():
            print("Server ready.")
            return
        time.sleep(0.5)

    stop_server()
    raise RuntimeError("Timed out waiting for mlx_lm.server to start.")


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
    """Local STT → local LLM → macOS TTS."""
    start_server()
    client = OpenAI(
        base_url=BASE_URL,
        api_key="unused",
    )
    messages = [{
        "role": "system",
        "content": "Your name is Jimmy. You are a helpful assistant. Only output text, no special characters or parentheses.",
    }]

    print(f"Local model: {MODEL}")
    print()

    while True:
        user = get_user_text()
        if user is None:
            break

        messages.append({"role": "user", "content": user})

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
            )
        except Exception as e:
            print(f"Local model request failed: {e}")
            messages.pop()
            continue

        if len(resp.choices) == 0:
            print(f"No response from model. Error: {getattr(resp, 'error', None)}")
            continue

        content = resp.choices[0].message.content or ""
        print("Bot:", content)
        messages.append({"role": "assistant", "content": content})
        if TTS_ENABLED and content:
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                subprocess.call(["say", content], stdout=devnull, stderr=devnull)

    stop_server()


if __name__ == "__main__":
    chat_loop()

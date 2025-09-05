# Local-Voice-AI-Agent
A Windows-first, local voice chat agent: speak → Moonshine STT → Gemma 3 via Ollama → Kokoro TTS → streamed back in a web UI. Optimized for low downloads (1B), adds short memory, retries, timing metrics, and CLI/config controls.

## Features

- Real-time speech-to-text conversion
- Local LLM inference using Ollama
- Text-to-speech response generation
- Web interface for interaction
- Phone number interface option

## Prerequisites (Windows only)

- Windows 10/11
- [Ollama](https://ollama.ai/) – run LLMs locally
- [uv](https://github.com/astral-sh/uv) – fast Python package / resolver

## Installation

### 1) Windows setup

PowerShell:

```powershell
# Go to project folder (adjust the path if different)
cd C:\Users\<you>\Downloads\local-voice-ai-agent-main

# Install uv (one-time) and add to PATH for this session
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

# Create and activate a Python 3.13 virtual environment
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Python dependencies
uv sync

# Pull the smallest model (saves data)
ollama pull gemma3:1b
```

### 2) (Optional) Create your own GitHub repo (your URL)

```powershell
# initialize a new git repo in this folder
git init
git add .
git commit -m "Initial commit: local voice AI agent (Windows)"

# create a new repo on GitHub (via website), then add your remote URL
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

### 3) (If you cloned a fresh copy) Create venv and install deps

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
uv sync
```

### 4. Download required models in Ollama

```powershell
ollama pull gemma3:1b
# Optional higher-quality model (bigger download)
# ollama pull gemma3:4b
```

## Usage

### Basic Voice Chat

```powershell
python .\local_voice_chat.py            # local only
python .\local_voice_chat.py --share    # public link (requires internet)
```

### Advanced Voice Chat (with system prompt)

#### Web UI (default)
```powershell
python .\local_voice_chat_advanced.py                         # defaults to gemma3:1b
python .\local_voice_chat_advanced.py --share                 # try public link
python .\local_voice_chat_advanced.py --server-name 0.0.0.0   # LAN access
python .\local_voice_chat_advanced.py --system-prompt .\system_prompt.txt
python .\local_voice_chat_advanced.py --max-tokens 150 --temperature 0.6 --top-p 0.85
```

#### Phone Number Interface
Get a temporary phone number that anyone can call to interact with your AI:
```powershell
python .\local_voice_chat_advanced.py --phone
```

This will provide you with a temporary phone number that you can call to interact with the AI using your voice.

## New features (1B-friendly)

- Short conversational memory (keeps last few turns) for more natural chat
- Simple retry on LLM errors for robustness
- Timing metrics (STT, LLM, TTS) logged to console
- Tuning flags: `--model`, `--max-tokens`, `--temperature`, `--top-p`
- System prompt file via `--system-prompt path/to/file.txt`
- Optional config file (`--config config.yaml`) to set defaults
- Easy sharing: `--share` (public) or `--server-name 0.0.0.0` (LAN)

Example `config.yaml` (copy `config.example.yaml`):

```yaml
model: gemma3:1b
max_tokens: 160
temperature: 0.65
top_p: 0.9
memory_turns: 4
system_prompt_file: system_prompt.txt
```

Example `system_prompt.txt` (copy `system_prompt.example.txt`):

```text
You are a friendly English tutor. Keep replies short and clear. Correct mispronunciations gently.
```

## How it works

The application uses:
- `FastRTC` for WebRTC communication
- `Moonshine` for local speech-to-text conversion
- `Kokoro` for text-to-speech synthesis
- `Ollama` for running local LLM inference with `Gemma` models

When you speak, your audio is:
1. Transcribed to text using Moonshine
2. Sent to a local LLM via Ollama for processing
3. The LLM response is converted back to speech with Kokoro
4. The audio response is streamed back to you via FastRTC

## Troubleshooting

- Public link fails: use local mode or LAN (`--server-name 0.0.0.0`)
- Mic not heard: allow microphone in Windows Privacy Settings and browser
- Model missing: run `ollama pull gemma3:1b`
- FFmpeg warning from pydub: safe to ignore for this app

## Credits / Reference

- Inspired by the YouTube tutorial: [Local Voice AI Agent tutorial](https://youtu.be/M6vI4Wk-Y4Q?si=BGuYTTjvWTLQ1dAY)

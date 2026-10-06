# Local Voice AI Agent

A real-time voice chat application powered by local AI models. Have voice conversations with local LLMs via Ollama (Gemma 3). Runs fully on your laptop.

## Features

- Real-time speech-to-text conversion (Moonshine)
- Local LLM inference using Ollama (Gemma 3)
- Text-to-speech response generation (Kokoro)
- Short conversational memory for natural multi-turn chat
- Simple retry on LLM errors for robustness
- Timing metrics (STT, LLM, TTS) logged to console
- Web interface via Gradio / FastRTC
- Phone number interface option

## Prerequisites (Windows only)

- Windows 10/11
- [Ollama](https://ollama.ai/) – run LLMs locally
- [uv](https://github.com/astral-sh/uv) – fast Python package manager / resolver

## Installation

### 1) Windows setup

PowerShell:

```powershell
# Go to project folder (adjust the path if different)
cd C:\Users\<you>\Downloads\Local-Voice-AI-Agent

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

### 2) (Optional) Create your own GitHub repo

```powershell
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

### Web UI (default)
```powershell
python .\local_voice_chat.py                                       # defaults to gemma3:1b
python .\local_voice_chat.py --share                               # public link (requires internet)
python .\local_voice_chat.py --server-name 0.0.0.0                 # LAN access
python .\local_voice_chat.py --system-prompt .\system_prompt.txt
python .\local_voice_chat.py --max-tokens 150 --temperature 0.6 --top-p 0.85
```

### Phone Number Interface
Get a temporary phone number that anyone can call to interact with your AI:
```powershell
python .\local_voice_chat.py --phone
```

## Configuration

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `gemma3:1b` | Ollama model to use |
| `--config` | `config.yaml` | YAML config file with defaults |
| `--system-prompt` | — | Path to a text file with a custom system prompt |
| `--max-tokens` | `200` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.9` | Nucleus sampling top-p |
| `--share` | off | Create a public Gradio share link |
| `--server-name` | — | Gradio server_name (e.g., `0.0.0.0` for LAN) |
| `--phone` | off | Launch with FastRTC phone interface |
| `--log-level` | `DEBUG` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Config file

Copy `config.example.yaml` to `config.yaml` and edit:

```yaml
model: gemma3:1b
max_tokens: 160
temperature: 0.65
top_p: 0.9
memory_turns: 4
system_prompt_file: system_prompt.txt
```

CLI flags override config file values, which override hardcoded defaults.

### System prompt

Copy `system_prompt.example.txt` to `system_prompt.txt` and edit:

```text
You are a friendly English tutor. Keep replies short and clear. Correct mispronunciations gently.
```

## How it works

The application uses:
- **FastRTC** for WebRTC communication and VAD (voice activity detection)
- **Moonshine** for local speech-to-text conversion
- **Kokoro** for text-to-speech synthesis
- **Ollama** for running local LLM inference with Gemma models

When you speak, your audio is:
1. Captured and segmented by FastRTC's VAD (ReplyOnPause)
2. Transcribed to text using Moonshine
3. Sent to a local LLM via Ollama (with system prompt and conversation memory)
4. The LLM response is converted back to speech with Kokoro
5. The audio response is streamed back to you via FastRTC

## Troubleshooting

- Public link fails: use local mode or LAN (`--server-name 0.0.0.0`)
- Mic not heard: allow microphone in Windows Privacy Settings and browser
- Model missing: run `ollama pull gemma3:1b`
- FFmpeg warning from pydub: safe to ignore for this app

## Credits / Attribution

Based on [local-voice-ai-agent](https://github.com/jesuscopado/local-voice-ai-agent) by **Jesús Copado** ([MIT License](./LICENSE)).

- Tutorial: [Local Voice AI Agent in 19 lines of Python](https://youtu.be/M6vI4Wk-Y4Q?si=BGuYTTjvWTLQ1dAY)

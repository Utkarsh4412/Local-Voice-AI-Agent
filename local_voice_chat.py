import sys
import argparse
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat
import yaml

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

# Defaults (can be overridden by CLI flags)
MODEL_NAME = "gemma3:1b"
SYSTEM_PROMPT = (
    "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. "
    "Your output will be converted to audio so don't include emojis or special characters in your answers. "
    "Respond to what the user said in a creative and helpful way."
)
NUM_PREDICT = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# Short conversational memory (last few turns)
MAX_MEMORY_TURNS = 4
conversation_memory: Deque[Dict[str, str]] = deque(maxlen=MAX_MEMORY_TURNS * 2)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def echo(audio):
    # STT timing
    stt_start = time.perf_counter()
    transcript = stt_model.stt(audio)
    stt_ms = int((time.perf_counter() - stt_start) * 1000)
    logger.debug(f"🎤 Transcript ({stt_ms} ms): {transcript}")

    # Build messages: system + memory + new user
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation_memory:
        messages.extend(list(conversation_memory))
    messages.append({"role": "user", "content": transcript})

    # LLM call with simple retry
    response_text: str = ""
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            llm_start = time.perf_counter()
            response = chat(
                model=MODEL_NAME,
                messages=messages,
                options={
                    "num_predict": NUM_PREDICT,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                },
            )
            llm_ms = int((time.perf_counter() - llm_start) * 1000)
            response_text = response["message"]["content"]
            logger.debug(f"🤖 Response ({llm_ms} ms): {response_text}")
            last_err = None
            break
        except Exception as e:  # best-effort resilience
            last_err = e
            logger.warning(f"LLM request failed (attempt {attempt + 1}): {e}")
            time.sleep(0.2)

    if last_err is not None:
        response_text = "I'm sorry, I had trouble responding. Could you please repeat that?"

    # Update short memory
    conversation_memory.append({"role": "user", "content": transcript})
    conversation_memory.append({"role": "assistant", "content": response_text})

    # TTS timing
    tts_start = time.perf_counter()
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk
    tts_ms = int((time.perf_counter() - tts_start) * 1000)
    logger.debug(f"🔊 TTS time: {tts_ms} ms")


def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat Advanced")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (get a temp phone number)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Optional YAML config file with defaults",
    )
    parser.add_argument(
        "--model",
        default="gemma3:1b",
        help="Ollama model to use (default: gemma3:1b)",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Path to a text file with a custom system prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p (default: 0.9)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link for the Gradio UI",
    )
    parser.add_argument(
        "--server-name",
        default=None,
        help="Gradio server_name (e.g., 0.0.0.0 for LAN access)",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log verbosity (default: DEBUG)",
    )
    args = parser.parse_args()

    # Apply configuration (config file defaults, CLI overrides)
    config_data = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as cf:
                config_data = yaml.safe_load(cf) or {}
        except FileNotFoundError:
            config_data = {}
        except Exception as e:
            logger.warning(f"Could not read config file '{args.config}': {e}")

    MODEL_NAME = args.model or config_data.get("model", MODEL_NAME)
    NUM_PREDICT = args.max_tokens or config_data.get("max_tokens", NUM_PREDICT)
    TEMPERATURE = args.temperature if args.temperature is not None else config_data.get("temperature", TEMPERATURE)
    TOP_P = args.top_p if args.top_p is not None else config_data.get("top_p", TOP_P)
    MAX_MEMORY_TURNS = int(config_data.get("memory_turns", MAX_MEMORY_TURNS))
    # Recreate deque to change capacity (maxlen is read-only)
    if MAX_MEMORY_TURNS > 0:
        conversation_memory = deque(conversation_memory, maxlen=MAX_MEMORY_TURNS * 2)

    if args.system_prompt:
        try:
            with open(args.system_prompt, "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read().strip() or SYSTEM_PROMPT
        except Exception as e:
            logger.warning(f"Could not read system prompt file: {e}")
    elif config_data.get("system_prompt_file"):
        try:
            with open(config_data["system_prompt_file"], "r", encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read().strip() or SYSTEM_PROMPT
        except Exception as e:
            logger.warning(f"Could not read system prompt file from config: {e}")

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    stream = create_stream()

    if args.phone:
        logger.info("Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("Launching with Gradio UI...")
        launch_kwargs = {"share": args.share}
        if args.server_name:
            launch_kwargs["server_name"] = args.server_name
        stream.ui.launch(**launch_kwargs)

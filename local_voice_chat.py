import sys
import argparse
import time
from collections import deque
from typing import Deque, Dict, List, Optional
import yaml
from loguru import logger

from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from ollama import chat

class VoiceAgent:
    def __init__(self, config: dict):
        self.stt = get_stt_model()
        self.tts = get_tts_model()
        
        self.model_name = config.get("model", "gemma3:1b")
        self.max_tokens = config.get("max_tokens", 200)
        self.temperature = config.get("temperature", 0.7)
        self.top_p = config.get("top_p", 0.9)
        self.system_prompt = config.get("system_prompt", (
            "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. "
            "Your output will be converted to audio so don't include emojis or special characters in your answers. "
            "Respond to what the user said in a creative and helpful way."
        ))
        
        self.memory_turns = config.get("memory_turns", 4)
        self.history: Deque[Dict[str, str]] = deque(maxlen=self.memory_turns * 2) if self.memory_turns > 0 else deque()

    def process_audio(self, audio):
        # 1. Transcribe Audio
        start_stt = time.perf_counter()
        user_text = self.stt.stt(audio)
        logger.debug(f"🎤 Transcribed ({int((time.perf_counter() - start_stt) * 1000)}ms): {user_text}")

        # 2. Prepare Context
        context: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        context.extend(list(self.history))
        context.append({"role": "user", "content": user_text})

        # 3. Generate Response
        reply_text = ""
        error = None
        
        for attempt in range(2):
            try:
                start_llm = time.perf_counter()
                response = chat(
                    model=self.model_name,
                    messages=context,
                    options={
                        "num_predict": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    },
                )
                reply_text = response["message"]["content"]
                logger.debug(f"🤖 LLM Reply ({int((time.perf_counter() - start_llm) * 1000)}ms): {reply_text}")
                error = None
                break
            except Exception as e:
                error = e
                logger.warning(f"LLM generation failed (attempt {attempt + 1}): {e}")
                time.sleep(0.2)

        if error:
            reply_text = "I encountered an error connecting to my brain. Please try again."

        # 4. Update Memory
        if self.memory_turns > 0:
            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": reply_text})

        # 5. Synthesize Speech
        start_tts = time.perf_counter()
        for chunk in self.tts.stream_tts_sync(reply_text):
            yield chunk
        logger.debug(f"🔊 Synthesis finished in {int((time.perf_counter() - start_tts) * 1000)}ms")

    def build_stream(self):
        return Stream(ReplyOnPause(self.process_audio), modality="audio", mode="send-receive")


def load_config(args) -> dict:
    config = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config '{args.config}': {e}")
            
    # CLI Overrides
    if args.model: config["model"] = args.model
    if args.max_tokens is not None: config["max_tokens"] = args.max_tokens
    if args.temperature is not None: config["temperature"] = args.temperature
    if args.top_p is not None: config["top_p"] = args.top_p
    
    prompt_path = args.system_prompt or config.get("system_prompt_file")
    if prompt_path:
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                config["system_prompt"] = f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load system prompt: {e}")
            
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice AI Agent")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--model", help="Override Ollama model")
    parser.add_argument("--system-prompt", help="Path to system prompt text file")
    parser.add_argument("--max-tokens", type=int, help="Max generation tokens")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, help="Nucleus sampling top-p")
    parser.add_argument("--phone", action="store_true", help="Use FastRTC phone interface")
    parser.add_argument("--share", action="store_true", help="Create Gradio public link")
    parser.add_argument("--server-name", help="Gradio server name (e.g. 0.0.0.0)")
    parser.add_argument("--log-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    
    app_config = load_config(args)
    agent = VoiceAgent(app_config)
    stream = agent.build_stream()
    
    if args.phone:
        logger.info("Starting phone interface...")
        stream.fastphone()
    else:
        logger.info("Starting web interface...")
        kwargs = {"share": args.share}
        if args.server_name:
            kwargs["server_name"] = args.server_name
        stream.ui.launch(**kwargs)

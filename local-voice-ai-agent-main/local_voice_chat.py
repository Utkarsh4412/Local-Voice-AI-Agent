import argparse
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from ollama import chat

stt_model = get_stt_model()  # moonshine/base
tts_model = get_tts_model()  # kokoro

MODEL_NAME = "gemma3:1b"


def echo(audio):
    transcript = stt_model.stt(audio)
    response = chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": transcript}]
    )
    response_text = response["message"]["content"]
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Voice Chat (basic)")
    parser.add_argument(
        "--model",
        default="gemma3:1b",
        help="Ollama model to use (default: gemma3:1b)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public share link for the Gradio UI",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model

    stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")
    stream.ui.launch(share=args.share)

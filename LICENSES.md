# Third-Party Licenses

This file records the license for every model weight and runtime library
used by **vaak**. Entries are taken from each project's published license
file, PyPI metadata, or HuggingFace model card. Last verified 2026-10-08.

> **Commercial use column**: "Yes" = permissive, no revenue cap.
> "Restricted" = requires separate commercial agreement or has usage caps.

---

## Model Weights

| Component | Model / Checkpoint | License | Commercial use |
|---|---|---|---|
| STT | Moonshine Base / Tiny (UsefulSensors) | MIT | Yes |
| TTS | Kokoro v1.0 weights (`voices-v1.0.bin`, `kokoro-v1.0.onnx`) | Apache 2.0 | Yes |
| LLM | Llama 3.2 1B / 3B (Meta) | [Llama 3.2 Community License](https://llama.meta.com/llama3_2/license/) | Yes (companies with ≤ 700 M MAU); **restricted** above that threshold |
| LLM | Gemma 3 (Google) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | Restricted — must accept Google's terms; no sub-licensing |
| LLM | Any other Ollama model | Varies by model | Check the model's own license |

---

## Python Libraries

| Library | Version used | License | Source | Commercial use |
|---|---|---|---|---|
| `fastrtc` | 0.0.19 | Apache 2.0 | [GitHub](https://github.com/freddyaboulton/fastrtc) | Yes |
| `fastrtc-moonshine-onnx` | 20241016 | MIT | [GitHub](https://github.com/usefulsensors/moonshine) | Yes |
| `kokoro-onnx` | 0.4.7 | Apache 2.0 | [GitHub](https://github.com/thewh1teagle/kokoro-onnx) | Yes |
| `av` | 14.2.0 | BSD-3-Clause | [GitHub](https://github.com/PyAV-Org/PyAV) | Yes |
| `aiortc` | 1.11.0 | BSD-3-Clause | [GitHub](https://github.com/aiortc/aiortc) | Yes |
| `gradio` | ≥ 4.0, < 5.40 | Apache 2.0 | [GitHub](https://github.com/gradio-app/gradio) | Yes |
| `ollama` (Python client) | latest | MIT | [GitHub](https://github.com/ollama/ollama-python) | Yes |
| `pydantic` | ≥ 2.0 | MIT | [GitHub](https://github.com/pydantic/pydantic) | Yes |
| `loguru` | latest | MIT | [GitHub](https://github.com/Delgan/loguru) | Yes |
| `pyyaml` | latest | MIT | [GitHub](https://github.com/yaml/pyyaml) | Yes |
| `numpy` | latest | BSD-3-Clause | [GitHub](https://github.com/numpy/numpy) | Yes |
| `huggingface-hub` | latest | Apache 2.0 | [GitHub](https://github.com/huggingface/huggingface_hub) | Yes |

---

## This Repository

`vaak` itself is released under the MIT License. See [`LICENSE`](LICENSE).



---

## Note on Ollama-served Models

Ollama downloads model weights separately from its runtime. The Python
`ollama` client library is MIT-licensed, but **the weights it serves are
governed by their own licenses** (see the Model Weights table above).
Always verify the license of the specific model tag you pull before
deploying commercially.

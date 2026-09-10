# ninfer-xx90-win v0.1.0 — Windows x64 binaries

Prebuilt universal binaries for RTX 3090 (sm_86), RTX 4090 (sm_89), and RTX 5090 (sm_120a).
Built from the source at the `v0.1.0` tag of this repository (CUDA 13.x toolchain, MSVC + Ninja).

## Contents

- `ninfer.exe` — one-shot CLI inference
- `ninfer-serve.exe` — OpenAI/Anthropic-compatible HTTP server
- `ninfer-perplexity.exe` — offline causal-perplexity evaluation
- `*.dll` — required FFmpeg/libcurl runtime DLLs; keep them next to the .exe files
- `download-model.bat` — interactive model downloader (asks which GPU you have)
- `run-model-3090-4090.bat` — start the server with the groupwise-int artifact (3090/4090)
- `run-model-5090.bat` — start the server with the nvfp4 artifact (5090)

## Quick start

1. Unpack the archive into one folder (e.g. `C:\ninfer\`).
2. Run `download-model.bat` and pick 1 for RTX 3090/4090 or 2 for RTX 5090
   (downloads into `.\models`, resumable).
3. Run `run-model-3090-4090.bat` or `run-model-5090.bat`. The server listens on
   `http://127.0.0.1:8080/v1`; use model id `qwen3.8-27b` in requests.

## Requirements

- 64-bit Windows 10/11, x64
- NVIDIA GeForce RTX 3090 / 4090 / 5090 with a current driver (no CUDA Toolkit install needed)
- Microsoft Visual C++ 2015-2022 Redistributable (x64) — already present on most systems;
  get it from Microsoft ("VC++ 2015-2022") if the exes refuse to start with a missing MSVCP140 error
- A `.ninfer` model artifact (see Quick start above, or directly from
  https://huggingface.co/neroued/Qwen3.8-27B-NInfer / https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)

## Manual CLI run

```bat
ninfer.exe models\qwen3_8_27b.ninfer --prompt "Hello!" --max-context 32768 --max-new 512 --kv-dtype rk8v4 --spec dflash2 --draft-tokens 7 --greedy
```

Notes:

- On RTX 3090/4090 use the `groupwise-int` artifact (`qwen3_8_27b.ninfer`). The `nvfp4` artifact
  (`qwen3_8_27b_nvfp4.ninfer`) runs on RTX 5090 only.
- `--kv-dtype` accepts `bf16`, `int8`, `rk8v4`, `rk4v4`, `rk4v4-e8`, `rk2v4-e8` on all supported
  cards; `fp8`, `nvfp4`, and `k8v4` are RTX 5090 only.
- Speculative decoding: `--spec dflash2 --draft-tokens 7` (recommended) or `--spec mtp
  --draft-tokens 3`.
- With several GPUs, CUDA device indices can differ from `nvidia-smi` order. Set
  `CUDA_VISIBLE_DEVICES` (or `CUDA_DEVICE_ORDER=PCI_BUS_ID`) before choosing `--device`.
- Server model id for API requests is `qwen3.8-27b`.

Full option documentation: see the repository README and docs.

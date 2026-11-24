# Inkling MLX Agent

Run Inkling 2 Mini locally on Apple Silicon with Pi or OpenCode. The native MLX path preserves the official checkpoint's NVFP4 and MXFP8 expert weights and reuses the exact shared token prefix across Pi turns. The repository also contains an experimental attention LoRA training pipeline for Pi.

[Benchmark results and methodology](RESULTS.md) · [Training and evaluation](adaptation/README.md)

## Download the adapter

The [experimental v0.1.0 release](https://github.com/Yxonyx/inkling-pi-mlx-optimization/releases/tag/v0.1.0) contains the rank-16 attention LoRA for the official Inkling 2 Mini checkpoint. It is the bilingual English–Hungarian build. The native server applies it to every request when launched with `--adapter`. See the [adapter notes](results/adapter-v0.1.0.md) for the measurements, source credits and loading instructions.

## Measured on M2 Max

| Workload | Baseline | Optimized |
| --- | ---: | ---: |
| Full Pi task with three dependent file reads | 24.2 s | **14.3 s** |
| Six replayed Pi requests, including cache fill | 99.7 s | **43.8 s** |
| Median first-token latency on the four replay cache hits | 15.32 s | **1.40 s** |

Both columns use the same native MLX backend and adapter. Cache on and off produced identical generated token IDs, sampled log probabilities and stop reasons across the measured requests. The full-task figure comes from one read-only Pi task on a 64 GiB M2 Max. Speedup depends on how much prompt history can be reused.

## Experimental Pi training

An NVIDIA **B200** ran attention-only LoRA training on 85 verifier-passing Pi trajectories. In a preliminary 103-task evaluation, the candidate improved on 13 tasks and regressed on 12. The overall pass rate was 50.2% versus 50.5% for the stock model. A separate local adapter comparison reduced tool calls from 8 to 3 on one Go task and increased them from 9 to 10 on another shared success. These are early, task-level results. The adapter remains experimental pending a larger paired evaluation under the current Pi protocol.

A separate open-data continuation ran six attention-only LoRA updates from an existing Inkling checkpoint. It replayed the previous training mix and introduced 26 curated examples from two public sources. Six resolved Open-SWE-Traces edits were adapted to Pi and verified on isolated file excerpts. See the [training note](results/open-data-continuation.md).

## Setup

The oMLX setup below supports Pi and OpenCode. For the native checkpoint, see the [native MLX backend](#native-mlx-backend).

The recorded configuration uses Python 3.12, Pi 1.6.0, OpenCode 1.19.5, and oMLX 0.8.2. Use macOS on Apple Silicon, Node.js 22.19 or later, `uv`, and the Hugging Face `hf` CLI. OpenCode is required only for its agent smoke and matrix runs.

```sh
uv venv --python 3.12 .venv
uv venv --python 3.12 .venv-omlx
uv pip sync --python .venv/bin/python results/m2-max/mlx-requirements.txt
uv pip sync --python .venv-omlx/bin/python results/m2-max/omlx-requirements.txt
npm ci --ignore-scripts
```

Download the oQ4e model once, or set `OQ_MODEL` to an existing copy of the pinned snapshot. The download is about 16.9 GB.

```sh
OQ_MODEL=$(hf download mlx-community/Inkling-2-Mini-oQ4e-mtp \
  --revision 3eea042751cd296b7266628a8342c1749242e1fe)
OQ_MODEL=${OQ_MODEL#path=}
sh serve_omlx.sh "$OQ_MODEL" off
```

The server binds to `127.0.0.1:17481`. The arguments select DSpark state and, optionally, the `safe` or `balanced` memory guard. `safe` is the default. The recorded agent edit and matrix use `balanced` under controlled conditions.

## Verify and benchmark

In another terminal, from this repository:

```sh
.venv/bin/python check_tool_calls.py http://127.0.0.1:17481/v1 inkling-mini
sh smoke_agents.sh read
.venv/bin/python bench_cache.py http://127.0.0.1:17481/v1 inkling-mini bench_mlx.py
```

For the verified edit, restart the server with `sh serve_omlx.sh "$OQ_MODEL" off balanced` and run `sh smoke_agents.sh edit`. The smoke tests use separate fixture directories and a verifier outside both working copies. See the [Pi](profiles/pi/models.json) and [OpenCode](profiles/opencode.json) profiles for model IDs and context limits.

Stop the standalone server before starting the agent matrix. The matrix starts and stops its own server for each cell and saves logs under `runs/`.

```sh
.venv/bin/python bench_agent_matrix.py "$OQ_MODEL"
```

The matrix exercises the same edit with all four combinations of reusable prefix cache and DSpark. Use `--cell on-off` for one cell. Cache-off disables oMLX's cross-request paged cache. It is an ablation and should be reported separately from a comparison against stock oMLX defaults.

For raw MLX-LM inference, download the independent baseline checkpoint, about 17.3 GB.

```sh
BASE_MODEL=$(hf download mlx-community/Inkling-2-Mini-4bit \
  --revision 9856dc53edead71405f8348930937c3638ff21a2)
BASE_MODEL=${BASE_MODEL#path=}
.venv/bin/python bench_mlx.py "$BASE_MODEL" bench_mlx.py \
  --contexts 4096 16384 32768 65536 --repeats 1 --max-tokens 64
```

Model weights remain in their original repositories. [oQ4e checkpoint](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp) · [4-bit MLX checkpoint](https://huggingface.co/mlx-community/Inkling-2-Mini-4bit) · [MLX-LM implementation](https://github.com/ml-explore/mlx-lm/pull/2086)

## Native MLX backend

After the setup above, fetch the pinned official model and the community MLX-VLM wheel. The wheel is kept in an isolated import directory within the existing virtual environment.

```sh
NATIVE_MODEL=$(hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7 --exclude mtp.safetensors --quiet)
NATIVE_WHEEL=$(hf download nativ-community/Inkling-2-Mini-nvfp4 \
  runtime/mlx_vlm-0.8.0+inkling2mini-py3-none-any.whl \
  --revision 26f71eef259a1dd7fc476778d67020351e0dfb8c --quiet)
printf '%s  %s\n' d4a9c48a56719eae44879151f7c147776ee0dc3e593c8cb9206c9cc056e29745 "$NATIVE_WHEEL" | shasum -a 256 -c -
uv pip install --python .venv-omlx/bin/python --no-deps --target .venv-omlx/native-overlay "$NATIVE_WHEEL"
PYTHONPATH=.venv-omlx/native-overlay HF_HUB_OFFLINE=1 \
  .venv-omlx/bin/python -m adaptation.local_mlx "$NATIVE_MODEL"
```

The expected wheel SHA-256 is `d4a9c48a56719eae44879151f7c147776ee0dc3e593c8cb9206c9cc056e29745`. The server listens on loopback port 17681 and accepts token IDs. It serves the stock checkpoint by default, supports `--adapter DIRECTORY` for a compatible attention LoRA, and accepts `--no-prefix-cache` for an A/B comparison. The v0.1.0 release supplies a compatible adapter for this runtime.

Check prefix reuse and exact output parity with `python3 verify_native_prefix.py` while the server is running.

To serve Pi-compatible chat, prepare the pinned Tinker virtual environment using the [training setup](adaptation/README.md#tinker-preflight-and-one-step-sft), then start the local protocol bridge in another terminal.

```sh
mkdir -p runs
HF_HUB_OFFLINE=1 .venv-tinker/bin/python -m adaptation.pi_proxy \
  --backend local --base-url http://127.0.0.1:17681/v1 --model inkling-native \
  --capture "runs/native-pi-$(date +%s).jsonl" --token-cap 10000000 --port 17682
```

Start Pi with the included native profile. The bridge formats Pi's messages and tools with the pinned Inkling renderer.

```sh
mkdir -p runs/native-pi-profile
cp profiles/pi-native/models.json runs/native-pi-profile/models.json
PI_CODING_AGENT_DIR="$PWD/runs/native-pi-profile" node_modules/.bin/pi \
  --provider inkling-local --model inkling-native --offline --no-mcp \
  --no-extensions --no-skills --no-prompt-templates --no-context-files \
  --no-session --no-approve
```

## License

The project code is available under [Apache 2.0](LICENSE). Model weights and dependencies are distributed by their respective publishers.

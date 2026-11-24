# Benchmark results

The local tests ran on an Apple M2 Max with 64 GiB of unified memory. Runtime tests hold the model weights fixed. Training tests change the adapter and compare verified task outcomes.

## Native MLX prompt-prefix reuse

The two arms used the same official Inkling 2 Mini checkpoint, attention LoRA, MLX runtime and sampling settings. The official expert weights stayed in NVFP4 for 20 layers and MXFP8 for four. The backend keeps one prompt checkpoint, aligned to a 256-token boundary, up to 8,192 tokens. It restores that state only when every cached prompt token matches the new request. The native runtime uses the pinned [community MLX-VLM wheel](https://huggingface.co/nativ-community/Inkling-2-Mini-nvfp4/tree/26f71eef259a1dd7fc476778d67020351e0dfb8c/runtime).

| Measurement | Cache off | Cache on | Scope |
| --- | ---: | ---: | --- |
| Pi task wall time | 24.170 s | 14.262 s | One local task with three dependent file reads |
| Total HTTP time | 99.691 s | 43.822 s | Six captured Pi requests from two sessions |
| Median first-token latency | 15.321 s | 1.402 s | Four cache hits in the six-request replay |

The live task completed the same three reads and returned the correct answer in both arms. The exact prompt IDs, generated IDs, sampled log probabilities and stop reasons matched on all four model requests. The replay used temperature 1.0, seed 23 and 32 output tokens per request. All 192 generated tokens, sampled log probabilities and stop reasons matched. Replay totals include the two initial cache misses. The first-token latency is measured inside the native backend. It describes model readiness before the chat bridge delivers a response to Pi.

[Sanitized timing data](results/m2-max/native-prefix-pi.json) · [Native server](adaptation/local_mlx.py)

## Pi training experiments

- **B200 rejection-sampling fine-tuning.** An attention LoRA was trained on 85 verifier-passing Pi trajectories. In a preliminary comparison on 103 tasks with three attempts per arm and task, pass rates were 50.5% for stock and 50.2% for the adapter. The adapter improved on 13 tasks, regressed on 12 and tied on 78. The earlier evaluation transport omitted reasoning between turns, limiting how directly these results apply to the current Pi protocol. [Aggregated outcomes](results/adaptation/pi-training-preview.json)
- **Local adapter smoke.** Against the previous adapter, a later GRPO candidate used 3 versus 8 tool calls on one shared Go success and 10 versus 9 on one shared Python success. Both adapters failed the third task. These three paired observations establish the variation across tasks.
- **Current adapter versus stock.** Both completed one task and timed out on another, where both edited workspaces passed the external verifier. On the completed task stock used 4 calls and 40.3 s, and the adapter used 5 calls and 52.2 s. The protocol preflight stopped the remaining comparison. The adapter is still experimental.

## oMLX and MLX-LM baselines

These earlier tests used independent community checkpoints and a separate runtime. The recorded environment was macOS 27.2.1, MLX 0.32.4 and oMLX 0.8.2.

| Test | Observation |
| --- | --- |
| oMLX repeated prefix | Median time to the first streamed token fell from 11.120 s to 0.646 s. Median request time fell from 12.610 s to 2.120 s. Output hashes matched in all three pairs. |
| oMLX DSpark | Median decode increased from 37.76 to 40.62 tokens/s with the same oQ4e checkpoint. Output hashes differed across modes. |
| Raw MLX-LM, 65,536 input tokens | One request reached the first token in 165.41 s and decoded at 24.60 tokens/s, with 22.37 GiB peak MLX allocation. |
| Verified edit | Pi and OpenCode each passed five independent checks on a copy of the same Python fixture. Each configuration in the four-cell agent matrix ran once. |

The repeated-prefix and DSpark implementations in this section come from [oMLX](https://github.com/jundot/omlx/releases/tag/v0.8.2). The speed figures describe the measured workloads. The single edit per configuration provides smoke coverage.

[Repeated-prefix pairs](results/m2-max/) · [Raw MLX-LM](results/m2-max/mlx-baseline.jsonl) · [Agent matrix](results/m2-max/agent-matrix.jsonl)

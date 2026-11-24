# Pi training and evaluation

The training experiments adapt Inkling 2 Mini with attention LoRA. Pi trajectories retain the original tool declarations, assistant tool calls and tool results. An external verifier scores the agent's edited workspace. The native runner evaluates the stock model and the adapter as separate arms on the same mixed-precision checkpoint.

[Current results](../RESULTS.md#pi-training-experiments) · [Native MLX setup](../README.md#native-mlx-backend)

## Tinker preflight and one-step SFT

Prepare the pinned environment for the local Pi protocol bridge and the training tools.

```sh
uv venv --python 3.12 .venv-tinker
uv pip sync --python .venv-tinker/bin/python adaptation/requirements-tinker.txt
```

The training preflight renders Pi traces with Inkling's `tml_v0` renderer, checks the prompt and loss masks, and writes an exact plan digest. SFT uses a supplied JSONL file of verified traces. The command below prepares a dry run.

```sh
.venv-tinker/bin/python -m adaptation.sft /path/to/verified-traces.jsonl
```

The input format and paid-run flags are described in [the SFT entry point](sft.py). The preflight and the local bridge run without paid inference. The B200 experiment used the [GPU training entry point](modal_inkling.py) and the run-level measurements summarized in [the results](../RESULTS.md#pi-training-experiments).

## Evaluation protocol

Pair stock and adapter on the same held-out tasks using the same Pi version, tool list, reasoning effort, sampling settings, context window and timeout. Apply the task verifier outside the agent's workspace. Record pass rate, timeouts and per-task gains and regressions. Report elapsed time, output tokens and tool calls on tasks solved by both arms. An agent that reaches its timeout can leave a correct patch. Record verifier reward and agent completion separately.

The adapter remains experimental. Broader release claims require a new paired run that passes protocol preflight and reproduces the improvement on independent tasks.

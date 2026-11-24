# Open-data continuation

This experiment resumed an existing Inkling 2 Mini checkpoint with rank-16 attention-only LoRA. It ran six optimizer steps across 100 examples and at most 96,607 rendered tokens.

| Group | Examples | Share of loss weight |
| --- | ---: | ---: |
| Prior mix replay | 74 | 52% |
| Reviewed Aya instruction examples | 20 | 28% |
| Open-SWE-Traces edits | 6 | 20% |

The edits were drawn from resolved OpenCode trajectories in Open-SWE-Traces v2.0. Each example was adapted to a Pi read/edit/read sequence on an isolated file excerpt. We executed those actions with Pi tools and checked the final file bytes. This validates the reconstructed edit. The upstream dataset reports the full repository outcome. [Aya Dataset, Apache 2.0](https://huggingface.co/datasets/CohereLabs/aya_dataset) · [NVIDIA Open-SWE-Traces, CC BY 4.0](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)

## Local regression smoke

Both the starting and continued adapters passed 8 of 8 checks over two four-turn conversations on native MLX. The continued checkpoint stopped normally at every turn and produced zero repeated final answers after topic changes. The prompts were held outside the training mix. The check covers dialogue continuity and repetition. A larger paired Pi capability evaluation is pending.

The list-price upper estimate for this continuation is USD 0.0725, based on rendered training tokens. [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/)

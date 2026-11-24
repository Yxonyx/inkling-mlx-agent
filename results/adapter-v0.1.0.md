# Inkling 2 Mini attention LoRA v0.1.0

This release contains the continuation adapter tested with native MLX and the Pi protocol. It uses rank 16 and alpha 32 across six attention projections in each of the 24 physical layers. Apply it to the official [Inkling 2 Mini base revision](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7).

## Load on Apple Silicon

Follow the [native MLX setup](../README.md#native-mlx-backend) to prepare `NATIVE_MODEL` and the pinned runtime. Download and extract the [v0.1.0 adapter](https://github.com/Yxonyx/inkling-pi-mlx-optimization/releases/tag/v0.1.0) from the release.

```sh
mkdir -p adapters
unzip inkling-2-mini-attention-lora-v0.1.0.zip -d adapters
PYTHONPATH=.venv-omlx/native-overlay HF_HUB_OFFLINE=1 \
  .venv-omlx/bin/python -m adaptation.local_mlx "$NATIVE_MODEL" \
  --adapter adapters/inkling-2-mini-attention-lora-v0.1.0
```

Use the Pi bridge and profile described in the native MLX setup. This launch applies the adapter to every request. The archive contains the adapter weights, configuration, license and attribution notice.

## Training and measured behavior

The released adapter continues an initial English–Hungarian attention LoRA. The continuation ran six optimizer steps over 100 examples, with 74 replay items, 20 reviewed Hungarian Aya prompts and six Open-SWE-Traces edits adapted to Pi fixtures. The 100 examples represent the continuation stage. [Training note](open-data-continuation.md)

| Check | Comparison | Released adapter |
| --- | ---: | ---: |
| English basics | 14/14 stock | 14/14 |
| English challenge, 512 generated tokens | 1/7 stock | 2/7 |
| Hungarian dialogue continuity | 8/8 prior adapter | 8/8 |

The English challenge gain was one answer completed within the token limit. The Hungarian check covered two short conversations, including translation, a topic switch and recall of a previous turn. Hungarian improvement over stock and overall Pi task performance need a broader paired evaluation. Occasional shifts in formality were observed in hands-on use.

The separate M2 Max cache benchmark reduced one Pi task from 24.17 s to 14.26 s and six replayed requests from 99.69 s to 43.82 s. Both sides of that benchmark used the same weights. [Measurements and scope](../RESULTS.md)

## Attribution

Base model by Thinking Machines Lab, Apache 2.0. Reviewed Hungarian prompts were adapted from [Cohere Labs Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset), Apache 2.0. Six resolved code fixes came from [NVIDIA Open-SWE-Traces v2.0](https://huggingface.co/datasets/nvidia/Open-SWE-Traces), CC BY 4.0, and were reconstructed as small Pi read/edit/read fixtures. The release archive includes the checked-out source instance IDs and repository licenses in its notice.

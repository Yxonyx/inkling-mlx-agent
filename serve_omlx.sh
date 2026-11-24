#!/bin/sh
set -eu

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  printf 'Usage: sh serve_omlx.sh MODEL_SNAPSHOT off|on [safe|balanced]\n' >&2
  exit 2
fi

case "$2" in
  off) mtp=false ;;
  on) mtp=true ;;
  *) printf 'Select off or on for DSpark.\n' >&2; exit 2 ;;
esac

guard="${3:-safe}"
case "$guard" in
  safe|balanced) ;;
  *) printf 'Select safe or balanced for the memory guard.\n' >&2; exit 2 ;;
esac

snapshot=$(cd "$1" && pwd -P)
cd "$(dirname "$0")"
base_dir="${INKLING_SERVER_STATE:-$PWD/runs/omlx-$2}"
cache="${INKLING_PREFIX_CACHE:-on}"
case "$cache" in
  off|on) ;;
  *) printf 'Select off or on for INKLING_PREFIX_CACHE.\n' >&2; exit 2 ;;
esac
mkdir -p "$base_dir/models"

.venv-omlx/bin/python - "$snapshot" <<'PY'
import json
import sys
from pathlib import Path

snapshot = Path(sys.argv[1])
index = json.loads((snapshot / "model.safetensors.index.json").read_text())
files = set(index["weight_map"].values()) | {"mtp.safetensors"}
missing = sorted(name for name in files if not (snapshot / name).is_file())
if missing:
    sys.exit(f"Incomplete oQ4e checkpoint: {', '.join(missing)}")
PY

ln -sfn "$snapshot" "$base_dir/models/inkling-mini"
# INKLING_LORA_ADAPTER: an oMLX adapter folder from adaptation.mlx_adapter convert.
.venv-omlx/bin/python - "$base_dir/model_settings.json" "$mtp" "${INKLING_LORA_ADAPTER:-}" <<'PY'
import json
import sys
from pathlib import Path

settings = {"mtp_enabled": sys.argv[2] == "true"}
if sys.argv[3]:
    adapter = Path(sys.argv[3]).resolve()
    if not (adapter / "adapter_config.json").is_file():
        sys.exit(f"Not an adapter folder: {adapter}")
    settings["lora_adapter"] = str(adapter)
Path(sys.argv[1]).write_text(json.dumps({"version": 2, "models": {"inkling-mini": settings}}) + "\n")
PY

exec .venv-omlx/bin/python - "$base_dir" "$guard" "$cache" <<'PY'
import sys
from functools import partial
from pathlib import Path

from omlx.cli import main
from omlx.cluster import launch, worker_shim

base = Path(sys.argv[1]).resolve()
guard = sys.argv[2]
cache = sys.argv[3]
# v0.8.2's startup helpers otherwise use ~/.omlx even with --base-path.
launch.reap_orphaned_launches = partial(
    launch.reap_orphaned_launches, state_dir=base / "cluster" / "runtime"
)
worker_shim.ensure_cluster_python_shim = partial(
    worker_shim.ensure_cluster_python_shim, home=base
)
sys.argv[1:] = [
    "serve", "--host", "127.0.0.1", "--port", "17481", "--base-path", str(base),
    "--model-dir", str(base / "models"), "--no-hf-cache", "--memory-guard", guard,
    "--max-concurrent-requests", "1", "--paged-ssd-cache-max-size", "8GB",
]
if cache == "off":
    sys.argv.append("--no-cache")
main()
PY

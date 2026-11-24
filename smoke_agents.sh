#!/bin/sh
set -eu
cd "$(dirname "$0")"
project="$PWD"
case "${1:-read}" in
  read)
    work="$project/runs/agent-smoke"
    task='Use the read tool to read README.md. Reply with only the exact first line.'
    pi_tools=read
    oc_agent=inkling-smoke
    ;;
  edit)
    work="$project/runs/agent-edit-smoke"
    task='Edit slug.py. In normalize_slug, lowercase text, turn each run of whitespace or hyphens into one hyphen, and remove leading and trailing hyphens. Preserve other characters and the function signature. Use only the Python standard library.'
    pi_tools=read,edit
    oc_agent=inkling-edit-smoke
    ;;
  *) printf 'Usage: sh smoke_agents.sh [read|edit]\n' >&2; exit 2 ;;
esac
work="${INKLING_SMOKE_STATE:-$work}"

mkdir -p "$work/pi-fixture" "$work/oc-fixture" "$work/pi" "$work/opencode"
cp profiles/pi/models.json "$work/pi/models.json"
for fixture in "$work/pi-fixture" "$work/oc-fixture"; do
  printf 'inkling-fixture-97c1\n' > "$fixture/README.md"
  cp fixtures/slug.py "$fixture/slug.py"
done

cd "$work/pi-fixture"
started=$(date +%s)
pi_status=0
PI_CODING_AGENT_DIR="$work/pi" "$project/node_modules/.bin/pi" \
  --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files \
  --no-session --mode json --tools "$pi_tools" --model inkling-local/inkling-mini \
  "$task" > "$work/pi.jsonl" || pi_status=$?
pi_elapsed=$(($(date +%s) - started))

cd "$work/oc-fixture"
started=$(date +%s)
oc_status=0
env XDG_CONFIG_HOME="$work/opencode/config" XDG_DATA_HOME="$work/opencode/data" \
  XDG_STATE_HOME="$work/opencode/state" XDG_CACHE_HOME="$work/opencode/cache" \
  OPENCODE_TEST_HOME="$work/opencode" OPENCODE_DISABLE_EXTERNAL_SKILLS=1 \
  OPENCODE_DISABLE_DEFAULT_PLUGINS=1 OPENCODE_DISABLE_PROJECT_CONFIG=1 \
  OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_CONFIG="$project/profiles/opencode.json" \
  opencode run --pure --agent "$oc_agent" --format json --model inkling-local/inkling-mini \
  "$task" > "$work/opencode.jsonl" || oc_status=$?
oc_elapsed=$(($(date +%s) - started))

if [ "$oc_agent" = inkling-edit-smoke ]; then
  "$project/.venv/bin/python" "$project/verify_edit.py" \
    "$work" "$pi_elapsed" "$oc_elapsed" "$pi_status" "$oc_status"
  exit
fi

"$project/.venv/bin/python" - "$work" "$pi_status" "$oc_status" <<'PY'
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
expected = "inkling-fixture-97c1"
pi = [json.loads(line) for line in (work / "pi.jsonl").read_text().splitlines()]
oc = [json.loads(line) for line in (work / "opencode.jsonl").read_text().splitlines()]
pi_read = any(
    row["type"] == "tool_execution_end" and row.get("toolName") == "read"
    and not row.get("isError") and expected in json.dumps(row.get("result"))
    for row in pi
)
pi_answers = [
    part.get("text", "").strip()
    for row in pi if row["type"] == "message_end"
    and row.get("message", {}).get("role") == "assistant"
    for part in row["message"]["content"] if part.get("type") == "text"
]
oc_read = any(
    row["type"] == "tool_use" and row["part"].get("tool") == "read"
    and row["part"].get("state", {}).get("status") == "completed"
    and expected in row["part"]["state"].get("output", "")
    for row in oc
)
oc_answers = [row["part"]["text"].strip() for row in oc if row["type"] == "text"]
checks = {
    "pi": sys.argv[2] == "0" and pi_read and (pi_answers or [""])[-1] == expected,
    "opencode": sys.argv[3] == "0" and oc_read and (oc_answers or [""])[-1] == expected,
}
print(json.dumps(checks))
sys.exit(0 if all(checks.values()) else 1)
PY

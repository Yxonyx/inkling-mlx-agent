"""Run the existing coding fixture in each oMLX prefix-cache and DSpark mode."""

import argparse
import itertools
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from bench_api import measure


ROOT = Path(__file__).resolve().parent
API = "http://127.0.0.1:17481/v1"
MODEL = "inkling-mini"


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_for_server(process):
    deadline = monotonic() + 45
    while process.poll() is None and monotonic() < deadline:
        try:
            with urlopen(f"{API}/models", timeout=1) as response:
                models = json.load(response)["data"]
            if any(row["id"] == MODEL for row in models):
                return
        except (URLError, TimeoutError):
            sleep(0.5)
    raise RuntimeError("oMLX did not become ready")


def run_cell(snapshot, cell, cache, dspark, timeout):
    cell.mkdir()
    with socket.socket() as port_check:
        port_check.settimeout(0.5)
        if port_check.connect_ex(("127.0.0.1", 17481)) == 0:
            raise RuntimeError("Port 17481 is already in use")
    env = dict(os.environ, INKLING_PREFIX_CACHE=cache, INKLING_SERVER_STATE=str(cell / "server"))
    log_path = cell / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            ["sh", "serve_omlx.sh", str(snapshot), dspark, "balanced"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            wait_for_server(server)
            warmup = measure(API, MODEL, "Respond with only the word ready.", 8, 0)
            (cell / "warmup.json").write_text(json.dumps(warmup) + "\n")
            with (cell / "verifier.stderr").open("w") as stderr, (cell / "verifier.json").open("w") as output:
                agent = subprocess.Popen(
                    ["sh", "smoke_agents.sh", "edit"], cwd=ROOT,
                    env=dict(os.environ, INKLING_SMOKE_STATE=str(cell / "agents")),
                    stdout=output, stderr=stderr, start_new_session=True,
                )
                try:
                    agent.wait(timeout=timeout)
                finally:
                    stop(agent)
        finally:
            stop(server)

    events = log_path.read_text()
    accepted = re.findall(r"MTP\[\d+\].*?accept=(\d+)/(\d+)", events)
    outcomes = json.loads((cell / "verifier.json").read_text())
    for agent, trace_name in (("pi", "pi.jsonl"), ("opencode", "opencode.jsonl")):
        trace = [json.loads(line) for line in (cell / "agents" / trace_name).read_text().splitlines()]
        event_type = "tool_execution_end" if agent == "pi" else "tool_use"
        outcomes[agent]["tool_events"] = sum(row.get("type") == event_type for row in trace)
    result = {
        "prefix_cache": cache,
        "dspark": dspark,
        "prefix_restores": len(re.findall(r"Prefix cache restore", events)),
        "draft_loaded": "Inkling 2 DSpark drafter loaded" in events,
        "draft_accepted": sum(int(count) for count, _ in accepted),
        "draft_attempted": sum(int(count) for _, count in accepted),
        "agents": outcomes,
    }
    (cell / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--cell", choices=("all", "off-off", "on-off", "off-on", "on-on"), default="all")
    parser.add_argument("--timeout", type=int, default=240, help="Seconds for the two agents per cell")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="agent-matrix-", dir=runs))
    print(json.dumps({"output": str(output)}), flush=True)
    pairs = itertools.product(("off", "on"), repeat=2)
    for cache, dspark in pairs:
        name = f"{cache}-{dspark}"
        if args.cell in ("all", name):
            row = run_cell(args.snapshot.resolve(), output / name, cache, dspark, args.timeout)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

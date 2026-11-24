"""Validate and run the versioned Inkling Pi task set against local inference."""

import argparse
from dataclasses import asdict
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from time import monotonic
from urllib.parse import urlparse

from adaptation.model_proxy import Proxy
from adaptation.tasks import BY_ID, TASKS


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "inkling-pi-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def suite_digest():
    return digest([asdict(item) for item in TASKS])


def write_files(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def verify(task, workspace):
    with tempfile.TemporaryDirectory(prefix="inkling-verify-") as temp:
        target = Path(temp)
        for name in task.starter:
            source = workspace / name
            if source.is_symlink() or not source.is_file():
                return 1, f"Missing regular file {name}"
            shutil.copyfile(source, target / name)
        try:
            command = [sys.executable, "-B", "-c", task.oracle]
            if sys.platform == "darwin":
                command = ["/usr/bin/sandbox-exec", "-p", sandbox([target]), *command]
            result = subprocess.run(
                command, cwd=target,
                capture_output=True, text=True, timeout=10,
                env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return 124, "External verifier timed out"


def validate():
    families = {split: {item.family for item in TASKS if item.split == split} for split in ("train", "holdout")}
    if families["train"] & families["holdout"] or len(BY_ID) != len(TASKS):
        raise RuntimeError("Task IDs must be unique and the split families must be disjoint")
    for item in TASKS:
        with tempfile.TemporaryDirectory(prefix="inkling-suite-") as temp:
            workspace = Path(temp)
            write_files(workspace, item.starter)
            before, _ = verify(item, workspace)
            write_files(workspace, item.reference)
            after, detail = verify(item, workspace)
            if before == 0 or after != 0:
                raise RuntimeError(f"{item.id}: starter={before} reference={after}\n{detail}")
    print(json.dumps({
        "suite_sha256": suite_digest(),
        "train": sum(item.split == "train" for item in TASKS),
        "holdout": sum(item.split == "holdout" for item in TASKS),
        "verifiers": "starter rejected and reference accepted",
    }))


def sandbox(paths, port=None, frozen=()):
    # The agent reads and writes only its own work, home and tmp folders, and reaches only
    # the inference-only proxy. Captures and events live outside its reach.
    paths = [str(Path(path).resolve()) for path in paths]
    read_paths = [
        "/System", "/usr", "/bin", "/sbin", "/opt/homebrew", "/Library/Developer",
        "/Applications/Xcode.app", "/private/etc", "/dev",
        str(ROOT / "node_modules"), str(ROOT / ".venv"),
        str(Path(sys.executable).resolve().parents[1]), str(ROOT / "adaptation" / "capture.ts"),
        *paths,
    ]
    readable = " ".join(f"(subpath {json.dumps(path)})" for path in read_paths)
    writable = " ".join(f"(subpath {json.dumps(path)})" for path in paths)
    allow_model = f'(allow network-outbound (remote tcp "localhost:{port}"))' if port else ""
    # Later rules win: these files stay readable but not writable, renamable or removable.
    frozen = " ".join(f"(literal {json.dumps(str(Path(path).resolve()))})" for path in frozen)
    deny_frozen = f"(deny file-write* {frozen})" if frozen else ""
    return f"""
(version 1)
(allow default)
(deny file-read-data (subpath "/Users") (subpath "/Volumes"))
(allow file-read-data {readable})
(deny file-write*)
(allow file-write* {writable} (literal "/dev/null"))
{deny_frozen}
(deny network*)
{allow_model}
"""


def run_task(item, args, output):
    trial = output / item.id
    (trial / "capture").mkdir(parents=True)
    proxy = Proxy(args.base_url.replace("{task}", item.id), trial / "capture" / "requests.jsonl").start()
    try:
        return run_trial(item, args, trial, proxy.address[1])
    finally:
        proxy.close()


def run_trial(item, args, trial, port):
    work, home, tmp = (trial / name for name in ("work", "home", "tmp"))
    for directory in (work, home, tmp):
        directory.mkdir()
    write_files(work, item.starter)
    agent_dir = home / "pi"
    agent_dir.mkdir()
    model = {
        "id": args.model, "name": args.label, "input": ["text"], "reasoning": False,
        "contextWindow": 16384, "maxTokens": 4096,
    }
    (agent_dir / "models.json").write_text(json.dumps({"providers": {"bench": {
        "baseUrl": f"http://127.0.0.1:{port}/v1",
        "api": "openai-completions", "apiKey": "local",
        "compat": {"supportsDeveloperRole": False, "supportsStore": False}, "models": [model],
    }}}))
    profile = trial / "sandbox.sb"
    # Pi writes trust state next to models.json, but the agent must not repoint the endpoint.
    profile.write_text(sandbox([work, home, tmp], port, frozen=[agent_dir, agent_dir / "models.json"]))
    endpoint = agent_dir / "models.json"
    endpoint_sha = hashlib.sha256(endpoint.read_bytes()).hexdigest()
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required")
    env = {
        "PATH": f"{Path(sys.executable).parent}:{Path(node).parent}:/usr/bin:/bin",
        "TMPDIR": str(tmp), "LANG": "en_US.UTF-8",
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "INKLING_PI_EFFORT": str(args.effort), "INKLING_PI_TEMPERATURE": str(args.temperature),
    }
    prompt = (
        "Use only files in the working directory and Python's standard library. "
        "Make the required edits and run a focused test when useful.\n\n" + item.instruction
    )
    command = [
        "/usr/bin/sandbox-exec", "-f", str(profile), str(ROOT / "node_modules/.bin/pi"),
        "--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-context-files",
        "--no-session", "--mode", "json", "--tools", "read,write,edit,bash",
        "-e", str(ROOT / "adaptation/capture.ts"), "--model", f"bench/{args.model}", prompt,
    ]
    start = monotonic()
    with (trial / "events.jsonl").open("w") as events, (trial / "agent.stderr").open("w") as err:
        process = subprocess.Popen(
            # Pi appends piped stdin to the prompt and waits for EOF when stdin is not a TTY.
            command, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=events, stderr=err,
            start_new_session=True,
        )
        try:
            agent_exit = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            agent_exit = 124
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    elapsed = round(monotonic() - start, 3)
    # The sandbox guards the path, not the inode: a hard link from a writable folder could
    # still change the file. Reject the trial unless it is the same single-link file.
    if endpoint.is_symlink() or endpoint.stat().st_nlink != 1 \
            or hashlib.sha256(endpoint.read_bytes()).hexdigest() != endpoint_sha:
        raise RuntimeError(f"{item.id}: Pi's models.json was changed during the trial")
    verification, detail = verify(item, work)
    (trial / "verifier.txt").write_text(detail)
    patch = []
    for name, before in item.starter.items():
        edited = work / name
        if not edited.is_symlink() and edited.is_file():
            after = edited.read_text(errors="replace")
            patch.extend(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{name}", tofile=f"b/{name}",
            ))
    (trial / "changes.diff").write_text("".join(patch))
    events = [json.loads(line) for line in (trial / "events.jsonl").read_text().splitlines()]
    tool_ends = [row for row in events if row.get("type") == "tool_execution_end"]
    answers = [row["message"] for row in events if row.get("type") == "message_end"
               and row["message"].get("role") == "assistant"]
    terminal = answers[-1].get("stopReason") if answers else None
    usage = [answer.get("usage", {}) for answer in answers]
    pi_meta = ROOT / "node_modules/@earendil-works/pi-coding-agent/package.json"
    result = {
        "schema": SCHEMA, "suite_sha256": suite_digest(), "task": item.id, "split": item.split,
        "model": args.model, "label": args.label, "effort": args.effort,
        "passed": verification == 0 and agent_exit == 0 and terminal == "stop",
        "pi_version": json.loads(pi_meta.read_text())["version"], "timeout_s": args.timeout,
        "temperature": args.temperature, "context_window": model["contextWindow"], "output_limit": model["maxTokens"],
        "agent_exit": agent_exit, "verifier_exit": verification, "elapsed_s": elapsed,
        "tool_calls": len(tool_ends), "failed_tool_calls": sum(bool(row.get("isError")) for row in tool_ends),
        "input_tokens": sum(row.get("input", 0) for row in usage),
        "cache_read_tokens": sum(row.get("cacheRead", 0) for row in usage),
        "output_tokens": sum(row.get("output", 0) for row in usage),
        "assistant_turns": len(answers),
        "terminal_stop": terminal,
    }
    (trial / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    run = commands.add_parser("run")
    run.add_argument("--split", choices=("train", "holdout"), required=True)
    run.add_argument("--task", action="append", choices=tuple(BY_ID))
    run.add_argument("--base-url", default="http://127.0.0.1:17481/v1")
    run.add_argument("--model", default="inkling-mini")
    run.add_argument("--label", default="inkling-base")
    run.add_argument("--effort", type=float, default=0.2)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        validate()
        return
    endpoint = urlparse(args.base_url)
    if endpoint.hostname != "127.0.0.1" or endpoint.scheme != "http" or not endpoint.port:
        parser.error("The benchmark requires an explicit local IPv4 HTTP endpoint")
    if not 0 <= args.effort <= 0.99 or not 0 <= args.temperature <= 2 or args.timeout <= 0:
        parser.error("Expected a valid effort, temperature and positive timeout")
    selected = [BY_ID[name] for name in args.task] if args.task else [item for item in TASKS if item.split == args.split]
    if any(item.split != args.split for item in selected) or len({item.id for item in selected}) != len(selected):
        parser.error("Tasks must be unique and belong to the requested split")
    output = args.output.resolve()
    output.mkdir(parents=True)
    for item in selected:
        print(json.dumps(run_task(item, args, output)), flush=True)


if __name__ == "__main__":
    main()

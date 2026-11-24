"""Per-trial inference metrics and a side-by-side comparison of run groups.

For each Harbor trial: pass, total/output tokens, tool calls, turns, the time until the first
file modification, and repetition loops. A loop is any assistant turn that stopped on the
output limit, or whose text/thinking repeats one non-trivial line at least LOOP_REPEATS times.

  .venv-tinker/bin/python -m adaptation.protocol_check runs/tinker-smoke runs/tinker-smoke-t1 ...
"""

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

LOOP_REPEATS = 8
# Any file write (scratch files and new tests included) vs. an in-place edit of existing source.
WRITE_ANY = re.compile(r"(\.write\(|write_text\(|sed\s+-i|\bperl\s+-p?i|\bpatch\b|\btee\b|cat\s*>|[^2&]>\s*[\w./-]+)")
WRITE_BASH = re.compile(r"(sed\s+-i|\bperl\s+-p?i|\bpatch\b|\btee\b|cat\s+>|>\s*[\w./-]+\.\w+)")


def turn_text(message):
    return "".join((part.get("text") or part.get("thinking") or "") for part in message.get("content", [])
                   if isinstance(part, dict))


def loop_turns(answers):
    loops = []
    for index, message in enumerate(answers):
        lines = [line.strip() for line in turn_text(message).splitlines() if len(line.strip()) > 20]
        repeat = collections.Counter(lines).most_common(1)[0][1] if lines else 0
        if message.get("stopReason") == "length" or repeat >= LOOP_REPEATS:
            loops.append({"turn": index, "stop": message.get("stopReason"), "max_line_repeats": repeat})
    return loops


TARGETS = re.compile(r"(?:cat\s*>\s*|>\s*|sed\s+-i\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+|perl\s+-p?i\S*\s+(?:'[^']*'|\S+)\s+)([\w./-]+)")


def changed_files(diff_path):
    """Relative paths that differ in the final workspace."""
    if not diff_path.is_file():
        return set()
    return {line[6:].strip() for line in diff_path.read_text(errors="replace").splitlines() if line.startswith("+++ b/")}


def call_targets(part):
    arguments = part.get("arguments", {}) or {}
    if part.get("name") in ("edit", "write"):
        return [arguments.get("path") or arguments.get("file_path") or ""]
    if part.get("name") == "bash":
        return TARGETS.findall(arguments.get("command", ""))
    return []


def first_change_s(events, final_files):
    """Seconds from the first message to the turn with (a) the first successful file write of any
    kind and (b) the first write to a file that differs in the final workspace (a lasting change)."""
    messages = [row["message"] for row in events if row.get("type") == "message_end"]
    stamps = [message.get("timestamp") for message in messages if message.get("timestamp")]
    if not stamps:
        return None, None
    start, write, lasting = min(stamps), None, None
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for part in message.get("content", []):
            # No success filter: a command can change a file and still exit non-zero (e.g. edit + failing check).
            if part.get("type") != "toolCall":
                continue
            seconds = round((message["timestamp"] - start) / 1000, 1)
            targets = call_targets(part)
            if write is None and (part.get("name") in ("edit", "write") or
                                  (part.get("name") == "bash" and (targets or WRITE_BASH.search(json.dumps(part.get("arguments")))))):
                write = seconds
            # Heuristic: the call names a file that differs at the end and shows a write signal.
            text = json.dumps(part.get("arguments", {}))
            names = {f for f in final_files} | {f.rsplit("/", 1)[-1] for f in final_files}
            writes = part.get("name") in ("edit", "write") or WRITE_ANY.search(text)
            if lasting is None and writes and any(name and name in text for name in names):
                lasting = seconds
    return write, lasting


TASKS = Path(__file__).resolve().parents[1] / "runs" / "datasets" / "scale800@65940b91855c" / "tasks"


def junit_counts(path):
    import xml.etree.ElementTree as ET
    if not path.is_file():
        return None
    cases = ET.parse(path).getroot().iter("testcase")
    total = failed = 0
    for case in cases:
        total += 1
        if any(child.tag in ("failure", "error", "skipped") for child in case):
            failed += 1
    return total - failed, total


def verifier_progress(trial, task):
    """Objective partial progress from the verifier: fail-to-pass tests fixed and pass-to-pass tests broken."""
    verifier = trial / "verifier"
    result = verifier / "test-result.json"
    manifest = TASKS / task / "tests" / "manifest.json"
    if result.is_file() and manifest.is_file():  # Go tasks
        report = json.loads(result.read_text())
        bad = {(row[0], row[1]) for row in report.get("failed_or_skipped", []) + report.get("missing_or_duplicate", [])}
        expected = json.loads(manifest.read_text())["expected"]
        f2p = [(e["package"], e["test"]) for e in expected if e["group"] == "FAIL_TO_PASS"]
        p2p = [(e["package"], e["test"]) for e in expected if e["group"] == "PASS_TO_PASS"]
        return {"f2p_fixed": sum(t not in bad for t in f2p), "f2p_total": len(f2p),
                "p2p_broken": sum(t in bad for t in p2p), "p2p_total": len(p2p)}
    f2p, p2p = junit_counts(verifier / "fail-to-pass.xml"), junit_counts(verifier / "pass-to-pass.xml")
    if f2p:  # Python tasks
        return {"f2p_fixed": f2p[0], "f2p_total": f2p[1],
                "p2p_broken": (p2p[1] - p2p[0]) if p2p else None, "p2p_total": p2p[1] if p2p else None}
    return None


def call_progress(events, first_lasting_s):
    """Tool calls before the first lasting change (all calls if none) and exact repeats of an earlier call."""
    messages = [row["message"] for row in events if row.get("type") == "message_end"]
    stamps = [m.get("timestamp") for m in messages if m.get("timestamp")]
    start = min(stamps) if stamps else 0
    seen, repeats, before = set(), 0, 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for part in message.get("content", []):
            if part.get("type") != "toolCall":
                continue
            key = (part.get("name"), json.dumps(part.get("arguments"), sort_keys=True))
            repeats += key in seen
            seen.add(key)
            if first_lasting_s is None or (message["timestamp"] - start) / 1000 < first_lasting_s:
                before += 1
    return before, repeats


def trial_metrics(trial):
    result = json.loads((trial / "result.json").read_text())
    events = [json.loads(line) for line in (trial / "events.jsonl").read_text().splitlines() if line.startswith("{")]
    answers = [row["message"] for row in events if row.get("type") == "message_end"
               and row["message"].get("role") == "assistant"]
    loops = loop_turns(answers)
    diff = trial / "changes.diff"
    first_write, first_edit = first_change_s(events, changed_files(diff))
    return {
        "trial": trial.name, "temperature": result.get("temperature"), "effort": result.get("effort"),
        "backend": result.get("backend", "omlx-local"), "protocol": (result.get("protocol") or {}).get("id"),
        "protocol_violations": len(result.get("protocol_violations") or []),
        "passed": result.get("passed"), "terminal_stop": result.get("terminal_stop"),
        "tool_calls": result.get("tool_calls"), "turns": result.get("assistant_turns"),
        "prompt_tokens": result.get("input_tokens", 0) + result.get("cache_read_tokens", 0),
        "output_tokens": result.get("output_tokens"), "elapsed_s": result.get("elapsed_s"),
        "first_write_s": first_write, "first_lasting_change_s": first_edit,
        "final_diff_lines": sum(1 for _ in diff.open()) if diff.is_file() else None,
        "loop_turns": len(loops), "loops": loops[:3],
        "verifier_progress": verifier_progress(trial, result.get("task", "")),
        "calls_before_lasting_change": call_progress(events, first_edit)[0],
        "repeated_calls": call_progress(events, first_edit)[1],
    }


def summarize(rows):
    def mean(key):
        values = [row[key] for row in rows if row[key] is not None]
        return round(statistics.mean(values), 1) if values else None
    return {"trials": len(rows), "passed": sum(bool(row["passed"]) for row in rows),
            "with_loop": sum(row["loop_turns"] > 0 for row in rows),
            "violations": sum(row["protocol_violations"] for row in rows),
            "with_lasting_change": sum(row["first_lasting_change_s"] is not None for row in rows),
            "with_final_diff": sum(bool(row["final_diff_lines"]) for row in rows),
            "f2p_fixed": sum((row["verifier_progress"] or {}).get("f2p_fixed", 0) for row in rows),
            "f2p_total": sum((row["verifier_progress"] or {}).get("f2p_total", 0) for row in rows),
            "p2p_broken": sum((row["verifier_progress"] or {}).get("p2p_broken") or 0 for row in rows),
            **{f"mean_{key}": mean(key) for key in ("output_tokens", "prompt_tokens", "tool_calls", "turns",
                                                    "elapsed_s", "first_write_s", "first_lasting_change_s",
                                                    "calls_before_lasting_change", "repeated_calls")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    for root in args.runs:
        for trial in sorted(root.glob("*--r*")):
            if (trial / "result.json").is_file():
                row = trial_metrics(trial)
                groups[f"{row['backend']} T={row['temperature']} effort={row['effort']}"].append(row)
    report = {group: {"summary": summarize(rows), "trials": rows} for group, rows in groups.items()}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""Run the external slug fixture verifier for each agent's working copy."""

import json
import subprocess
import sys
from pathlib import Path


VERIFIER = r"""
from slug import normalize_slug

cases = [
    ("", ""),
    (" - \t-", ""),
    (" A--B \t C -", "a-b-c"),
    ("Déjà__Vu\n  Gone", "déjà__vu-gone"),
    ("already-normal", "already-normal"),
]
for supplied, expected in cases:
    actual = normalize_slug(supplied)
    assert actual == expected, (supplied, expected, actual)
"""


def main():
    work = Path(sys.argv[1])
    report = {}
    for agent, directory, elapsed, status in zip(
        ("pi", "opencode"), ("pi-fixture", "oc-fixture"),
        sys.argv[2:4], sys.argv[4:6], strict=True,
    ):
        result = subprocess.run(
            [sys.executable, "-B", "-c", VERIFIER],
            cwd=work / directory, capture_output=True, text=True, timeout=10,
        )
        report[agent] = {
            "passed": result.returncode == 0,
            "elapsed_s": int(elapsed),
            "agent_exit": int(status),
        }
        if result.returncode:
            print(result.stderr, file=sys.stderr)
    print(json.dumps(report))
    sys.exit(0 if all(agent["passed"] for agent in report.values()) else 1)


if __name__ == "__main__":
    main()

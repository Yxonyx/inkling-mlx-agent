"""Pin, audit and split the public Harbor task collections used for Pi adaptation.

Every source is read at an exact revision. The audit records licenses, task counts and
per-task file inventories, then checks train/eval overlap at three levels: task id,
upstream repository and instruction text (word 5-gram Jaccard). Evaluation-only
collections are never written into a training manifest.

  .venv-tinker/bin/python -m adaptation.harbor_data audit
  .venv-tinker/bin/python -m adaptation.harbor_data dev-set
"""

import argparse
import collections
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs" / "datasets"
RESULTS = ROOT / "results" / "adaptation"

SOURCES = {
    "scale800": {"repo": "harness-hub/scale800-harbor", "revision": "65940b91855c085398f24c010068da6f724074d5",
                 "role": "train", "kind": "hf"},
    "scale800_meta": {"repo": "harness-hub/agentic-coding-envs", "revision": "5d320e93b7832c6c66e6a5acd8403fb1a3c50bc6",
                      "role": "metadata", "kind": "hf", "file": "views/scale800_v1/train.jsonl"},
    "termgrade": {"repo": "harness-hub/termgrade-train-harbor", "revision": "151205e3714d8d1cec6b7d5770d344853e11d7aa",
                  "role": "train", "kind": "hf"},
    "terminal_bench_2_1": {"repo": "harborframework/terminal-bench-2.1", "revision": "3e235dff6880252a587fa479c09bcb1e16edf2eb",
                           "role": "final_eval", "kind": "hf"},
    "swe_bench_pro_hard51": {"repo": "scaleapi/SWE-bench_Pro-os", "revision": "66f92766bba642462d4bbe5479e83f91f9211862",
                             "role": "final_eval", "kind": "github", "ids": "v2/hard51_ids.txt", "tasks": "v2/tasks"},
}
WORD = re.compile(r"[a-z0-9_]+")


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else json.dumps(data, sort_keys=True).encode()).hexdigest()


def hf_tasks(name):
    """Return {task: {"files": {path: blob_id}, "instruction": text, "toml": text}} at the pinned revision."""
    from huggingface_hub import HfApi, snapshot_download

    source = SOURCES[name]
    api = HfApi()
    info = api.dataset_info(source["repo"], revision=source["revision"])
    if info.sha != source["revision"]:
        raise ValueError(f"{source['repo']} resolved to {info.sha}")
    tasks = collections.defaultdict(dict)
    for entry in api.list_repo_tree(source["repo"], repo_type="dataset", revision=source["revision"],
                                    path_in_repo="tasks", recursive=True):
        parts = entry.path.split("/")
        if len(parts) >= 3 and hasattr(entry, "blob_id"):
            tasks[parts[1]][entry.path] = entry.blob_id
    local = snapshot_download(source["repo"], repo_type="dataset", revision=source["revision"],
                              allow_patterns=["tasks/*/instruction.md", "tasks/*/task.toml", "README.md", "LICENSE"],
                              local_dir=CACHE / f"{name}@{source['revision'][:12]}")
    out = {}
    for task, files in tasks.items():
        base = Path(local) / "tasks" / task
        out[task] = {
            "files": files,
            "instruction": (base / "instruction.md").read_text(errors="replace") if (base / "instruction.md").is_file() else "",
            "toml": (base / "task.toml").read_text() if (base / "task.toml").is_file() else "",
        }
    card = (info.card_data or {}).get("license") if info.card_data else None
    return out, {"license": card, "private": info.private, "gated": info.gated}


def github_hard51():
    source = SOURCES["swe_bench_pro_hard51"]
    repo, ref = source["repo"], source["revision"]

    def content(path):
        raw = subprocess.run(["gh", "api", f"repos/{repo}/contents/{path}?ref={ref}", "-H",
                              "Accept: application/vnd.github.raw"], capture_output=True, check=True)
        return raw.stdout

    ids = content(source["ids"]).decode().split()
    tasks = {}
    for task in ids:
        tasks[task] = {"instruction": content(f"{source['tasks']}/{task}/instruction.md").decode(errors="replace"),
                       "toml": content(f"{source['tasks']}/{task}/task.toml").decode(), "files": {}}
    license_id = json.loads(subprocess.run(["gh", "api", f"repos/{repo}"], capture_output=True, check=True).stdout)["license"]["spdx_id"]
    return tasks, {"license": license_id, "private": False, "gated": False}


def scale800_meta():
    from huggingface_hub import hf_hub_download

    source = SOURCES["scale800_meta"]
    path = hf_hub_download(source["repo"], source["file"], repo_type="dataset", revision=source["revision"],
                           local_dir=CACHE / f"scale800_meta@{source['revision'][:12]}")
    return {row["task_id"]: row for row in map(json.loads, Path(path).read_text().splitlines())}


def upstream(task_id, meta=None):
    """owner/name of the source repository, lower-case, or None when unknown."""
    if meta and meta.get("upstream_repo"):
        return meta["upstream_repo"].lower().removeprefix("https://github.com/")
    match = re.match(r"^(?:instance_)?([A-Za-z0-9-]+)__([A-Za-z0-9_.-]+?)(?:\.[0-9a-f]{8})?(?:[.-]|$)", task_id)
    return f"{match.group(1)}/{match.group(2)}".lower() if match else None


def shingles(text, n=5):
    words = WORD.findall(text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def near_duplicates(train, evaluation, threshold=0.5):
    index = collections.defaultdict(set)
    train_sets = {}
    for key, text in train.items():
        train_sets[key] = shingles(text)
        for gram in train_sets[key]:
            index[gram].add(key)
    hits = []
    for key, text in evaluation.items():
        grams = shingles(text)
        counts = collections.Counter(other for gram in grams for other in index.get(gram, ()))
        for other, shared in counts.items():
            union = len(grams | train_sets[other])
            if union and shared / union >= threshold:
                hits.append({"eval": key, "train": other, "jaccard": round(shared / union, 3)})
    return hits


def audit():
    collections_, meta = {}, scale800_meta()
    for name in ("scale800", "termgrade", "terminal_bench_2_1"):
        collections_[name] = hf_tasks(name)
    collections_["swe_bench_pro_hard51"] = github_hard51()
    report = {"sources": {}, "overlap": {}}
    for name, (tasks, info) in collections_.items():
        source = SOURCES[name]
        images = collections.Counter()
        for task in tasks.values():
            match = re.search(r'docker_image\s*=\s*"([^"]+)"', task["toml"])
            images["prebuilt" if match else "build"] += 1
        report["sources"][name] = {
            **{k: source[k] for k in ("repo", "revision", "role")}, **info, "tasks": len(tasks),
            "images": dict(images),
            "inventory_sha256": sha256({task: value["files"] or sha256(value["instruction"] + value["toml"])
                                        for task, value in sorted(tasks.items())}),
        }
    rows = list(meta.values())
    report["sources"]["scale800"]["task_ids"] = sorted(collections_["scale800"][0])
    report["sources"]["scale800"]["metadata"] = {
        "view_rows": len(rows), "tasks_in_harbor_repo": sum(task in collections_["scale800"][0] for task in meta),
        "source_licenses": dict(collections.Counter(r["source_license_spdx"] for r in rows)),
        "task_bundle_licenses": dict(collections.Counter(str(r["task_bundle_license_spdx"]) for r in rows)),
        "usage": dict(collections.Counter(r["usage"] for r in rows)),
        "training_admissible": dict(collections.Counter(str(r["training_admissible"]) for r in rows)),
        "verified": dict(collections.Counter(str(r["verified"]) for r in rows)),
        "semantic_nonoverlap_established": dict(collections.Counter(str(r["semantic_nonoverlap_established"]) for r in rows)),
        "qwen_scored": sum(r["qwen_status"] == "scored" for r in rows),
        "qwen_always": sum(r["qwen_status"] == "scored" and r["qwen_solve_rate"] == 1 for r in rows),
        "qwen_mixed": sum(r["qwen_status"] == "scored" and 0 < r["qwen_solve_rate"] < 1 for r in rows),
        "qwen_never": sum(r["qwen_status"] == "scored" and r["qwen_solve_rate"] == 0 for r in rows),
    }
    train = {f"{name}/{task}": value["instruction"] for name in ("scale800", "termgrade")
             for task, value in collections_[name][0].items()}
    train_repos = {upstream(task, meta.get(task)) for task in collections_["scale800"][0]} - {None}
    for name in ("terminal_bench_2_1", "swe_bench_pro_hard51"):
        tasks = collections_[name][0]
        evaluation = {f"{name}/{task}": value["instruction"] for task, value in tasks.items()}
        eval_repos = {upstream(task) for task in tasks} - {None} if name == "swe_bench_pro_hard51" else set()
        report["overlap"][name] = {
            "task_id": sorted(set(tasks) & {key.split("/", 1)[1] for key in train}),
            "upstream_repo": sorted(train_repos & eval_repos),
            "instruction_jaccard_ge_0.5": near_duplicates(train, evaluation),
        }
    report["overlap"]["scale800_upstream_repos"] = sorted(train_repos)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "datasets.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def dev_set(size=24, seed=0, per_repo=5, quota=(("python", 14), ("go", 10))):
    """Freeze a repository-disjoint, language-stratified scale800 development set.

    Whole repositories are held out, so none of their tasks may enter a training manifest.
    Within a repository, tasks with a measured Qwen solve rate nearest 0.5 come first;
    the Qwen rate is only a prior until Inkling is measured.
    """
    import random

    meta = scale800_meta()
    harbor = set(json.loads((RESULTS / "datasets.json").read_text())["sources"]["scale800"]["task_ids"])
    by_repo = collections.defaultdict(list)
    for task, row in sorted(meta.items()):
        if task in harbor:
            by_repo[upstream(task, row)].append(row)
    rng = random.Random(seed)

    def prior(row):
        rate = row["qwen_solve_rate"] if row["qwen_status"] == "scored" else None
        return (rate is None, abs((rate or 0) - 0.5), rng.random())

    chosen, held = [], []
    for language, target in quota:
        repos = sorted(repo for repo, rows in by_repo.items() if rows[0]["language"] == language)
        rng.shuffle(repos)
        # Prefer mid-sized repositories so a held-out repo does not drain the train pool.
        repos.sort(key=lambda repo: abs(len(by_repo[repo]) - 20))
        taken = 0
        for repo in repos:
            if taken >= target:
                break
            rows = sorted(by_repo[repo], key=prior)[:min(per_repo, target - taken)]
            held.append(repo)
            chosen.extend(rows)
            taken += len(rows)
    if len(chosen) != size:
        raise ValueError(f"Dev set has {len(chosen)} tasks, expected {size}")
    manifest = {
        "source": SOURCES["scale800"], "seed": seed, "quota": dict(quota), "held_out_repos": sorted(held),
        "held_out_train_tasks": sum(len(by_repo[repo]) for repo in held) - len(chosen),
        "tasks": sorted(row["task_id"] for row in chosen),
        "rule": "Development set only. Every task from held_out_repos is excluded from training and tuning manifests.",
    }
    manifest["sha256"] = sha256(manifest)
    (RESULTS / "dev-set.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def train_pool(size=140, seed=0, per_repo=10, quota=(("python", 84), ("go", 56))):
    """Freeze the SFT-1 collection pool: scale800 Harbor tasks outside every dev-set repository.

    Tasks with a Qwen solve rate above zero come first (more likely to yield verified passes),
    then unscored ones; at most per_repo tasks per repository keep the pool diverse.
    """
    import random

    meta = scale800_meta()
    report = json.loads((RESULTS / "datasets.json").read_text())
    harbor = set(report["sources"]["scale800"]["task_ids"])
    dev = json.loads((RESULTS / "dev-set.json").read_text())
    held = set(dev["held_out_repos"])
    rng = random.Random(seed)
    by_repo = collections.defaultdict(list)
    for task, row in sorted(meta.items()):
        if task in harbor and upstream(task, row) not in held:
            by_repo[upstream(task, row)].append(row)

    def prior(row):
        rate = row["qwen_solve_rate"] if row["qwen_status"] == "scored" else None
        return (0 if rate and rate > 0 else 1 if rate is None else 2, rng.random())

    chosen = []
    for language, target in quota:
        candidates = []
        for repo, rows in sorted(by_repo.items()):
            if rows[0]["language"] == language:
                candidates.extend(sorted(rows, key=prior)[:per_repo])
        candidates.sort(key=prior)
        chosen.extend(candidates[:target])
    tasks = sorted(row["task_id"] for row in chosen)
    repos = {upstream(task, meta[task]) for task in tasks}
    overlap = {"dev_tasks": sorted(set(tasks) & set(dev["tasks"])), "dev_repos": sorted(repos & held),
               "final_eval": {name: v["task_id"] + v["upstream_repo"] + [h["train"] for h in v["instruction_jaccard_ge_0.5"]]
                              for name, v in report["overlap"].items() if isinstance(v, dict)}}
    if overlap["dev_tasks"] or overlap["dev_repos"] or any(overlap["final_eval"].values()):
        raise ValueError(f"Train pool overlaps evaluation data: {overlap}")
    manifest = {"source": SOURCES["scale800"], "seed": seed, "quota": dict(quota), "per_repo": per_repo,
                "repos": len(repos), "excluded_dev_repos": sorted(held), "overlap": overlap, "tasks": tasks,
                "rule": "SFT-1 collection pool. Never mix with dev-set.json or the final benchmarks."}
    manifest["sha256"] = sha256(manifest)
    (RESULTS / "sft1-pool.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["audit", "dev-set", "train-pool"])
    args = parser.parse_args()
    result = audit() if args.command == "audit" else dev_set() if args.command == "dev-set" else train_pool()
    summary = {k: v for k, v in result.items() if k != "tasks"} if args.command != "audit" else {
        "sources": {k: {x: v[x] for x in ("revision", "role", "license", "tasks")} for k, v in result["sources"].items()},
        "overlap": {k: {x: len(y) for x, y in v.items()} for k, v in result["overlap"].items() if isinstance(v, dict)},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

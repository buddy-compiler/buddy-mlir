#!/usr/bin/env python3
"""Subprocess-isolated unique-scope numeric coverage runner.

Each op runs in its own worker process. SIGSEGV/SIGABRT from LLVM JIT
only kills the worker, the parent records it as a crash and moves on.
Output is JSONL written incrementally; re-running resumes from it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
WORKER = THIS_DIR / "aten_coverage_worker.py"


def _default_out_path() -> Path:
    build_dir = os.environ.get("BUDDY_MLIR_BUILD_DIR")
    base = Path(build_dir) if build_dir else REPO_ROOT / "build"
    return base / "tests" / "oc_unique_numeric_isolated.jsonl"


DEFAULT_OUT = _default_out_path()
WORKER_TIMEOUT_S = 120

NUMERIC_OVERRIDES = {
    "add": "Tensor",
    "copysign": "Tensor",
    "div": "Tensor",
    "eq": "Tensor",
    "fmod": "Tensor",
    "frexp": "Tensor",
    "ge": "Tensor",
    "gt": "Tensor",
    "le": "Tensor",
    "lt": "Tensor",
    "mul": "Tensor",
    "ne": "Tensor",
    "normal": "Tensor_Tensor",
    "norm": "ScalarOpt_dim",
    "rand": "default",
    "randn": "default",
    "remainder": "Tensor",
    "repeat_interleave": "Tensor",
    "select": "int",
    "sub": "Tensor",
    "where": "self",
}
PRIORITY = ("default", "Tensor", "self", "Tensor_out", "out")


def _pick_overload(overloads: list[str]) -> str:
    for want in PRIORITY:
        if want in overloads:
            return want
    return overloads[0] if overloads else ""


def _build_op_list() -> list[str]:
    catalog = json.loads((THIS_DIR / "aten_coverage_catalog.json").read_text())
    unique_ops = [
        it["op"]
        for it in json.loads(
            (THIS_DIR / "aten_coverage_unique_ops.json").read_text()
        )
    ]
    by_op: dict[str, list[str]] = defaultdict(list)
    for e in catalog:
        by_op[e["op"]].append(e["overload"])
    selected: list[str] = []
    for op in unique_ops:
        ov = by_op.get(op, [])
        if not ov:
            continue
        forced = NUMERIC_OVERRIDES.get(op)
        if forced and forced in ov:
            selected.append(f"{op}.{forced}")
        else:
            selected.append(f"{op}.{_pick_overload(ov)}")
    return selected


def _load_done(out_path: Path) -> set[str]:
    done: set[str] = set()
    if not out_path.exists():
        return done
    for line in out_path.read_text().splitlines():
        with contextlib.suppress(Exception):
            done.add(json.loads(line)["name"])
    return done


def _run_worker(name: str) -> dict:
    env = dict(os.environ)
    # Bootstrap is done inside the worker, but propagate the build dirs
    # if the caller already exported them so they win over the defaults.
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), name],
        env=env,
        capture_output=True,
        timeout=WORKER_TIMEOUT_S,
    )
    last = ""
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            last = line
    if last:
        with contextlib.suppress(Exception):
            return json.loads(last)
    # Worker died before printing a JSON record.
    sig = -proc.returncode if proc.returncode < 0 else proc.returncode
    reason = f"worker_died:exit={sig}"
    err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    if err:
        reason += f":{err[-1][:120]}"
    return {"name": name, "status": "fail", "reason": reason}


def _summarize(out_path: Path) -> None:
    recs = [
        json.loads(line)
        for line in out_path.read_text().splitlines()
        if line.strip()
    ]
    s = Counter(r["status"] for r in recs)
    total = len(recs)
    print()
    print("=== UNIQUE-SCOPE NUMERIC COVERAGE (subprocess-isolated) ===")
    print(
        f"total={total}  pass={s['pass']}  fail={s['fail']}  skip={s['skip']}"
    )
    if total:
        print(f"pass/total = {s['pass']}/{total} = {s['pass'] / total:.4f}")


def _run_worker_safe(name: str) -> dict:
    try:
        return _run_worker(name)
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "status": "fail",
            "reason": f"worker_timeout:{WORKER_TIMEOUT_S}s",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most N ops (0 = all). Useful for smoke testing.",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers (default: min(8, nproc)).",
    )
    args = parser.parse_args()

    if args.fresh and args.out.exists():
        args.out.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    selected = _build_op_list()
    if args.limit:
        selected = selected[: args.limit]
    done = _load_done(args.out)
    todo = [n for n in selected if n not in done]
    print(
        f"[plan] total={len(selected)} resume_skip={len(done)} "
        f"todo={len(todo)} jobs={args.jobs} out={args.out}",
        flush=True,
    )

    if args.jobs <= 1:
        with args.out.open("a") as fout:
            for i, name in enumerate(todo, 1):
                print(f"[{i}/{len(todo)}] RUN {name}", flush=True)
                rec = _run_worker_safe(name)
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
    else:
        with (
            args.out.open("a") as fout,
            ProcessPoolExecutor(max_workers=args.jobs) as pool,
        ):
            futures = {pool.submit(_run_worker_safe, n): n for n in todo}
            for completed, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
                print(
                    f"[{completed}/{len(todo)}] {rec['status'].upper()} "
                    f"{rec['name']}",
                    flush=True,
                )

    print("DONE", flush=True)
    _summarize(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

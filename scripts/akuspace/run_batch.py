#!/usr/bin/env python3
"""Queue AKUSPACE API workflows sequentially and print one compact summary."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from comfy_api import (
    ComfyClient,
    WorkflowError,
    apply_overrides,
    history_error,
    output_refs,
    read_workflow,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW_DIR = REPO_ROOT / "workflows" / "akuspace"


def select_workflows(directory: Path, patterns: list[str]) -> list[Path]:
    paths = sorted(directory.glob("*.json"))
    if patterns:
        paths = [path for path in paths if any(p in path.stem for p in patterns)]
    if not paths:
        detail = f" matching {patterns}" if patterns else ""
        raise WorkflowError(f"no JSON workflows in {directory}{detail}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="*", help="optional filename substrings")
    parser.add_argument("--workflow-dir", type=Path, default=DEFAULT_WORKFLOW_DIR)
    parser.add_argument(
        "--server",
        default=os.environ.get("COMFY_URL", "http://127.0.0.1:8189"),
    )
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per graph")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NODE.INPUT=VALUE",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate without a server")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-free", action="store_true", help="do not unload between graphs")
    args = parser.parse_args()

    try:
        paths = select_workflows(args.workflow_dir, args.patterns)
        prepared = []
        for path in paths:
            graph = read_workflow(path)
            apply_overrides(graph, args.overrides)
            prepared.append((path, graph))
    except WorkflowError as exc:
        parser.error(str(exc))

    if args.dry_run:
        for path, graph in prepared:
            print(f"OK  {path.name:<42} {len(graph):>3} nodes")
        print(f"\n{len(prepared)} workflow(s) validated; nothing queued")
        return 0

    client = ComfyClient(args.server)
    try:
        client.check()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 2

    results: list[tuple[str, str, int, list[str]]] = []
    for path, graph in prepared:
        print(f"\n=== {path.stem} ===")
        started = time.monotonic()
        try:
            if not args.no_free:
                client.free_models()
            prompt_id = client.queue(graph)
            print(f"queued {prompt_id}")
            history = client.wait(prompt_id, args.timeout)
            problem = history_error(history)
            if problem:
                raise RuntimeError(problem)
            refs = output_refs(history)
            results.append(("OK", path.stem, round(time.monotonic() - started), refs))
            for ref in refs:
                print(f"  {ref}")
        except (RuntimeError, TimeoutError) as exc:
            results.append(("FAIL", path.stem, round(time.monotonic() - started), [str(exc)]))
            print(f"FAILED: {exc}")
            if args.fail_fast:
                break

    print("\nSTATUS  SECONDS  WORKFLOW")
    for status, name, seconds, _ in results:
        print(f"{status:<6}  {seconds:>7}  {name}")
    return 1 if any(status == "FAIL" for status, *_ in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run a controlled AKUSPACE LoRA-on versus LoRA-off comparison."""

from __future__ import annotations

import argparse
import copy
import os
import time
from pathlib import Path

from comfy_api import (
    ComfyClient,
    WorkflowError,
    apply_overrides,
    find_nodes,
    history_error,
    output_refs,
    read_workflow,
    with_output_suffix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW = (
    REPO_ROOT / "workflows" / "akuspace" / "akuspace_t2v_native_onepass.json"
)


def build_arms(graph: dict, on_strength: float) -> tuple[str, dict, dict]:
    if on_strength <= 0:
        raise WorkflowError("--on-strength must be greater than zero")
    lora_nodes = find_nodes(graph, "LoraLoaderModelOnly")
    if len(lora_nodes) != 1:
        raise WorkflowError(
            f"A/B needs exactly one LoraLoaderModelOnly; found {len(lora_nodes)}"
        )
    lora_id = lora_nodes[0]

    enabled = with_output_suffix(graph, "lora_on")
    disabled = with_output_suffix(graph, "lora_off")
    enabled[lora_id]["inputs"]["strength_model"] = on_strength
    disabled[lora_id]["inputs"]["strength_model"] = 0.0

    left = copy.deepcopy(enabled)
    right = copy.deepcopy(disabled)
    for candidate in (left, right):
        for node in candidate.values():
            if "filename_prefix" in node["inputs"]:
                node["inputs"]["filename_prefix"] = "<arm>"
    differing = [node_id for node_id in left if left[node_id] != right[node_id]]
    if differing != [lora_id]:
        raise WorkflowError(f"A/B changed more than the LoRA node: {differing}")
    return lora_id, enabled, disabled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument(
        "--server",
        default=os.environ.get("COMFY_URL", "http://127.0.0.1:8189"),
    )
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per arm")
    parser.add_argument("--on-strength", type=float, default=1.0)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NODE.INPUT=VALUE",
    )
    parser.add_argument("--dry-run", action="store_true", help="validate without a server")
    parser.add_argument("--no-free", action="store_true", help="do not unload between arms")
    args = parser.parse_args()

    try:
        graph = read_workflow(args.workflow)
        apply_overrides(graph, args.overrides)
        lora_id, enabled, disabled = build_arms(graph, args.on_strength)
    except WorkflowError as exc:
        parser.error(str(exc))

    print(f"workflow: {args.workflow}")
    print(f"isolated node: {lora_id} (LoraLoaderModelOnly)")
    print(f"arms: strength {args.on_strength:g} vs 0; seeds and all other inputs unchanged")
    if args.dry_run:
        print("OK — both arms validated; nothing queued")
        return 0

    client = ComfyClient(args.server)
    try:
        client.check()
        for label, arm in (("lora_on", enabled), ("lora_off", disabled)):
            print(f"\n=== {label} ===")
            if not args.no_free:
                client.free_models()
            started = time.monotonic()
            prompt_id = client.queue(arm)
            print(f"queued {prompt_id}")
            history = client.wait(prompt_id, args.timeout)
            problem = history_error(history)
            if problem:
                raise RuntimeError(problem)
            print(f"completed in {round(time.monotonic() - started)}s")
            for ref in output_refs(history):
                print(f"  {ref}")
    except (RuntimeError, TimeoutError) as exc:
        print(f"FAILED: {exc}")
        return 1

    print("\nCompare the _lora_on and _lora_off outputs with the same seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Statically verify the public AKUSPACE API workflows before shipping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from comfy_api import find_nodes, is_link, read_workflow, validate_workflow

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW_DIR = REPO_ROOT / "workflows" / "akuspace"
EXPECTED_FILES = {
    "akuspace_a2a_treat_recording.json",
    "akuspace_t2v_2stage_hq.json",
    "akuspace_t2v_native_onepass.json",
}
AKUSPACE_CONDITIONING = {"Koshi_AKUSPACEPrompt", "Koshi_AKUSPACETextEncode"}
OUTPUT_CLASSES = {
    "PreviewAny",
    "SaveAudio",
    "SaveAudioAdvanced",
    "SaveVideo",
    "VHS_VideoCombine",
}


def reachable_nodes(graph: dict) -> tuple[set[str], list[str]]:
    outputs = [
        node_id for node_id, node in graph.items() if node["class_type"] in OUTPUT_CLASSES
    ]
    seen: set[str] = set()
    stack = outputs[:]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for value in graph[node_id]["inputs"].values():
            if is_link(value):
                stack.append(value[0])
    return seen, outputs


def upstream_nodes(graph: dict, start_id: str) -> set[str]:
    seen: set[str] = set()
    stack = [start_id]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        for value in graph[node_id]["inputs"].values():
            if is_link(value):
                stack.append(value[0])
    return seen


def verify_graph(path: Path, graph: dict) -> tuple[list[str], list[str]]:
    errors = validate_workflow(graph)
    notes: list[str] = []
    classes = {node["class_type"] for node in graph.values()}

    conditioning = classes & AKUSPACE_CONDITIONING
    if not conditioning:
        errors.append("missing an AKUSPACE Prompt or Text Encode node")

    lora_nodes = find_nodes(graph, "LoraLoaderModelOnly")
    if len(lora_nodes) != 1:
        errors.append(f"expected one LoraLoaderModelOnly, found {len(lora_nodes)}")
    elif "akuspace" not in str(graph[lora_nodes[0]]["inputs"].get("lora_name", "")).lower():
        errors.append("LoRA loader does not name an AKUSPACE checkpoint")

    aligned = find_nodes(graph, "AKUSPACEReferenceAudioAligned")
    if len(aligned) != 1:
        errors.append(f"expected one aligned reference node, found {len(aligned)}")
    elif len(lora_nodes) == 1:
        model = graph[aligned[0]]["inputs"].get("model")
        if not is_link(model) or lora_nodes[0] not in upstream_nodes(graph, model[0]):
            errors.append("aligned reference node does not receive the LoRA-patched model")

    if "LTXVEmptyLatentAudio" not in classes:
        errors.append("audio target is not empty; a pinned target cannot exercise the LoRA")
    if "SetLatentNoiseMask" in classes:
        errors.append("audio target is pinned by SetLatentNoiseMask")

    for node_id in find_nodes(graph, "LTXVDualCFGGuider"):
        inputs = graph[node_id]["inputs"]
        for field in ("video_cfg", "audio_cfg"):
            if inputs.get(field) != 1.0:
                errors.append(f"node {node_id}.{field} must stay at 1.0 for distilled LTX")

    seen, outputs = reachable_nodes(graph)
    if not outputs:
        errors.append("no supported output node")
    orphaned = sorted(
        set(graph) - seen,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if orphaned:
        errors.append(f"nodes not connected to an output: {', '.join(orphaned)}")

    notes.append(
        f"{len(graph)} nodes; {len(outputs)} output node(s); "
        f"conditioning={','.join(sorted(conditioning)) or 'none'}"
    )
    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-dir", type=Path, default=DEFAULT_WORKFLOW_DIR)
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()

    paths = sorted(args.workflow_dir.glob("*.json"))
    missing = sorted(EXPECTED_FILES - {path.name for path in paths})
    report: dict[str, dict[str, list[str]]] = {}
    if missing:
        report["<catalogue>"] = {"errors": [f"missing: {', '.join(missing)}"], "notes": []}

    for path in paths:
        try:
            graph = read_workflow(path)
            errors, notes = verify_graph(path, graph)
        except ValueError as exc:
            errors, notes = [str(exc)], []
        report[path.name] = {"errors": errors, "notes": notes}

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for name, result in report.items():
            status = "FAIL" if result["errors"] else "OK"
            print(f"{status:<4} {name}")
            for note in result["notes"]:
                print(f"     {note}")
            for error in result["errors"]:
                print(f"     ERROR: {error}")

    failures = sum(bool(result["errors"]) for result in report.values())
    if failures:
        print(f"\nFAILED — {failures} workflow group(s) need attention")
        return 1
    print(f"\nOK — {len(paths)} public AKUSPACE workflows verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check the rooms-chain graph against a ComfyUI /object_info dump. Never queues.

    curl -s http://127.0.0.1:8189/object_info > /tmp/object_info.json
    python3 scripts/akuspace/check_rooms_chain.py /tmp/object_info.json

Checks, for every node class the server knows:
  1. SPEC (rooms_chain_spec.py) matches object_info: socket names and types,
     widget names and order, output types.
  2. The committed API graph: class exists, every required input is present, no
     unknown inputs, link types match, combo values are allowed, numbers in range.
  3. The same for four variants with bypassed groups switched on (first frames,
     depth/canny, stage 2, everything), so enabling a group cannot leave a socket
     dangling. Classes the server does not know (ComfyUI-LTXVideo, World Builder
     on a box without them) are checked against SPEC only and listed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import build_rooms_chain as b
from build_rooms_chain import API_PATH, build
from rooms_chain_spec import BYPASS, SOURCE_VALIDATED, SPEC, UI_ONLY, to_api

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"}
# File pickers list what this server has on disk; a missing file is a note, not an error.
FILE_INPUTS = {("UNETLoader", "unet_name"), ("CLIPLoader", "clip_name"), ("VAELoader", "vae_name"),
               ("LoraLoaderModelOnly", "lora_name"), ("LatentUpscaleModelLoader", "model_name"),
               ("LoadAudio", "audio"), ("LoadImage", "image")}
MISSING_FILES: set[str] = set()


def is_widget(spec) -> bool:
    typ, opts = spec[0], (spec[1] if len(spec) > 1 else {})
    if isinstance(typ, list):
        return True
    return all(t in WIDGET_TYPES for t in str(typ).split(",")) and not opts.get("forceInput")


def type_ok(have: str, want) -> bool:
    if isinstance(want, list):
        want = "COMBO"
    return want == "*" or have == want or have in str(want).split(",") or (have == "INT" and want == "FLOAT")


def check_spec(info: dict) -> list[str]:
    errors = []
    for cls, (sockets, widgets, outputs) in SPEC.items():
        if cls in UI_ONLY or cls not in info:
            continue
        node = info[cls]
        order = node.get("input_order", {})
        specs = {**node["input"].get("required", {}), **node["input"].get("optional", {})}
        names = order.get("required", []) + order.get("optional", [])
        want_sockets = [(n, specs[n][0]) for n in names if not is_widget(specs[n])]
        want_widgets = [n for n in names if is_widget(specs[n])]
        have_sockets = [(n, t) for n, t, _ in sockets]
        if [n for n, _ in want_sockets] != [n for n, _ in have_sockets]:
            errors.append(f"SPEC {cls}: sockets {have_sockets} != server {want_sockets}")
        for (n, t), (_, w) in zip(have_sockets, want_sockets):
            if not type_ok(t, w) and t != w:
                errors.append(f"SPEC {cls}.{n}: type {t} != server {w}")
        if [n for n, _ in widgets] != want_widgets:
            errors.append(f"SPEC {cls}: widgets {[n for n, _ in widgets]} != server {want_widgets}")
        server_out = node["output"]
        if len(server_out) != len(outputs) and not (cls == "SaveVideo" or cls == "SaveAudioAdvanced"):
            errors.append(f"SPEC {cls}: {len(outputs)} outputs != server {len(server_out)}")
        for (t, _), w in zip(outputs, server_out):
            if t != w and not (t == "COMBO" and isinstance(w, list)):
                errors.append(f"SPEC {cls}: output {t} != server {w}")
    return errors


def check_value(where: str, spec, value) -> list[str]:
    typ, opts = spec[0], (spec[1] if len(spec) > 1 else {})
    if isinstance(typ, list) or typ == "COMBO":
        options = typ if isinstance(typ, list) else opts.get("options", [])
        if options and value not in options:
            return [f"{where}: {value!r} not in combo options"]
    elif typ == "COMFY_DYNAMICCOMBO_V3":
        keys = [o["key"] for o in opts.get("options", [])]
        if value not in keys:
            return [f"{where}: {value!r} not in {keys}"]
    elif typ in ("INT", "FLOAT"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return [f"{where}: {value!r} is not a number"]
        if "min" in opts and value < opts["min"] or "max" in opts and value > opts["max"]:
            return [f"{where}: {value} outside [{opts.get('min')}, {opts.get('max')}]"]
    elif typ == "BOOLEAN" and not isinstance(value, bool):
        return [f"{where}: {value!r} is not a boolean"]
    return []


def out_type(info: dict, api: dict, src: str, slot: int):
    cls = api[src]["class_type"]
    if cls in info:
        outs = info[cls]["output"]
        return outs[slot] if slot < len(outs) else None
    outs = SPEC[cls][2]
    return outs[slot][0] if slot < len(outs) else None


def check_api(info: dict, api: dict, label: str, unknown: set) -> list[str]:
    errors = []
    for nid, node in api.items():
        cls, inputs = node["class_type"], node["inputs"]
        where = f"[{label}] {nid} {cls}"
        if cls not in info:
            if cls not in SOURCE_VALIDATED:
                errors.append(f"{where}: class not on server and not source-validated")
                continue
            unknown.add(cls)
            sockets, widgets, _ = SPEC[cls]
            required = [n for n, _, opt in sockets if not opt] + [n for n, _ in widgets if n not in ("use_streaming", "tile_frames")]
            specs = {n: [t] for n, t, _ in sockets}
        else:
            req = info[cls]["input"].get("required", {})
            specs = {**req, **info[cls]["input"].get("optional", {})}
            required = list(req)
            if cls in ("SaveVideo",):
                required = [r for r in required if r != "codec"]
        for name in required:
            if name not in inputs:
                errors.append(f"{where}: missing required input {name}")
        for name, value in inputs.items():
            if cls in info and name not in specs:
                errors.append(f"{where}: unknown input {name}")
                continue
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                src, slot = value
                if src not in api:
                    errors.append(f"{where}.{name}: links to missing node {src}")
                    continue
                have = out_type(info, api, src, slot)
                want = specs[name][0] if name in specs else None
                if have is None or (want is not None and not type_ok(have if not isinstance(have, list) else "COMBO", want)):
                    errors.append(f"{where}.{name}: {api[src]['class_type']} slot {slot} gives {have}, wants {want}")
            elif (cls, name) in FILE_INPUTS:
                if check_value(where, specs[name], value):
                    MISSING_FILES.add(str(value))
            elif cls in info:
                errors += check_value(f"{where}.{name}", specs[name], value)
    return errors


def variant(nodes: dict, ids) -> dict:
    for nid in ids:
        nodes[nid].mode = 0
    return to_api(nodes)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    errors = check_spec(info)
    committed = json.loads(API_PATH.read_text(encoding="utf-8"))
    _, fresh = build()
    if committed != fresh:
        errors.append("committed API graph is stale; run build_rooms_chain.py")
    unknown: set[str] = set()
    errors += check_api(info, committed, "default", unknown)

    def fresh_nodes():
        nodes = {}
        b.shared(nodes)
        for s in (1, 2, 3):
            b.segment(nodes, s)
        b.join(nodes)
        return nodes

    first = [s * 100 + k for s in (1, 2, 3) for k in (12, 13, 14)]
    depth = [6] + [s * 100 + k for s in (1, 2, 3) for k in (17, 18, 19, 20, 21)]
    stage2 = [70, 82, 83] + [s * 100 + k for s in (1, 2, 3) for k in range(30, 36)]
    for label, ids in (("first-frame", first), ("depth", depth), ("stage2", stage2), ("all", first + depth + stage2)):
        nodes = fresh_nodes()
        api = variant(nodes, ids)
        if any(n.mode == BYPASS for n in nodes.values()) and label == "all":
            errors.append("[all] some node is still bypassed")
        errors += check_api(info, api, label, unknown)

    for e in errors:
        print("ERROR", e)
    print(f"server classes checked: {sorted(c for c in {n['class_type'] for n in committed.values()} if c in info)}")
    print(f"files not on this server (expected on the GPU box): {sorted(MISSING_FILES)}")
    print(f"source-validated only (not on this server): {sorted(unknown)}")
    print("FAIL" if errors else f"OK - API default graph {len(committed)} nodes + 4 enabled-group variants")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

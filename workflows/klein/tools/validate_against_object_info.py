"""Check both workflow files against a ComfyUI /object_info dump. Read-only.

    python3 workflows/klein/tools/validate_against_object_info.py [object_info source]

Checks: every class exists; every API input name is declared (dynamic-combo
children and autogrow slots included); every required input is present;
every link points at an existing node and output slot with a compatible type;
widget values are in range / in the combo list; UI links agree with the UI
node socket tables; UI widgets_values length matches the widget count.
Model-file combos (empty lists on a box without the weights) are reported as
"not installed here" rather than as errors.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oi  # noqa: E402

ROOT = os.path.dirname(HERE)
MODEL_WIDGETS = {"unet_name", "clip_name", "vae_name", "ckpt_name", "image"}


def compatible(out_type, in_type):
    """Mirrors comfy_execution.validation.validate_node_input (non-strict).

    MatchType on either side passes server-side, so for the ComfySwitchNode
    output the builder records the resolved type (IMAGE) in the UI file and
    the downstream sockets are all IMAGE.
    """
    if "*" in (out_type, in_type) or "COMFY_MATCHTYPE_V3" in (out_type, in_type):
        return True
    return bool(set(out_type.split(",")) & set(in_type.split(",")))


def check_value(where, name, typ, opts, value, errors, notes):
    if typ == "COMBO":
        options = opts.get("options", [])
        if value not in options:
            if name in MODEL_WIDGETS and len(options) <= 8:
                notes.append(f"{where}.{name} = {value!r}: not installed here (expected)")
            else:
                errors.append(f"{where}.{name} = {value!r} not in {options[:12]}")
    elif typ == "INT" and (not isinstance(value, int) or isinstance(value, bool)):
        errors.append(f"{where}.{name} expects INT, got {value!r}")
    elif typ == "FLOAT" and not isinstance(value, (int, float)):
        errors.append(f"{where}.{name} expects FLOAT, got {value!r}")
    elif typ == "BOOLEAN" and not isinstance(value, bool):
        errors.append(f"{where}.{name} expects BOOLEAN, got {value!r}")
    elif typ == "STRING" and not isinstance(value, str):
        errors.append(f"{where}.{name} expects STRING, got {value!r}")
    if typ in ("INT", "FLOAT") and isinstance(value, (int, float)):
        if "min" in opts and value < opts["min"] or "max" in opts and value > opts["max"]:
            errors.append(f"{where}.{name} = {value} outside [{opts.get('min')}, {opts.get('max')}]")


def validate_api(api, info, errors, notes):
    for nid, node in api.items():
        where = f"api[{nid}] {node['class_type']}"
        spec = info.get(node["class_type"])
        if spec is None:
            errors.append(f"{where}: class not in object_info")
            continue
        declared = {}
        for name, typ, opts, optional in oi.input_defs(spec):
            declared[name] = (typ, opts, optional)
            if typ == "COMFY_DYNAMICCOMBO_V3" and name in node["inputs"]:
                for child, cspec in oi.dynamic_combo_children(opts, node["inputs"][name]):
                    declared[f"{name}.{child}"] = (cspec[0], cspec[1] if len(cspec) > 1 else {}, False)
        for name, value in node["inputs"].items():
            base = name.split(".")[0]
            if name not in declared and not (base in declared and declared[base][0] == "COMFY_AUTOGROW_V3"):
                errors.append(f"{where}: unknown input {name!r}")
                continue
            typ, opts, _ = declared.get(name, declared[base])
            if isinstance(value, dict) and "__value__" in value:
                continue  # list-valued widget, unwrapped by the server
            if isinstance(value, list) and not (len(value) == 2 and isinstance(value[0], str)):
                errors.append(f"{where}.{name}: bare list is read as a link; wrap it in __value__")
                continue
            if isinstance(value, list):
                src = api.get(value[0])
                if src is None:
                    errors.append(f"{where}.{name}: link to missing node {value[0]}")
                    continue
                outs = info[src["class_type"]]["output"]
                if value[1] >= len(outs):
                    errors.append(f"{where}.{name}: source {value[0]} has no output slot {value[1]}")
                    continue
                if typ == "COMFY_AUTOGROW_V3":
                    typ = "IMAGE"
                if not compatible(outs[value[1]], typ):
                    errors.append(f"{where}.{name}: {outs[value[1]]} -> {typ} mismatch")
            elif typ == "COMFY_DYNAMICCOMBO_V3":
                keys = [o["key"] for o in opts.get("options", [])]
                if value not in keys:
                    errors.append(f"{where}.{name} = {value!r} not in {keys}")
            else:
                check_value(where, name, typ, opts, value, errors, notes)
        for name, (typ, opts, optional) in declared.items():
            if not optional and "." not in name and name not in node["inputs"]:
                if typ == "COMFY_AUTOGROW_V3" and any(k.startswith(name + ".") for k in node["inputs"]):
                    continue
                errors.append(f"{where}: required input {name!r} missing")


def validate_ui(ui, info, errors):
    nodes = {n["id"]: n for n in ui["nodes"]}
    for link in ui["links"]:
        lid, src, sslot, dst, dslot, ltype = link
        if src not in nodes or dst not in nodes:
            errors.append(f"ui link {lid}: missing node")
            continue
        out = nodes[src]["outputs"][sslot]
        inp = nodes[dst]["inputs"][dslot]
        if lid not in out["links"] or inp["link"] != lid:
            errors.append(f"ui link {lid}: socket tables disagree")
        if not compatible(ltype, inp["type"]):
            errors.append(f"ui link {lid}: {ltype} -> {nodes[dst]['type']}.{inp['name']} ({inp['type']})")
        spec = info[nodes[dst]["type"]]
        names = [n for n, *_ in oi.input_defs(spec)]
        if inp["name"].split(".")[0] not in names:
            errors.append(f"ui link {lid}: {nodes[dst]['type']} has no input {inp['name']}")
    for node in ui["nodes"]:
        if node["type"] == "MarkdownNote":
            continue
        spec = info[node["type"]]
        count = 0
        for name, typ, opts, _ in oi.input_defs(spec):
            if oi.is_widget(typ, opts):
                count += 1
                if typ == "COMFY_DYNAMICCOMBO_V3":
                    count += 1  # resize -> megapixels child
                if opts.get("control_after_generate"):
                    count += 1
                if opts.get("image_upload"):
                    count += 1
        if len(node["widgets_values"]) != count:
            errors.append(f"ui node {node['id']} {node['type']}: {len(node['widgets_values'])} widget values, expected {count}")


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8189"
    info = oi.load(source)
    with open(os.path.join(ROOT, "api", "klein_sam_region_swap_api.json")) as f:
        api = json.load(f)
    with open(os.path.join(ROOT, "klein_sam_region_swap.json")) as f:
        ui = json.load(f)
    errors, notes = [], []
    validate_api(api, info, errors, notes)
    validate_ui(ui, info, errors)
    classes = sorted({n["type"] for n in ui["nodes"]})
    print(f"classes checked ({len(classes)}): {', '.join(classes)}")
    for n in notes:
        print("note:", n)
    for e in errors:
        print("ERROR:", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()

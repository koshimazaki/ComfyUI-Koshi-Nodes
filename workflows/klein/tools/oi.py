"""Helpers for reading a ComfyUI /object_info dump."""
import json
import urllib.request

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3",
                "BOUNDING_BOXES", "IMAGECOMPARE", "COLOR"}


def load(source):
    if source.startswith("http"):
        with urllib.request.urlopen(source.rstrip("/") + "/object_info", timeout=60) as r:
            return json.load(r)
    with open(source) as f:
        return json.load(f)


def input_defs(info):
    """Yield (name, type, opts, optional) in definition order, hidden inputs skipped."""
    order = info.get("input_order") or {k: list(v) for k, v in info["input"].items()}
    for section in ("required", "optional"):
        for name in order.get(section, []):
            spec = info["input"][section][name]
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(typ, list):
                typ = "COMBO"
                opts = dict(opts, options=spec[0])
            elif typ == "COMBO":
                opts = dict(opts)
            yield name, typ, opts, section == "optional"


def is_widget(typ, opts):
    return typ in WIDGET_TYPES and not opts.get("forceInput")


def dynamic_combo_children(opts, selected):
    for option in opts.get("options", []):
        if option.get("key") == selected:
            for section in ("required", "optional"):
                for name, spec in option.get("inputs", {}).get(section, {}).items():
                    yield name, spec

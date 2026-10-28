"""Build the UI workflow and the API prompt from graph_spec.NODES.

    python3 workflows/klein/tools/build_klein_sam_region_swap.py [object_info source]

The object_info source is a running ComfyUI URL (default http://127.0.0.1:8189)
or a saved object_info JSON file. It is only read, never queued.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import graph_spec as spec  # noqa: E402
import oi  # noqa: E402
from notes import NOTES  # noqa: E402

OUT_DIR = os.path.dirname(HERE)
UI_PATH = os.path.join(OUT_DIR, "klein_sam_region_swap.json")
API_PATH = os.path.join(OUT_DIR, "api", "klein_sam_region_swap_api.json")


def socket_inputs(node, info):
    """Socket (non-widget) inputs in definition order, autogrow slots expanded."""
    links = node.get("links", {})
    for name, typ, opts, optional in oi.input_defs(info):
        if typ == "COMFY_AUTOGROW_V3":
            prefix = name + "."
            slots = sorted({k for k in links if k.startswith(prefix)}
                           | {k for k, _ in node.get("extra_sockets", []) if k.startswith(prefix)},
                           key=lambda k: int(k.rsplit("image", 1)[-1]))
            item_type = list(opts["template"]["input"].get("required", {}).values()
                             or opts["template"]["input"].get("optional", {}).values())[0][0]
            for slot in slots:
                yield slot, item_type, True
        elif not oi.is_widget(typ, opts):
            if typ == "COMFY_MATCHTYPE_V3":
                typ = node.get("link_types", {}).get(name, "*")
            yield name, node.get("link_types", {}).get(name, typ), optional


def widget_inputs(info):
    for name, typ, opts, optional in oi.input_defs(info):
        if oi.is_widget(typ, opts):
            yield name, typ


def build(object_info):
    by_id = {n["id"]: n for n in spec.NODES}
    links, ui_nodes, link_id = [], [], 0
    # First pass: input slot tables so links know their target slot.
    slot_tables = {}
    for node in spec.NODES:
        info = object_info[node["type"]]
        table = [dict(name=n, type=t, link=None, **({"shape": 7} if opt else {}))
                 for n, t, opt in socket_inputs(node, info)]
        socket_names = {e["name"] for e in table}
        for name, typ in widget_inputs(info):
            if name in node.get("links", {}) and name not in socket_names:
                table.append(dict(name=name, type=typ, widget={"name": name}, link=None))
        slot_tables[node["id"]] = table

    out_links = {n["id"]: {} for n in spec.NODES}
    for node in spec.NODES:
        table = slot_tables[node["id"]]
        for name, (src, slot) in node.get("links", {}).items():
            target_slot = next(i for i, e in enumerate(table) if e["name"] == name)
            src_info = object_info[by_id[src]["type"]]
            src_type = by_id[src].get("out_types", src_info["output"])[slot]
            link_id += 1
            links.append([link_id, src, slot, node["id"], target_slot, src_type])
            table[target_slot]["link"] = link_id
            out_links[src].setdefault(slot, []).append(link_id)

    for order, node in enumerate(spec.NODES):
        info = object_info[node["type"]]
        out_types = node.get("out_types", info["output"])
        outputs = [dict(name=info["output_name"][i], type=t,
                        links=out_links[node["id"]].get(i, []), slot_index=i)
                   for i, t in enumerate(out_types)]
        ui = dict(id=node["id"], type=node["type"], pos=node["pos"], size=node["size"],
                  flags={}, order=order, mode=node.get("mode", 0),
                  inputs=slot_tables[node["id"]], outputs=outputs,
                  properties={"Node name for S&R": node["type"]},
                  widgets_values=[v for _, v in node.get("widgets", [])])
        if node.get("title"):
            ui["title"] = node["title"]
        ui.update(node.get("colour", {}))
        ui_nodes.append(ui)

    for i, (title, pos, size, text) in enumerate(NOTES):
        ui_nodes.append(dict(id=90 + i, type="MarkdownNote", pos=pos, size=size, flags={},
                             order=len(ui_nodes), mode=0, inputs=[], outputs=[],
                             title=title, properties={}, widgets_values=[text],
                             color="#432", bgcolor="#653"))

    ui_doc = {
        "id": "7b1f3c2e-klein-sam-region-swap",
        "revision": 0,
        "last_node_id": max(n["id"] for n in ui_nodes),
        "last_link_id": link_id,
        "nodes": ui_nodes,
        "links": links,
        "groups": [dict(id=i + 1, flags={}, **g) for i, g in enumerate(spec.GROUPS)],
        "config": {},
        "extra": {"ds": {"scale": 0.45, "offset": [60, 60]}, "frontendVersion": "1.55.18"},
        "version": 0.4,
    }

    api = {}
    for node in spec.NODES:
        if node.get("api") is False:
            continue
        inputs = {}
        linked = node.get("links", {})
        for name, value in node.get("widgets", []):
            if name.startswith("~") or name in linked:
                continue
            # A bare list means a link in API format; list widgets are wrapped.
            inputs[name] = {"__value__": value} if isinstance(value, list) else value
        for name, (src, slot) in linked.items():
            if by_id[src].get("api") is False:
                continue
            inputs[name] = [str(src), slot]
        api[str(node["id"])] = {"class_type": node["type"], "inputs": inputs,
                                "_meta": {"title": node.get("title", node["type"])}}
    return ui_doc, api


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8189"
    ui_doc, api = build(oi.load(source))
    os.makedirs(os.path.dirname(API_PATH), exist_ok=True)
    with open(UI_PATH, "w") as f:
        json.dump(ui_doc, f, indent=1, ensure_ascii=False)
        f.write("\n")
    with open(API_PATH, "w") as f:
        json.dump(api, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {UI_PATH} ({len(ui_doc['nodes'])} nodes, {len(ui_doc['links'])} links)")
    print(f"wrote {API_PATH} ({len(api)} nodes)")


if __name__ == "__main__":
    main()

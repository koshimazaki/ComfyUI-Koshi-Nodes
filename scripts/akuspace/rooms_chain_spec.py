"""Node schemas and UI/API serialisation for the AKUSPACE rooms-chain graph.

SPEC holds, per node class, the socket inputs (in schema order), the widgets (in
widget order, with defaults) and the outputs. Core and Koshi entries are checked
against a live ComfyUI ``/object_info`` dump by ``check_rooms_chain.py``. The
ComfyUI-LTXVideo and World Builder entries were transcribed by hand from source
(LTXVideo 220e1d4: iclora.py; camera-director feat/world-builder-depth-acoustics:
custom_nodes/director_world_builder/__init__.py and acoustics.py).

Standard library only, so the builder runs without ComfyUI installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

S = "socket"

# class: (sockets [(name, type, optional)], widgets [(name, default)], outputs [(type, name)])
SPEC: dict[str, tuple[list, list, list]] = {
    "UNETLoader": ([], [("unet_name", ""), ("weight_dtype", "default")], [("MODEL", "MODEL")]),
    "LoraLoaderModelOnly": ([("model", "MODEL", False)], [("lora_name", ""), ("strength_model", 1.0)], [("MODEL", "MODEL")]),
    "CLIPLoader": ([], [("clip_name", ""), ("type", "ltxv"), ("device", "default")], [("CLIP", "CLIP")]),
    "VAELoader": ([], [("vae_name", "")], [("VAE", "VAE")]),
    "PrimitiveInt": ([], [("value", 0)], [("INT", "INT")]),
    "PrimitiveFloat": ([], [("value", 0.0)], [("FLOAT", "FLOAT")]),
    "PrimitiveStringMultiline": ([], [("value", "")], [("STRING", "STRING")]),
    "Koshi_AKUSPACEPrompt": (
        [("prompt", "STRING", True)],
        [("space_mode", "Room"), ("application", "Low"), ("room_preset", "medium_room"),
         ("effect_level", "low"), ("outdoor_time", "day"), ("sfx_level", "low"),
         ("source_type", "female spoken voice")],
        [("STRING", "Prompt")],
    ),
    "PreviewAny": ([("source", "*", False)], [], [("STRING", "STRING")]),
    "CLIPTextEncode": ([("clip", "CLIP", False)], [("text", "")], [("CONDITIONING", "CONDITIONING")]),
    "LTXVConditioning": (
        [("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False)],
        [("frame_rate", 24.0)],
        [("CONDITIONING", "positive"), ("CONDITIONING", "negative")],
    ),
    "LoadAudio": ([], [("audio", "")], [("AUDIO", "AUDIO")]),
    "TrimAudioDuration": ([("audio", "AUDIO", False)], [("start_index", 0.0), ("duration", 60.0)], [("AUDIO", "AUDIO")]),
    "AKUSPACEReferenceAudioAligned": (
        [("model", "MODEL", False), ("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False),
         ("reference_audio", "AUDIO", False), ("audio_vae", "VAE", False)],
        [("reference_guidance_scale", 0.0)],
        [("MODEL", "model"), ("CONDITIONING", "positive"), ("CONDITIONING", "negative")],
    ),
    "EmptyLTXVLatentVideo": ([], [("width", 768), ("height", 512), ("length", 97), ("batch_size", 1)], [("LATENT", "LATENT")]),
    "LTXVEmptyLatentAudio": (
        [("audio_vae", "VAE", False)],
        [("frames_number", 97), ("frame_rate", 25.0), ("batch_size", 1)],
        [("LATENT", "Latent")],
    ),
    "LoadImage": ([], [("image", "")], [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "LTXVPreprocess": ([("image", "IMAGE", False)], [("img_compression", 35)], [("IMAGE", "output_image")]),
    "LTXVImgToVideoInplace": (
        [("vae", "VAE", False), ("image", "IMAGE", False), ("latent", "LATENT", False)],
        [("strength", 1.0), ("bypass", False)],
        [("LATENT", "latent")],
    ),
    "ImageFromBatch": ([("image", "IMAGE", False)], [("batch_index", 0), ("length", 1)], [("IMAGE", "IMAGE")]),
    "LTXVAddGuide": (
        [("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False), ("vae", "VAE", False),
         ("latent", "LATENT", False), ("image", "IMAGE", False), ("attention_mask", "MASK", True),
         ("iclora_parameters", "IC_LORA_PARAMETERS", True)],
        [("frame_idx", 0), ("strength", 1.0)],
        [("CONDITIONING", "positive"), ("CONDITIONING", "negative"), ("LATENT", "latent")],
    ),
    "LTXVConcatAVLatent": (
        [("video_latent", "LATENT", False), ("audio_latent", "LATENT", False)], [], [("LATENT", "latent")]
    ),
    "RandomNoise": ([], [("noise_seed", 0)], [("NOISE", "NOISE")]),
    "LTXVDualCFGGuider": (
        [("model", "MODEL", False), ("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False)],
        [("video_cfg", 1.0), ("audio_cfg", 1.0)],
        [("GUIDER", "GUIDER")],
    ),
    "KSamplerSelect": ([], [("sampler_name", "euler_ancestral")], [("SAMPLER", "SAMPLER")]),
    "ManualSigmas": ([], [("sigmas", "")], [("SIGMAS", "SIGMAS")]),
    "SamplerCustomAdvanced": (
        [("noise", "NOISE", False), ("guider", "GUIDER", False), ("sampler", "SAMPLER", False),
         ("sigmas", "SIGMAS", False), ("latent_image", "LATENT", False)],
        [],
        [("LATENT", "output"), ("LATENT", "denoised_output")],
    ),
    "LTXVSeparateAVLatent": ([("av_latent", "LATENT", False)], [], [("LATENT", "video_latent"), ("LATENT", "audio_latent")]),
    "LTXVCropGuides": (
        [("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False), ("latent", "LATENT", False)],
        [],
        [("CONDITIONING", "positive"), ("CONDITIONING", "negative"), ("LATENT", "latent")],
    ),
    "LatentUpscaleModelLoader": ([], [("model_name", "")], [("LATENT_UPSCALE_MODEL", "LATENT_UPSCALE_MODEL")]),
    "LTXVLatentUpsampler": (
        [("samples", "LATENT", False), ("upscale_model", "LATENT_UPSCALE_MODEL", False), ("vae", "VAE", False)],
        [],
        [("LATENT", "LATENT")],
    ),
    "VAEDecodeTiled": (
        [("samples", "LATENT", False), ("vae", "VAE", False)],
        [("tile_size", 512), ("overlap", 64), ("temporal_size", 64), ("temporal_overlap", 8)],
        [("IMAGE", "IMAGE")],
    ),
    "LTXVAudioVAEDecode": ([("samples", "LATENT", False), ("audio_vae", "VAE", False)], [], [("AUDIO", "Audio")]),
    "CreateVideo": (
        [("images", "IMAGE", False), ("audio", "AUDIO", True)],
        [("fps", 24.0), ("bit_depth", 8), ("color_space", "sRGB"), ("codec", "none")],
        [("VIDEO", "VIDEO")],
    ),
    "SaveVideo": ([("video", "VIDEO", False)], [("filename_prefix", "video/ComfyUI"), ("format", "auto"), ("codec", "auto")], []),
    "SaveAudioAdvanced": ([("audio", "AUDIO", False)], [("filename_prefix", "audio/ComfyUI"), ("format", "flac")], []),
    "ImageBatch": ([("image1", "IMAGE", False), ("image2", "IMAGE", False)], [], [("IMAGE", "IMAGE")]),
    "AudioConcat": ([("audio1", "AUDIO", False), ("audio2", "AUDIO", False)], [("direction", "after")], [("AUDIO", "AUDIO")]),
    "MarkdownNote": ([], [("text", "")], []),
    # --- ComfyUI-LTXVideo 220e1d4, iclora.py (validated against source, not object_info) ---
    "LTXICLoRALoaderModelOnly": (
        [("model", "MODEL", False)],
        [("lora_name", ""), ("strength_model", 1.0)],
        [("MODEL", "model"), ("FLOAT", "latent_downscale_factor")],
    ),
    "LTXAddVideoICLoRAGuide": (
        [("positive", "CONDITIONING", False), ("negative", "CONDITIONING", False), ("vae", "VAE", False),
         ("latent", "LATENT", False), ("image", "IMAGE", False)],
        [("frame_idx", 0), ("strength", 1.0), ("latent_downscale_factor", 1.0), ("crop", "disabled"),
         ("use_tiled_encode", False), ("tile_size", 256), ("tile_overlap", 64),
         ("use_streaming", False), ("tile_frames", 97)],
        [("CONDITIONING", "positive"), ("CONDITIONING", "negative"), ("LATENT", "latent")],
    ),
    # --- World Builder (camera-director PR #165), validated against source ---
    "DirectorSceneCapture": (
        [],
        [("capture_file", ""), ("audio_file", ""), ("recipe_json", "{}"), ("depth_file", "")],
        [("IMAGE", "frames"), ("AUDIO", "audio"), ("FLOAT", "fps"), ("STRING", "recipe"),
         ("IMAGE", "depth"), ("IMAGE", "canny")],
    ),
    "DirectorSceneAcoustics": (
        [("recipe", "STRING", False)],
        [],
        [("STRING", "space_preset"), ("STRING", "level"), ("FLOAT", "rt60"), ("STRING", "room_summary"),
         ("COMBO", "space_mode"), ("COMBO", "application"), ("COMBO", "room_preset")],
    ),
}

SOURCE_VALIDATED = {"LTXICLoRALoaderModelOnly", "LTXAddVideoICLoRAGuide", "DirectorSceneCapture", "DirectorSceneAcoustics"}
UI_ONLY = {"MarkdownNote"}
# Widgets the UI serialises but the API graph leaves at the server default, matching
# the verified one-pass graph (which omits them).
API_SKIP = {"CLIPLoader": {"device"}, "CreateVideo": {"color_space", "codec"}}
# Extra UI-only widget values that follow a named widget (seed control, upload button).
UI_AFTER = {("RandomNoise", "noise_seed"): "fixed", ("PrimitiveInt", "value"): "fixed", ("LoadImage", "image"): "image"}
BYPASS = 4


@dataclass
class Node:
    id: int
    cls: str
    pos: tuple[float, float]
    widgets: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)  # input name -> (source id, output slot)
    mode: int = 0
    title: str | None = None
    size: tuple[float, float] | None = None


def widget_values(node: Node) -> dict:
    _, widgets, _ = SPEC[node.cls]
    unknown = set(node.widgets) - {name for name, _ in widgets}
    if unknown:
        raise ValueError(f"node {node.id} {node.cls}: unknown widgets {sorted(unknown)}")
    return {name: node.widgets.get(name, default) for name, default in widgets}


def ui_inputs(node: Node) -> list[tuple[str, str, bool]]:
    """Socket inputs in schema order, then linked widgets: (name, type, is_widget)."""
    sockets, widgets, _ = SPEC[node.cls]
    names = {name for name, _ in widgets}
    unknown = set(node.links) - names - {name for name, _, _ in sockets}
    if unknown:
        raise ValueError(f"node {node.id} {node.cls}: unknown inputs {sorted(unknown)}")
    rows = [(name, typ, False) for name, typ, _ in sockets]
    rows += [(name, None, True) for name, _ in widgets if name in node.links]
    return rows


def node_size(node: Node) -> tuple[float, float]:
    if node.size:
        return node.size
    sockets, widgets, outputs = SPEC[node.cls]
    rows = max(len(sockets) + len(node.links), len(outputs)) + len(widgets)
    return (380, 60 + 26 * rows)


def output_type(nodes: dict, src: int, slot: int) -> str:
    outputs = SPEC[nodes[src].cls][2]
    if slot >= len(outputs):
        raise ValueError(f"node {src} {nodes[src].cls} has no output slot {slot}")
    return outputs[slot][0]


def input_type(nodes: dict, node: Node, name: str) -> str:
    for socket, typ, _ in SPEC[node.cls][0]:
        if socket == name:
            return typ
    src, slot = node.links[name]
    return output_type(nodes, src, slot)  # linked widget takes its source's primitive type


def resolve(nodes: dict, src: int, slot: int, want: str):
    """Follow bypassed nodes the way the ComfyUI frontend does when it builds a prompt:
    output k maps to input k if the types match, otherwise to the first input of that type."""
    seen = set()
    while nodes[src].mode == BYPASS:
        if src in seen:
            raise ValueError(f"bypass loop at node {src}")
        seen.add(src)
        node = nodes[src]
        rows = ui_inputs(node)
        pick = None
        if slot < len(rows) and input_type(nodes, node, rows[slot][0]) == want and rows[slot][0] in node.links:
            pick = rows[slot][0]
        else:
            for name, _, _ in rows:
                if name in node.links and input_type(nodes, node, name) == want:
                    pick = name
                    break
        if pick is None:
            return None
        src, slot = node.links[pick]
    return src, slot


def to_api(nodes: dict) -> dict:
    api = {}
    for node in sorted(nodes.values(), key=lambda n: n.id):
        if node.mode == BYPASS or node.cls in UI_ONLY:
            continue
        sockets = {name: optional for name, _, optional in SPEC[node.cls][0]}
        inputs = {}
        for name, value in widget_values(node).items():
            if name in node.links or name in API_SKIP.get(node.cls, set()):
                continue
            inputs[name] = value
        for name, (src, slot) in node.links.items():
            target = resolve(nodes, src, slot, output_type(nodes, src, slot))
            if target is None:
                if sockets.get(name, False):
                    continue  # optional socket fed only by a bypassed chain
                raise ValueError(f"node {node.id}.{name}: bypassed source leaves a required input empty")
            inputs[name] = [str(target[0]), target[1]]
        api[str(node.id)] = {"class_type": node.cls, "inputs": inputs}
        if node.title:
            api[str(node.id)]["_meta"] = {"title": node.title}
    return api


def to_ui(nodes: dict, groups: list[dict], extra_meta: dict) -> dict:
    links, ui_nodes, link_id = [], [], 0
    out_links: dict[tuple[int, int], list[int]] = {}
    in_link: dict[tuple[int, str], int] = {}
    for node in sorted(nodes.values(), key=lambda n: n.id):
        for name, (src, slot) in node.links.items():
            link_id += 1
            typ = output_type(nodes, src, slot)
            want = input_type(nodes, node, name)
            if want not in ("*", typ) and not (typ == "INT" and want == "FLOAT"):
                raise ValueError(f"link {src}:{slot} ({typ}) -> {node.id}.{name} ({want}) type mismatch")
            target_slot = [r[0] for r in ui_inputs(node)].index(name)
            links.append([link_id, src, slot, node.id, target_slot, typ])
            out_links.setdefault((src, slot), []).append(link_id)
            in_link[(node.id, name)] = link_id
    for order, node in enumerate(sorted(nodes.values(), key=lambda n: n.id)):
        inputs = []
        for name, typ, is_widget in ui_inputs(node):
            row = {"name": name, "type": typ or input_type(nodes, node, name), "link": in_link.get((node.id, name))}
            if is_widget:
                row["widget"] = {"name": name}
            inputs.append(row)
        outputs = [
            {"name": name, "type": typ, "links": out_links.get((node.id, slot), []), "slot_index": slot}
            for slot, (typ, name) in enumerate(SPEC[node.cls][2])
        ]
        values = []
        for name, value in widget_values(node).items():
            values.append(value)
            if (node.cls, name) in UI_AFTER:
                values.append(UI_AFTER[(node.cls, name)])
        entry = {
            "id": node.id, "type": node.cls, "pos": list(node.pos),
            "size": list(node_size(node)), "flags": {}, "order": order,
            "mode": node.mode, "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": node.cls}, "widgets_values": values,
        }
        if node.title:
            entry["title"] = node.title
        ui_nodes.append(entry)
    return {
        "id": "6f1c2a8e-akus-rooms-chain-0001", "revision": 0,
        "last_node_id": max(nodes), "last_link_id": link_id,
        "nodes": ui_nodes, "links": links, "groups": groups, "config": {},
        "extra": {"ds": {"scale": 0.45, "offset": [200, 100]}, **extra_meta}, "version": 0.4,
    }


def group(title: str, nodes: dict, ids, color: str = "#3f789e", pad: int = 40) -> dict:
    members = [nodes[i] for i in ids]
    x0 = min(n.pos[0] for n in members) - pad
    y0 = min(n.pos[1] for n in members) - pad - 30
    x1 = max(n.pos[0] + node_size(n)[0] for n in members) + pad
    y1 = max(n.pos[1] + node_size(n)[1] for n in members) + pad
    return {"title": title, "bounding": [x0, y0, x1 - x0, y1 - y0], "color": color, "font_size": 24, "flags": {}}

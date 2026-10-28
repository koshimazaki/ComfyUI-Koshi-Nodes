#!/usr/bin/env python3
"""Build the AKUSPACE rooms-chain workflow (UI and API format) from one definition.

One character walks through three rooms. LTX-2.5 renders three chained segments;
segments 2 and 3 continue from the last 17 decoded frames of the segment before,
pinned at frame 0 with core LTXVAddGuide and stripped again with LTXVCropGuides
(the same pair the official LTX-2.5 first/last-frame template uses). Each segment
carries its own AKUSPACE room preset and its own time-aligned reference voice.

    python3 scripts/akuspace/build_rooms_chain.py            # write both files
    python3 scripts/akuspace/build_rooms_chain.py --check    # fail if they are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rooms_chain_spec import BYPASS, Node, group, to_api, to_ui

REPO = Path(__file__).resolve().parents[2]
UI_PATH = REPO / "workflows" / "akuspace" / "akuspace_rooms_chain.json"
API_PATH = REPO / "workflows" / "akuspace" / "api" / "akuspace_rooms_chain_api.json"

FPS = 24
OVERLAP = 17  # 8*2+1 frames: three latent frames of hand-off context
LENGTHS = {1: 241, 2: 257, 3: 257}  # 8k+1; 241 = 10.04 s, 257 = 17 overlap + 240 new
UNET = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
CLIP = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
LORA = "akuspace-ltx25-v0.5.safetensors"
UNION_LORA = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
SIGMAS_1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
SIGMAS_2 = "0.85, 0.7250, 0.4219, 0.0"
NEGATIVE = (
    "cut, shot change, scene change, warped face, deformed hands, extra fingers, text, "
    "captions, watermark, distorted, clipped, metallic, unintelligible speech"
)
CHARACTER = (
    "a woman in her late twenties with very short cropped dark hair, warm brown skin, black "
    "over-ear headphones resting around her neck, an open black leather biker jacket over a "
    "black top, loose black cargo trousers and black boots"
)
ROOMS = {
    1: ("cathedral", 558811532553686, (
        f"A steady tracking medium-wide shot follows {CHARACTER} as she walks slowly down the "
        "nave of a vast empty stone cathedral, tall gothic columns and high vaulted ceilings, "
        "shafts of warm light falling through stained-glass windows. She speaks clearly to the "
        "camera as she walks, glancing up at the vaulted ceiling, then heads towards a low wooden "
        "side door. Photorealistic, film-grade colour grading, stable exposure and sharp detail.")),
    2: ("small_room", 558811532553687, (
        f"Continuous shot, same framing. {CHARACTER[0].upper() + CHARACTER[1:]} steps through the "
        "low wooden door into a cramped DJ booth tiled floor to ceiling in glossy warm lime yellow "
        "square tiles, a single vertical neon tube on the left wall, two turntables and a mixer on "
        "the counter. The camera follows her in. She keeps speaking clearly to the camera, touches "
        "the tiles, then walks towards a heavy steel door at the back. Photorealistic, film-grade "
        "colour grading, stable exposure and sharp detail.")),
    3: ("empty_club", 558811532553688, (
        f"Continuous shot, same framing. {CHARACTER[0].upper() + CHARACTER[1:]} pushes the heavy "
        "steel door open and walks out onto the floor of a large empty nightclub, dark concrete "
        "walls, a long bar on one side, a raised DJ stage and rows of unlit speaker stacks, haze "
        "catching coloured stage lights. The camera follows her. She keeps speaking clearly to the "
        "camera and turns slowly to take in the room. Photorealistic, film-grade colour grading, "
        "stable exposure and sharp detail.")),
}

X = [0, 460, 920, 1380, 1840, 2300, 2760, 3220, 3680]
BAND = {0: 0, 1: 1000, 2: 2500, 3: 4000}


def add(nodes: dict, node: Node) -> Node:
    if node.id in nodes:
        raise ValueError(f"duplicate node id {node.id}")
    nodes[node.id] = node
    return node


def shared(nodes: dict) -> None:
    y = BAND[0]
    add(nodes, Node(1, "UNETLoader", (X[0], y), {"unet_name": UNET}))
    add(nodes, Node(2, "LoraLoaderModelOnly", (X[0], y + 180), {"lora_name": LORA, "strength_model": 1.0}, {"model": (1, 0)}))
    add(nodes, Node(6, "LTXICLoRALoaderModelOnly", (X[0], y + 380), {"lora_name": UNION_LORA, "strength_model": 1.0},
                    {"model": (2, 0)}, mode=BYPASS, title="Union control IC-LoRA (depth/canny) - bypassed"))
    add(nodes, Node(3, "CLIPLoader", (X[1], y), {"clip_name": CLIP, "type": "ltxv"}))
    add(nodes, Node(4, "VAELoader", (X[1], y + 180), {"vae_name": "ltx-2.5-video-vae-bf16.safetensors"}, title="Video VAE"))
    add(nodes, Node(5, "VAELoader", (X[1], y + 300), {"vae_name": "ltx-2.5-audio-vae-bf16.safetensors"}, title="Audio VAE"))
    add(nodes, Node(26, "PrimitiveInt", (X[2], y), {"value": 832}, title="width"))
    add(nodes, Node(27, "PrimitiveInt", (X[2], y + 120), {"value": 464}, title="height (latent floors to 448)"))
    add(nodes, Node(31, "PrimitiveFloat", (X[2], y + 240), {"value": float(FPS)}, title="fps"))
    add(nodes, Node(13, "CLIPTextEncode", (X[3], y), {"text": NEGATIVE}, {"clip": (3, 0)}, title="Negative (shared)", size=(400, 220)))
    add(nodes, Node(52, "KSamplerSelect", (X[4], y), {"sampler_name": "euler_ancestral"}))
    add(nodes, Node(53, "ManualSigmas", (X[4], y + 140), {"sigmas": SIGMAS_1}, title="Stage-1 sigmas (8 steps)"))
    add(nodes, Node(70, "LatentUpscaleModelLoader", (X[5], y), {"model_name": "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"}, mode=BYPASS))
    add(nodes, Node(82, "KSamplerSelect", (X[5], y + 160), {"sampler_name": "euler_ancestral"}, mode=BYPASS))
    add(nodes, Node(83, "ManualSigmas", (X[5], y + 300), {"sigmas": SIGMAS_2}, mode=BYPASS, title="Stage-2 sigmas (3 steps)"))
    note = (
        "# AKUSPACE rooms chain (draft, not yet run on a GPU)\n\n"
        "Three LTX-2.5 segments, one character, three rooms: cathedral, small_room, empty_club.\n"
        f"Lengths {LENGTHS[1]} / {LENGTHS[2]} / {LENGTHS[3]} frames at {FPS} fps. Segments 2 and 3 are pinned at "
        f"frame 0 to the last {OVERLAP} decoded frames of the segment before (LTXVAddGuide), then the "
        f"{OVERLAP} overlap frames and their audio are trimmed before the join: 241 + 240 + 240 = 721 frames, 30.04 s.\n\n"
        "Bypassed by default: first-frame images, depth/canny IC-LoRA, stage-2 upscale. "
        "Enable the IC-LoRA loader together with any depth guide, and every stage-2 node together. "
        "See workflows/akuspace/README.md for what is unverified."
    )
    add(nodes, Node(990, "MarkdownNote", (X[6], y), {"text": note}, size=(860, 420), title="Read me"))


def segment(nodes: dict, s: int) -> list[list[int]]:
    b, y = s * 100, BAND[s]
    room, seed, prompt = ROOMS[s]
    length = LENGTHS[s]
    add(nodes, Node(b + 1, "PrimitiveStringMultiline", (X[0], y), {"value": prompt}, title=f"Seg {s} visual prompt", size=(400, 320)))
    add(nodes, Node(b + 2, "Koshi_AKUSPACEPrompt", (X[0], y + 360), {"room_preset": room}, {"prompt": (b + 1, 0)}, title=f"Seg {s} AKUSPACE: {room}"))
    add(nodes, Node(b + 3, "PreviewAny", (X[0], y + 640), {}, {"source": (b + 2, 0)}))
    add(nodes, Node(b + 6, "LoadAudio", (X[0], y + 760), {"audio": "dj_dry_voice.wav"}, title=f"Seg {s} reference voice (dry)"))
    add(nodes, Node(b + 7, "TrimAudioDuration", (X[0], y + 900), {"start_index": 0.0, "duration": length / FPS}, {"audio": (b + 6, 0)}))
    add(nodes, Node(b + 4, "CLIPTextEncode", (X[1], y), {}, {"clip": (3, 0), "text": (b + 2, 0)}))
    add(nodes, Node(b + 5, "LTXVConditioning", (X[1], y + 140), {}, {"positive": (b + 4, 0), "negative": (13, 0), "frame_rate": (31, 0)}))
    add(nodes, Node(b + 8, "AKUSPACEReferenceAudioAligned", (X[1], y + 300), {"reference_guidance_scale": 0.0},
                    {"model": (6, 0), "positive": (b + 5, 0), "negative": (b + 5, 1), "reference_audio": (b + 7, 0), "audio_vae": (5, 0)}))
    add(nodes, Node(b + 9, "PrimitiveInt", (X[1], y + 560), {"value": length}, title=f"Seg {s} frames (8k+1)"))
    add(nodes, Node(b + 10, "EmptyLTXVLatentVideo", (X[1], y + 680), {"width": 832, "height": 464, "length": length, "batch_size": 1},
                    {"width": (26, 0), "height": (27, 0), "length": (b + 9, 0)}))
    add(nodes, Node(b + 11, "LTXVEmptyLatentAudio", (X[1], y + 900), {"frames_number": length, "frame_rate": float(FPS), "batch_size": 1},
                    {"frames_number": (b + 9, 0), "frame_rate": (31, 0), "audio_vae": (5, 0)}))

    # Optional first frame (a FLUX Klein still), bypassed.
    add(nodes, Node(b + 12, "LoadImage", (X[2], y + 420), {"image": f"klein_seg{s}.png"}, mode=BYPASS,
                    title=f"Seg {s} Klein frame (optional)", size=(380, 340)))
    add(nodes, Node(b + 13, "LTXVPreprocess", (X[2], y + 800), {"img_compression": 18}, {"image": (b + 12, 0)}, mode=BYPASS))
    if s == 1:
        add(nodes, Node(b + 14, "LTXVImgToVideoInplace", (X[2], y + 930), {"strength": 0.7, "bypass": False},
                        {"vae": (4, 0), "image": (b + 13, 0), "latent": (b + 10, 0)}, mode=BYPASS, title="Seg 1 first frame"))
        pos, neg, lat = (b + 8, 1), (b + 8, 2), (b + 14, 0)
        hand_off = []
    else:
        prev = (s - 1) * 100
        add(nodes, Node(b + 15, "ImageFromBatch", (X[2], y), {"batch_index": LENGTHS[s - 1] - OVERLAP, "length": OVERLAP},
                        {"image": (prev + 40, 0)}, title=f"Seg {s - 1} tail: last {OVERLAP} frames"))
        add(nodes, Node(b + 16, "LTXVAddGuide", (X[2], y + 140), {"frame_idx": 0, "strength": 1.0},
                        {"positive": (b + 8, 1), "negative": (b + 8, 2), "vae": (4, 0), "latent": (b + 10, 0), "image": (b + 15, 0)},
                        title=f"Continue from seg {s - 1} (frames 0-{OVERLAP - 1})"))
        add(nodes, Node(b + 14, "LTXVAddGuide", (X[2], y + 930), {"frame_idx": OVERLAP, "strength": 0.7},
                        {"positive": (b + 16, 0), "negative": (b + 16, 1), "vae": (4, 0), "latent": (b + 16, 2), "image": (b + 13, 0)},
                        mode=BYPASS, title=f"Seg {s} Klein keyframe at frame {OVERLAP}"))
        pos, neg, lat = (b + 14, 0), (b + 14, 1), (b + 14, 2)
        hand_off = [b + 15, b + 16]

    # Depth / canny guidance slot from a World Builder capture, bypassed.
    depth_len, depth_idx = (length, 0) if s == 1 else (length - OVERLAP - 7, OVERLAP)  # 241 at 0 / 233 at 17
    add(nodes, Node(b + 17, "DirectorSceneCapture", (X[3], y), {}, mode=BYPASS, title=f"Seg {s} World Builder capture", size=(380, 260)))
    add(nodes, Node(b + 18, "DirectorSceneAcoustics", (X[3], y + 300), {}, {"recipe": (b + 17, 3)}, mode=BYPASS))
    add(nodes, Node(b + 19, "PreviewAny", (X[3], y + 560), {}, {"source": (b + 18, 3)}, mode=BYPASS, title="Suggested AKUSPACE room"))
    add(nodes, Node(b + 20, "ImageFromBatch", (X[3], y + 680), {"batch_index": 0, "length": depth_len}, {"image": (b + 17, 4)},
                    mode=BYPASS, title=f"Depth frames ({depth_len})"))
    add(nodes, Node(b + 21, "LTXAddVideoICLoRAGuide", (X[3], y + 840), {"frame_idx": depth_idx, "strength": 1.0},
                    {"positive": pos, "negative": neg, "vae": (4, 0), "latent": lat, "image": (b + 20, 0),
                     "latent_downscale_factor": (6, 1)}, mode=BYPASS, title=f"Seg {s} depth/canny guide"))
    pos, neg, lat = (b + 21, 0), (b + 21, 1), (b + 21, 2)

    # Stage-1 sample.
    add(nodes, Node(b + 22, "LTXVConcatAVLatent", (X[4], y), {}, {"video_latent": lat, "audio_latent": (b + 11, 0)}))
    add(nodes, Node(b + 23, "RandomNoise", (X[4], y + 120), {"noise_seed": seed}))
    add(nodes, Node(b + 24, "LTXVDualCFGGuider", (X[4], y + 250), {"video_cfg": 1.0, "audio_cfg": 1.0},
                    {"model": (b + 8, 0), "positive": pos, "negative": neg}))
    add(nodes, Node(b + 25, "SamplerCustomAdvanced", (X[4], y + 440), {},
                    {"noise": (b + 23, 0), "guider": (b + 24, 0), "sampler": (52, 0), "sigmas": (53, 0), "latent_image": (b + 22, 0)}))
    add(nodes, Node(b + 26, "LTXVSeparateAVLatent", (X[4], y + 640), {}, {"av_latent": (b + 25, 0)}))
    add(nodes, Node(b + 27, "LTXVCropGuides", (X[4], y + 770), {}, {"positive": pos, "negative": neg, "latent": (b + 26, 0)}))

    # Optional stage 2: x2 latent upscale + 3-step refine, bypassed.
    add(nodes, Node(b + 30, "LTXVLatentUpsampler", (X[5], y), {}, {"samples": (b + 27, 2), "upscale_model": (70, 0), "vae": (4, 0)}, mode=BYPASS))
    add(nodes, Node(b + 31, "LTXVConcatAVLatent", (X[5], y + 160), {}, {"video_latent": (b + 30, 0), "audio_latent": (b + 26, 1)}, mode=BYPASS))
    add(nodes, Node(b + 32, "RandomNoise", (X[5], y + 290), {"noise_seed": 42}, mode=BYPASS))
    add(nodes, Node(b + 33, "LTXVDualCFGGuider", (X[5], y + 420), {"video_cfg": 1.0, "audio_cfg": 1.0},
                    {"model": (b + 8, 0), "positive": (b + 27, 0), "negative": (b + 27, 1)}, mode=BYPASS))
    add(nodes, Node(b + 34, "SamplerCustomAdvanced", (X[5], y + 610), {},
                    {"noise": (b + 32, 0), "guider": (b + 33, 0), "sampler": (82, 0), "sigmas": (83, 0), "latent_image": (b + 31, 0)}, mode=BYPASS))
    add(nodes, Node(b + 35, "LTXVSeparateAVLatent", (X[5], y + 810), {}, {"av_latent": (b + 34, 0)}, mode=BYPASS))

    # Decode; audio always from stage 1 (see README).
    add(nodes, Node(b + 40, "VAEDecodeTiled", (X[6], y), {}, {"samples": (b + 35, 0), "vae": (4, 0)}))
    add(nodes, Node(b + 41, "LTXVAudioVAEDecode", (X[6], y + 240), {}, {"samples": (b + 26, 1), "audio_vae": (5, 0)}))
    add(nodes, Node(b + 42, "CreateVideo", (X[6], y + 370), {}, {"images": (b + 40, 0), "audio": (b + 41, 0), "fps": (31, 0)}))
    add(nodes, Node(b + 43, "SaveVideo", (X[6], y + 580), {"filename_prefix": f"video/AKU_rooms_seg{s}_{room}"}, {"video": (b + 42, 0)}))

    # Trim the hand-off overlap before the join.
    keep = length - (0 if s == 1 else OVERLAP)
    start = 0 if s == 1 else OVERLAP
    add(nodes, Node(b + 44, "TrimAudioDuration", (X[7], y), {"start_index": start / FPS, "duration": keep / FPS},
                    {"audio": (b + 41, 0)}, title=f"Seg {s} audio, {keep} frames"))
    trim = [b + 44]
    if s > 1:
        add(nodes, Node(b + 45, "ImageFromBatch", (X[7], y + 180), {"batch_index": OVERLAP, "length": keep},
                        {"image": (b + 40, 0)}, title=f"Seg {s} frames {OVERLAP}-{length - 1}"))
        trim.append(b + 45)

    return [
        [b + 1, b + 2, b + 3, b + 6, b + 7, b + 4, b + 5, b + 8, b + 9, b + 10, b + 11],
        hand_off,
        [b + 12, b + 13, b + 14],
        [b + 17, b + 18, b + 19, b + 20, b + 21],
        [b + 22, b + 23, b + 24, b + 25, b + 26, b + 27],
        [b + 40, b + 41, b + 42, b + 43],
        trim,
    ]


def join(nodes: dict) -> None:
    y = BAND[3] + 300
    add(nodes, Node(900, "ImageBatch", (X[8], BAND[2]), {}, {"image1": (140, 0), "image2": (245, 0)}))
    add(nodes, Node(901, "ImageBatch", (X[8], y), {}, {"image1": (900, 0), "image2": (345, 0)}))
    add(nodes, Node(902, "AudioConcat", (X[8], BAND[2] + 140), {"direction": "after"}, {"audio1": (144, 0), "audio2": (244, 0)}))
    add(nodes, Node(903, "AudioConcat", (X[8], y + 140), {"direction": "after"}, {"audio1": (902, 0), "audio2": (344, 0)}))
    add(nodes, Node(904, "CreateVideo", (X[8], y + 300), {}, {"images": (901, 0), "audio": (903, 0), "fps": (31, 0)}))
    add(nodes, Node(905, "SaveVideo", (X[8], y + 510), {"filename_prefix": "video/AKU_rooms_chain"}, {"video": (904, 0)}))
    add(nodes, Node(906, "SaveAudioAdvanced", (X[8], y + 720), {"filename_prefix": "video/AKU_rooms_chain_audio"}, {"audio": (903, 0)}))


def build() -> tuple[dict, dict]:
    nodes: dict[int, Node] = {}
    shared(nodes)
    seg_groups = {s: segment(nodes, s) for s in (1, 2, 3)}
    join(nodes)
    blue, green, grey, amber, purple = "#3f789e", "#3f7e5a", "#444444", "#8a6d1f", "#6a4c93"
    groups = [
        group("Shared models + settings", nodes, [1, 2, 3, 4, 5, 26, 27, 31, 13, 52, 53], blue),
        group("Control IC-LoRA loader (bypassed)", nodes, [6], amber, pad=20),
    ]
    for s, (main, hand_off, first, depth, sample, decode, trim) in seg_groups.items():
        room = ROOMS[s][0]
        groups.append(group(f"Seg {s} - {room}: prompt, AKUSPACE preset, reference voice", nodes, main, green))
        if hand_off:
            groups.append(group(f"Seg {s} - continuation from seg {s - 1} tail ({OVERLAP} frames)", nodes, hand_off, purple, pad=20))
        groups.append(group(f"Seg {s} - optional Klein first frame (bypassed)", nodes, first, amber, pad=20))
        groups.append(group(f"Seg {s} - depth/canny guide slot (bypassed)", nodes, depth, amber, pad=20))
        groups.append(group(f"Seg {s} - stage-1 sample", nodes, sample, blue))
        groups.append(group(f"Seg {s} - decode + per-segment preview", nodes, decode, grey))
        groups.append(group(f"Seg {s} - trim overlap for join", nodes, trim, grey, pad=20))
    stage2 = [70, 82, 83] + [s * 100 + k for s in (1, 2, 3) for k in range(30, 36)]
    groups.append(group("Optional stage-2 x2 upscale + refine, all segments (bypassed)", nodes, stage2, amber))
    groups.append(group("Join: one clip, one audio track", nodes, [900, 901, 902, 903, 904, 905, 906], green))
    ui = to_ui(nodes, groups, {"akuspace_rooms_chain": {"status": "draft, untested on GPU",
                                                        "ltxvideo_ref": "220e1d45aa0469f518e982102fca4dca92a61862"}})
    return ui, to_api(nodes)


def dump(data: dict) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed files differ")
    args = parser.parse_args()
    ui, api = build()
    targets = {UI_PATH: dump(ui), API_PATH: dump(api)}
    if args.check:
        stale = [str(p) for p, text in targets.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        for path in stale:
            print(f"stale: {path}")
        return 1 if stale else 0
    for path, text in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.relative_to(REPO)}")
    print(f"UI nodes {len(ui['nodes'])}, links {len(ui['links'])}, groups {len(ui['groups'])}; API nodes {len(api)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

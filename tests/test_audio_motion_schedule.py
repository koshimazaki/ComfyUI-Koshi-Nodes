"""Standalone test for the KN Audio -> Motion Schedule node.

Run:  python tests/test_audio_motion_schedule.py

Verifies BFL + Fill JSON ingestion, schedule shape, the exact KoshiMotionEngine
consumption contract, the Deforum string output, the numpy audio path, and
canonical MotionFrame field parity (drift guard). No ComfyUI / torch required.
"""

import dataclasses
import importlib.machinery
import importlib.util
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AUDIO_DIR = os.path.join(REPO, "nodes", "audio")
CORE_DIR = os.path.join(REPO, "nodes", "flux_motion", "core")


def load_audio_package():
    """Load nodes/audio as a package exactly like ComfyUI's dynamic loader."""
    pkg = "kn_audio_test"
    spec = importlib.util.spec_from_file_location(
        pkg, os.path.join(AUDIO_DIR, "__init__.py"),
        submodule_search_locations=[AUDIO_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = module
    spec.loader.exec_module(module)
    return module


def load_canonical_motion_frame():
    """Independently load the real MotionFrame for field-parity comparison."""
    pkg = "kn_fmcore_test"
    pkg_spec = importlib.machinery.ModuleSpec(pkg, loader=None, is_package=True)
    pkg_spec.submodule_search_locations = [CORE_DIR]
    sys.modules[pkg] = importlib.util.module_from_spec(pkg_spec)
    sub = pkg + ".schedule_parser"
    sp_spec = importlib.util.spec_from_file_location(sub, os.path.join(CORE_DIR, "schedule_parser.py"))
    sp = importlib.util.module_from_spec(sp_spec)
    sys.modules[sub] = sp
    sp_spec.loader.exec_module(sp)
    return sp.MotionFrame


def make_bfl_json(seconds=4.0, sr=44100):
    """Synthetic BFL AudioAnalysisResult: a kick every 0.5s in the low band."""
    n = int(seconds * 90)
    waveform = []
    for i in range(n):
        t = i / 90.0
        beat_phase = (t % 0.5) / 0.5
        low = max(0.0, 1.0 - beat_phase * 4.0)          # sharp decay after each beat
        high = max(0.0, 0.2 + 0.2 * math.sin(t * 40.0))
        mid = 0.3 + 0.1 * math.sin(t * 8.0)
        amp = max(low, mid, high)
        waveform.append({"time": t, "peak": amp, "rms": amp * 0.8, "amplitude": amp,
                         "low": low, "mid": mid, "high": high})
    markers = []
    k = 0
    while k * 0.5 < seconds:
        t = k * 0.5
        markers.append({"id": f"hit-{k + 1}-{int(t * 1000)}", "time": t, "relativeTime": t,
                        "kind": "kick", "band": "low", "amplitude": 0.9,
                        "low": 0.95, "mid": 0.3, "high": 0.1, "confidence": 0.9})
        k += 1
    return json.dumps({
        "fileName": "test.wav", "duration": seconds, "sampleRate": sr,
        "start": 0.0, "analyzedDuration": seconds,
        "waveform": waveform, "markers": markers,
        "bandAverages": {"amplitude": 0.4, "low": 0.4, "mid": 0.3, "high": 0.3},
    })


def make_audio_input(seconds=2.0, sr=22050):
    """Synthetic ComfyUI AUDIO dict (numpy-backed): 110Hz tone pulsed at 2 Hz."""
    import numpy as np
    t = np.linspace(0.0, seconds, int(seconds * sr), endpoint=False)
    env = np.clip(np.sin(2 * math.pi * 2.0 * t), 0.0, 1.0)   # 2 pulses/sec
    tone = np.sin(2 * math.pi * 110.0 * t) * env
    waveform = tone.reshape(1, 1, -1)                         # [B, C, T], numpy ok
    return {"waveform": waveform, "sample_rate": sr}


def main():
    failures = []

    def check(name, condition):
        print(("  PASS  " if condition else "  FAIL  ") + name)
        if not condition:
            failures.append(name)

    audio_pkg = load_audio_package()
    node_cls = audio_pkg.NODE_CLASS_MAPPINGS["Koshi_AudioMotionSchedule"]
    node = node_cls()
    num = 48

    common = dict(num_frames=num, fps=24.0, feature="auto", base_zoom=1.0,
                  translation_gain=0.0, angle_gain=0.0, base_strength=0.65,
                  strength_gain=0.0)

    print("[1] BFL AudioAnalysisResult -> motion schedule (waveform driver)")
    schedule, zoom_str = node.generate(source="analysis_json", zoom_gain=0.3,
                                       smoothing=0.1, analysis_json=make_bfl_json(), **common)
    frames = schedule.get("motion_frames", [])
    check("returns a motion_frames list", isinstance(frames, list))
    check(f"len(motion_frames) == {num}", len(frames) == num)
    check("driver == 'waveform'", schedule.get("driver") == "waveform")
    zooms = [f.zoom for f in frames]
    check("zoom within engine range [0.5, 2.0]", all(0.5 <= z <= 2.0 for z in zooms))
    check("zoom reacts to audio (varies > 0.02)", (max(zooms) - min(zooms)) > 0.02)
    check("schedule carries fps/num_frames", schedule.get("fps") == 24.0 and schedule.get("num_frames") == num)

    print("[2] Exact KoshiMotionEngine consumption contract (motion_engine.py:50-57)")
    ok = True
    try:
        ms = schedule
        if ms is not None and "motion_frames" in ms:
            seq = ms["motion_frames"]
            for fi in (0, num // 2, num - 1):
                if 0 <= fi < len(seq):
                    mf = seq[fi]
                    _ = (float(mf.zoom), float(mf.angle),
                         float(mf.translation_x), float(mf.translation_y))
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("    error:", exc)
    check("engine reads mf.zoom/.angle/.translation_x/.translation_y", ok)

    print("[3] Deforum zoom string is keyframed & parseable")
    check("string starts at frame 0", zoom_str.startswith("0:("))
    check("matches Deforum keyframe format",
          bool(re.fullmatch(r"(\d+:\([-+]?\d*\.?\d+\))(, \d+:\([-+]?\d*\.?\d+\))*", zoom_str)))

    print("[4] Fill-Nodes envelope JSON (per-frame envelope -> waveform driver)")
    env_json = json.dumps({"envelope": [round(abs(math.sin(i / 5.0)), 4) for i in range(120)],
                           "total_frames": 120})
    sched2, _ = node.generate(source="analysis_json", zoom_gain=0.3, smoothing=0.1,
                              analysis_json=env_json, **common)
    z2 = [f.zoom for f in sched2["motion_frames"]]
    check("fill envelope -> 48 frames", len(z2) == num)
    check("fill envelope zoom reacts", (max(z2) - min(z2)) > 0.02)

    print("[5] Fill-Nodes beat/drum times JSON (sparse -> markers driver)")
    times_json = json.dumps({"kick_times": [0.0, 0.5, 1.0, 1.5], "snare_times": [0.25, 0.75],
                             "hihat_times": [0.1, 0.3, 0.6], "duration": 2.0})
    sched3, _ = node.generate(source="analysis_json", zoom_gain=0.3, smoothing=0.0,
                              analysis_json=times_json, **common)
    check("fill times -> 'markers' driver", sched3.get("driver") == "markers")
    check("fill times -> 48 frames", len(sched3["motion_frames"]) == num)

    print("[6] Canonical MotionFrame parity (drift guard)")
    mapping_mod = sys.modules["kn_audio_test.mapping"]
    used_cls = mapping_mod.get_motion_frame_cls()
    canonical = load_canonical_motion_frame()
    used_fields = [f.name for f in dataclasses.fields(used_cls)]
    canon_fields = [f.name for f in dataclasses.fields(canonical)]
    check("resolved MotionFrame fields match canonical", used_fields == canon_fields)
    check("resolved the real class (not local fallback)",
          used_cls is not mapping_mod.local_motion_frame_cls())

    print("[7] audio source mode (numpy STFT band analysis)")
    sched4, _ = node.generate(source="audio", zoom_gain=0.4, smoothing=0.15,
                              audio=make_audio_input(), **common)
    z4 = [f.zoom for f in sched4["motion_frames"]]
    check("audio -> 48 frames", len(z4) == num)
    check("audio -> 'waveform' driver", sched4.get("driver") == "waveform")
    check("audio zoom reacts", (max(z4) - min(z4)) > 0.01)

    print("[8] error handling")
    try:
        node.generate(source="analysis_json", zoom_gain=0.3, smoothing=0.1, analysis_json="", **common)
        check("empty analysis_json raises", False)
    except ValueError:
        check("empty analysis_json raises", True)
    try:
        bad = make_audio_input()
        bad["sample_rate"] = 0
        node.generate(source="audio", zoom_gain=0.3, smoothing=0.1, audio=bad, **common)
        check("audio sample_rate<=0 raises ValueError", False)
    except ValueError:
        check("audio sample_rate<=0 raises ValueError", True)

    print("[9] forced-feature fallback (no silent flat schedule)")
    wf_only = json.dumps({"analyzedDuration": 4.0, "markers": [],
                          "waveform": [{"time": i / 90, "amplitude": abs(math.sin(i / 8)),
                                        "low": abs(math.sin(i / 8)), "mid": 0.3, "high": 0.2}
                                       for i in range(360)]})
    sched_f, _ = node.generate(source="analysis_json", num_frames=num, fps=24.0, feature="markers",
                               base_zoom=1.0, zoom_gain=0.3, translation_gain=0.0, angle_gain=0.0,
                               base_strength=0.65, strength_gain=0.0, smoothing=0.1, analysis_json=wf_only)
    zf = [f.zoom for f in sched_f["motion_frames"]]
    check("forced 'markers' on waveform-only falls back to waveform", sched_f.get("driver") == "waveform")
    check("fallback still produces motion (not a flat schedule)", (max(zf) - min(zf)) > 0.02)
    times_only = json.dumps({"kick_times": [0.0, 0.5, 1.0, 1.5], "duration": 2.0})
    sched_g, _ = node.generate(source="analysis_json", num_frames=num, fps=24.0, feature="waveform",
                               base_zoom=1.0, zoom_gain=0.3, translation_gain=0.0, angle_gain=0.0,
                               base_strength=0.65, strength_gain=0.0, smoothing=0.0, analysis_json=times_only)
    check("forced 'waveform' on markers-only falls back to markers", sched_g.get("driver") == "markers")

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        return 1
    print("RESULT: ALL PASSED")
    return 0


def test_audio_motion_schedule_suite():
    """Run the standalone suite under pytest so it is part of the normal run."""
    assert main() == 0


def test_wires_to_motion_engine():
    """The audio node's schedule output must be the type the engine accepts."""
    audio_pkg = load_audio_package()
    node_cls = audio_pkg.NODE_CLASS_MAPPINGS["Koshi_AudioMotionSchedule"]

    sys.path.insert(0, REPO)
    from nodes.flux_motion.motion_engine import KoshiMotionEngine
    from nodes.flux_motion.schedule import KoshiSchedule

    engine_in = KoshiMotionEngine.INPUT_TYPES()["optional"]["motion_schedule"][0]
    assert node_cls.RETURN_TYPES[0] == engine_in
    assert KoshiSchedule.RETURN_TYPES[0] == engine_in


if __name__ == "__main__":
    raise SystemExit(main())

"""Feature-curve -> motion-schedule mapping for Koshi audio nodes.

Pure-numpy bridge that turns per-frame feature tracks (see ``features.py``) into a
``KOSHI_SCHEDULE`` consumable by ``KoshiMotionEngine``
(``nodes/flux_motion/motion_engine.py``).

The motion-frame objects reuse the canonical ``MotionFrame`` dataclass from
``nodes/flux_motion/core/schedule_parser.py`` when it is importable, and fall back
to an identical local definition so this module also works standalone (tests /
Comfy Cloud) without depending on the dynamic-loader package layout or torch.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("koshi.audio")


@dataclass
class _LocalMotionFrame:
    """Mirror of flux_motion.core.schedule_parser.MotionFrame.

    Only used as a last-resort fallback when the canonical class cannot be
    imported. Keep the field list in sync with the canonical dataclass.
    """

    frame_index: int
    zoom: float = 1.0
    angle: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.0
    strength: float = 0.65
    prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            "zoom": self.zoom,
            "angle": self.angle,
            "translation_x": self.translation_x,
            "translation_y": self.translation_y,
            "translation_z": self.translation_z,
        }


_MOTION_FRAME_CLS = None


def get_motion_frame_cls():
    """Return the canonical ``MotionFrame`` class, with robust fallbacks.

    1. Reuse flux_motion's already-loaded module (normal ComfyUI runtime, where
       ``nodes.flux_motion`` is loaded before ``nodes.audio``).
    2. Path-load ``schedule_parser.py`` inside a synthetic package (torch-free;
       used in tests / Cloud before flux_motion is imported).
    3. Fall back to the local mirror dataclass.
    """
    global _MOTION_FRAME_CLS
    if _MOTION_FRAME_CLS is not None:
        return _MOTION_FRAME_CLS

    direct = sys.modules.get("koshi_nodes_flux_motion.core.schedule_parser")
    if direct is not None and hasattr(direct, "MotionFrame"):
        _MOTION_FRAME_CLS = direct.MotionFrame
        return _MOTION_FRAME_CLS
    for name, mod in list(sys.modules.items()):
        if name.endswith("schedule_parser") and hasattr(mod, "MotionFrame"):
            _MOTION_FRAME_CLS = mod.MotionFrame
            return _MOTION_FRAME_CLS

    try:
        core_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "flux_motion", "core")
        )
        sp_path = os.path.join(core_dir, "schedule_parser.py")
        if os.path.isfile(sp_path):
            pkg = "koshi_audio_fmcore"
            if pkg not in sys.modules:
                pkg_spec = importlib.machinery.ModuleSpec(pkg, loader=None, is_package=True)
                pkg_spec.submodule_search_locations = [core_dir]
                sys.modules[pkg] = importlib.util.module_from_spec(pkg_spec)
            sub = pkg + ".schedule_parser"
            sp_spec = importlib.util.spec_from_file_location(sub, sp_path)
            sp_mod = importlib.util.module_from_spec(sp_spec)
            sys.modules[sub] = sp_mod
            sp_spec.loader.exec_module(sp_mod)
            if hasattr(sp_mod, "MotionFrame"):
                _MOTION_FRAME_CLS = sp_mod.MotionFrame
                return _MOTION_FRAME_CLS
    except Exception:
        pass

    _MOTION_FRAME_CLS = _LocalMotionFrame
    return _MOTION_FRAME_CLS


def local_motion_frame_cls():
    """Expose the local fallback class (for tests / introspection)."""
    return _LocalMotionFrame


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smooth(values: np.ndarray, amount: float) -> np.ndarray:
    """Box-filter smoothing; ``amount`` in [0, 1] scales the window up to ~25% len."""
    if amount <= 0.0 or values.size < 3:
        return values
    win = 1 + int(round(amount * 0.25 * values.size))
    if win <= 1:
        return values
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(values, kernel, mode="same")


def _resample(src_times: np.ndarray, src_vals: np.ndarray, num_frames: int) -> np.ndarray:
    """Stretch a (time, value) track across all frames via linear interpolation."""
    if src_vals.size == 0:
        return np.zeros(num_frames, dtype=float)
    if num_frames == 1:
        return np.array([float(src_vals.flat[0])], dtype=float)
    if src_vals.size == 1 or src_times.size < 2:
        return np.full(num_frames, float(src_vals.flat[0]), dtype=float)
    t0, t1 = float(src_times[0]), float(src_times[-1])
    if t1 <= t0:
        return np.full(num_frames, float(src_vals[0]), dtype=float)
    targets = np.linspace(t0, t1, num_frames)
    return np.interp(targets, src_times, src_vals)


def _bands_from_markers(markers: List[dict], num_frames: int) -> Dict[str, np.ndarray]:
    """Build per-frame band envelopes by interpolating sparse markers across frames."""
    keys = ("low", "mid", "high", "amplitude")
    if not markers:
        return {k: np.zeros(num_frames, dtype=float) for k in keys}

    times = np.array(
        [float(m.get("t", m.get("relativeTime", m.get("time", 0.0)))) for m in markers],
        dtype=float,
    )
    order = np.argsort(times)
    times = times[order]
    tmin, tmax = float(times[0]), float(times[-1])
    if tmax <= tmin or num_frames == 1:
        frame_pos = np.zeros(times.size, dtype=float)
    else:
        frame_pos = (times - tmin) / (tmax - tmin) * (num_frames - 1)

    frame_axis = np.arange(num_frames, dtype=float)
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        vals = np.array([float(markers[i].get(key, 0.0)) for i in order], dtype=float)
        if vals.size == 1:
            out[key] = np.full(num_frames, float(vals[0]), dtype=float)
        else:
            out[key] = np.interp(frame_axis, frame_pos, vals)
    return out


def frame_bands(tracks: dict, num_frames: int, feature: str, smoothing: float) -> Dict[str, np.ndarray]:
    """Resolve per-frame band arrays (low/mid/high/amplitude) from feature tracks.

    ``feature`` chooses the driver: ``auto`` (waveform if present, else markers),
    ``waveform`` (dense continuous envelope), or ``markers`` (sparse impacts). If a
    driver is explicitly requested but the tracks lack that data, fall back to the
    other driver (with a warning) rather than silently emitting a flat schedule.
    """
    keys = ("low", "mid", "high", "amplitude")
    time = tracks.get("time")
    has_cont = (
        time is not None
        and len(time) > 1
        and tracks.get("amplitude") is not None
        and len(tracks["amplitude"]) == len(time)
    )
    has_markers = bool(tracks.get("markers"))

    if feature == "auto":
        mode = "waveform" if has_cont else "markers"
    else:
        mode = feature

    # Forced-mode fallback: the requested driver has no data -> use the other one.
    if mode == "waveform" and not has_cont and has_markers:
        if feature != "auto":
            logger.warning("feature='waveform' but no continuous waveform data; falling back to markers.")
        mode = "markers"
    elif mode == "markers" and not has_markers and has_cont:
        if feature != "auto":
            logger.warning("feature='markers' but no marker data; falling back to waveform.")
        mode = "waveform"

    if mode == "waveform" and has_cont:
        t = np.asarray(time, dtype=float)
        out = {k: _resample(t, np.asarray(tracks.get(k, []), dtype=float), num_frames) for k in keys}
    elif mode == "markers" and has_markers:
        out = _bands_from_markers(tracks["markers"], num_frames)
    else:
        logger.warning("No usable feature data for driver '%s'; emitting a flat (no-motion) schedule.", mode)
        out = {k: np.zeros(num_frames, dtype=float) for k in keys}

    out = {k: _smooth(v, smoothing) for k, v in out.items()}
    out["_mode"] = mode
    return out


def build_motion_schedule(
    tracks: dict,
    num_frames: int,
    fps: float,
    *,
    feature: str = "auto",
    base_zoom: float = 1.0,
    zoom_gain: float = 0.12,
    angle_gain: float = 0.0,
    translation_gain: float = 0.0,
    base_strength: float = 0.65,
    strength_gain: float = 0.0,
    smoothing: float = 0.2,
) -> Tuple[dict, List[float]]:
    """Map feature tracks to a KOSHI_SCHEDULE.

    Mapping (all band energies are 0..1, scaled by the corresponding gain):
        zoom         = base_zoom     + low       * zoom_gain        (bass -> push-in)
        angle        =                 high      * angle_gain       (highs -> rotation)
        translation_x=                 mid       * translation_gain (mids -> drift)
        strength     = base_strength + amplitude * strength_gain    (energy -> denoise)

    Returns ``(schedule, zoom_values)`` where ``schedule`` is the engine-ready dict
    and ``zoom_values`` is the per-frame zoom list (for the Deforum string output).
    """
    num_frames = max(1, int(num_frames))
    bands = frame_bands(tracks, num_frames, feature, float(smoothing))
    motion_frame_cls = get_motion_frame_cls()

    frames = []
    zoom_values: List[float] = []
    for i in range(num_frames):
        low = float(bands["low"][i])
        mid = float(bands["mid"][i])
        high = float(bands["high"][i])
        amp = float(bands["amplitude"][i])

        zoom = _clamp(base_zoom + low * zoom_gain, 0.5, 2.0)
        angle = _clamp(high * angle_gain, -180.0, 180.0)
        translation_x = _clamp(mid * translation_gain, -100.0, 100.0)
        strength = _clamp(base_strength + amp * strength_gain, 0.0, 1.0)

        frames.append(
            motion_frame_cls(
                frame_index=i,
                zoom=zoom,
                angle=angle,
                translation_x=translation_x,
                translation_y=0.0,
                translation_z=0.0,
                strength=strength,
            )
        )
        zoom_values.append(zoom)

    schedule = {
        "motion_frames": frames,
        "fps": float(fps),
        "num_frames": num_frames,
        "driver": bands["_mode"],
        "meta": dict(tracks.get("meta", {})),
    }
    return schedule, zoom_values


def to_deforum_string(values: List[float], decimals: int = 3) -> str:
    """Compact Deforum-style keyframe string (compatible with KoshiSchedule)."""
    parts: List[str] = []
    last = None
    count = len(values)
    for i, value in enumerate(values):
        rounded = round(float(value), decimals)
        if rounded != last or i == 0 or i == count - 1:
            parts.append(f"{i}:({rounded})")
            last = rounded
    return ", ".join(parts)

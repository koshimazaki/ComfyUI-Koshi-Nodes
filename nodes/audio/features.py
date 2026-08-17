"""Feature extraction for Koshi audio -> motion nodes.

Turns one of three sources into a common ``tracks`` dict consumed by ``mapping.py``:

  - analysis_json: BFL dashboard ``AudioAnalysisResult`` (markers + dense waveform),
    or Fill-Nodes ``envelope`` / beat / drum-times JSON. No heavy deps -> Cloud safe.
  - audio:         a ComfyUI AUDIO dict, analyzed with a numpy STFT band-split.
  - video:         an MP4/path, per-frame brightness + motion magnitude via OpenCV.

tracks schema::

    {
      "time": 1-D float array (relative seconds) | None,
      "amplitude"/"low"/"mid"/"high": 1-D float arrays aligned to "time" | None,
      "duration": float,
      "markers": [ {"t", "band", "kind", "low", "mid", "high", "amplitude", "confidence"} ],
      "meta": {...},
    }

A track is "continuous" when ``time`` and the band arrays are populated; otherwise
``markers`` carries sparse impact points. ``mapping.frame_bands`` handles both.
"""

from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

# Band split (Hz) -- mirrors the BFL dashboard (ui/lib/audio-analysis.ts).
LOW_CUTOFF = 180.0
HIGH_CUTOFF = 3600.0


def _normalize(values) -> np.ndarray:
    """Robust 0..1 normalization (8th/98th percentile) -- matches BFL behaviour."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    floor = float(np.percentile(arr, 8))
    ceiling = float(np.percentile(arr, 98))
    spread = max(ceiling - floor, float(np.max(arr)) * 0.08, 1e-6)
    return np.clip((arr - floor) / spread, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# analysis_json (BFL dashboard + Fill-Nodes)                                  #
# --------------------------------------------------------------------------- #

def tracks_from_analysis_json(text: str) -> dict:
    """Parse a pasted analysis JSON, auto-detecting BFL or Fill-Nodes formats."""
    text = (text or "").strip()
    if not text:
        raise ValueError(
            "analysis_json is empty -- paste a BFL/Fill analysis JSON, or switch 'source'."
        )
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("analysis_json must be a JSON object.")

    if "waveform" in data or "markers" in data:
        return _tracks_from_bfl(data)
    if "envelope" in data:
        return _tracks_from_fill_envelope(data)
    if any(k in data for k in ("beat_times", "kick_times", "snare_times", "hihat_times")):
        return _tracks_from_fill_times(data)
    raise ValueError(
        "Unrecognized analysis JSON. Expected BFL AudioAnalysisResult (markers/waveform), "
        "Fill envelope (envelope/total_frames), or Fill beat/drum times (beat_times/kick_times...)."
    )


def _tracks_from_bfl(data: dict) -> dict:
    waveform = data.get("waveform") or []
    markers: List[dict] = []
    for m in data.get("markers") or []:
        markers.append({
            "t": float(m.get("relativeTime", m.get("time", 0.0))),
            "band": m.get("band", "mid"),
            "kind": m.get("kind", "beat"),
            "low": float(m.get("low", 0.0)),
            "mid": float(m.get("mid", 0.0)),
            "high": float(m.get("high", 0.0)),
            "amplitude": float(m.get("amplitude", 0.0)),
            "confidence": float(m.get("confidence", 0.0)),
        })

    if waveform:
        time = np.array([float(p.get("time", 0.0)) for p in waveform], dtype=float)
        amplitude = np.array([float(p.get("amplitude", 0.0)) for p in waveform], dtype=float)
        low = np.array([float(p.get("low", 0.0)) for p in waveform], dtype=float)
        mid = np.array([float(p.get("mid", 0.0)) for p in waveform], dtype=float)
        high = np.array([float(p.get("high", 0.0)) for p in waveform], dtype=float)
    else:
        time = amplitude = low = mid = high = None

    duration = float(
        data.get("analyzedDuration")
        or (time[-1] - time[0] if time is not None and len(time) > 1 else 0.0)
        or (markers[-1]["t"] if markers else 1.0)
    )
    return {
        "time": time, "amplitude": amplitude, "low": low, "mid": mid, "high": high,
        "duration": duration, "markers": markers,
        "meta": {"format": "bfl", "fileName": data.get("fileName"), "marker_count": len(markers)},
    }


def _tracks_from_fill_envelope(data: dict) -> dict:
    """Fill_Audio_Reactive_Envelope: {"envelope": [..per frame..], "total_frames": N}."""
    env = np.asarray(data.get("envelope", []), dtype=float)
    if env.size == 0:
        raise ValueError("Fill envelope JSON has an empty 'envelope' array.")
    env = _normalize(env) if float(np.max(env)) > 1.0 else np.clip(env, 0.0, 1.0)
    n = env.size
    time = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
    return {
        "time": time, "amplitude": env, "low": env.copy(), "mid": env.copy(), "high": env.copy(),
        "duration": 1.0, "markers": [],
        "meta": {"format": "fill_envelope", "total_frames": int(data.get("total_frames", n))},
    }


def _tracks_from_fill_times(data: dict) -> dict:
    """Fill BPM/Drum detectors: beat_times / kick_times / snare_times / hihat_times."""
    markers: List[dict] = []
    band_map = [
        ("kick_times", "low", "kick"),
        ("snare_times", "mid", "snare"),
        ("hihat_times", "high", "hat"),
        ("beat_times", "mid", "beat"),
    ]
    for key, band, kind in band_map:
        for t in data.get(key, []) or []:
            marker = {"t": float(t), "band": band, "kind": kind,
                      "low": 0.0, "mid": 0.0, "high": 0.0, "amplitude": 1.0, "confidence": 1.0}
            marker[band] = 1.0
            markers.append(marker)
    if not markers:
        raise ValueError("Fill times JSON contained no kick/snare/hihat/beat times.")
    markers.sort(key=lambda m: m["t"])
    duration = float(data.get("duration") or markers[-1]["t"] or 1.0)
    return {
        "time": None, "amplitude": None, "low": None, "mid": None, "high": None,
        "duration": duration, "markers": markers,
        "meta": {"format": "fill_times", "marker_count": len(markers)},
    }


# --------------------------------------------------------------------------- #
# audio (numpy STFT band-split)                                               #
# --------------------------------------------------------------------------- #

def tracks_from_audio(
    audio: dict,
    start: float = 0.0,
    duration: Optional[float] = None,
    frame_size: int = 2048,
    hop_size: int = 512,
) -> dict:
    """Analyze a ComfyUI AUDIO dict ({"waveform": [B,C,T], "sample_rate": int})."""
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("AUDIO input missing 'waveform'.")
    sr = int(audio.get("sample_rate", 44100))
    if sr <= 0:
        raise ValueError("AUDIO 'sample_rate' must be positive.")

    waveform = audio["waveform"]
    try:
        arr = waveform.detach().cpu().numpy()
    except AttributeError:
        arr = np.asarray(waveform)
    if arr.ndim == 3:
        arr = arr[0]
    mono = arr.mean(axis=0) if arr.ndim == 2 else arr.reshape(-1)
    mono = np.asarray(mono, dtype=float)

    total = mono.size / sr if sr else 0.0
    start_sample = int(max(0.0, start) * sr)
    if duration and duration > 0:
        end_sample = min(mono.size, start_sample + int(duration * sr))
    else:
        end_sample = mono.size
    seg = mono[start_sample:end_sample]
    if seg.size < frame_size:
        seg = np.pad(seg, (0, max(0, frame_size - seg.size)))

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    low_mask = freqs < LOW_CUTOFF
    mid_mask = (freqs >= LOW_CUTOFF) & (freqs < HIGH_CUTOFF)
    high_mask = freqs >= HIGH_CUTOFF
    window = np.hanning(frame_size)

    times: List[float] = []
    amp: List[float] = []
    low: List[float] = []
    mid: List[float] = []
    high: List[float] = []
    for offset in range(0, max(1, seg.size - frame_size + 1), hop_size):
        frame = seg[offset:offset + frame_size] * window
        power = np.abs(np.fft.rfft(frame)) ** 2
        amp.append(float(np.sqrt(np.mean(frame * frame))))
        low.append(float(np.sqrt(power[low_mask].mean())) if low_mask.any() else 0.0)
        mid.append(float(np.sqrt(power[mid_mask].mean())) if mid_mask.any() else 0.0)
        high.append(float(np.sqrt(power[high_mask].mean())) if high_mask.any() else 0.0)
        times.append((offset + frame_size / 2) / sr)

    return {
        "time": np.asarray(times, dtype=float),
        "amplitude": _normalize(amp), "low": _normalize(low),
        "mid": _normalize(mid), "high": _normalize(high),
        "duration": float(seg.size / sr if sr else 0.0), "markers": [],
        "meta": {"format": "audio", "sample_rate": sr, "total_duration": total},
    }


# --------------------------------------------------------------------------- #
# video (OpenCV brightness + motion magnitude)                                #
# --------------------------------------------------------------------------- #

def tracks_from_video(video_path: str, max_frames: int = 2048) -> dict:
    """Per-frame brightness / motion / detail from an MP4 (the 'extraction' mode)."""
    try:
        import cv2  # noqa: PLC0415 -- lazy: OpenCV is an optional, heavy dep
    except Exception as exc:  # ImportError or numpy/ABI mismatch
        raise ImportError(
            "source='video' needs a working OpenCV (opencv-python). "
            "Install/repair it, or use source='analysis_json' / 'audio'."
        ) from exc

    import os
    if not video_path or not os.path.isfile(video_path):
        raise ValueError(f"video_path not found: {video_path!r}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times: List[float] = []
    brightness: List[float] = []
    motion: List[float] = []
    detail: List[float] = []
    prev = None
    index = 0
    try:
        while index < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
            brightness.append(float(gray.mean()))
            detail.append(float(cv2.Laplacian(gray, cv2.CV_32F).var()))
            motion.append(float(np.abs(gray - prev).mean()) if prev is not None else 0.0)
            prev = gray
            times.append(index / fps if fps else float(index))
            index += 1
    finally:
        cap.release()

    if index == 0:
        raise ValueError(f"No frames decoded from {video_path!r}.")

    low = _normalize(motion)       # motion magnitude -> bass-like driver (zoom)
    high = _normalize(detail)      # texture/detail energy -> high band
    amplitude = _normalize(brightness)
    mid = np.clip((low + high) * 0.5, 0.0, 1.0)
    return {
        "time": np.asarray(times, dtype=float),
        "amplitude": amplitude, "low": low, "mid": mid, "high": high,
        "duration": float(times[-1] if times else 0.0), "markers": [],
        "meta": {"format": "video", "fps": fps, "frames": index},
    }

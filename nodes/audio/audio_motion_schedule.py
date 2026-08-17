"""KN Audio -> Motion Schedule node.

Bridges audio/video features into a ``KOSHI_MOTION_SCHEDULE`` for
``KoshiMotionEngine`` (plus a Deforum-style zoom schedule string). It does NOT
re-implement audio analysis that already exists in the community -- instead it
ingests the BFL dashboard's analysis JSON or Fill-Nodes' JSON and maps those
features onto Koshi's latent-space motion engine. Two extraction modes are also
built in for standalone use:

  - analysis_json : paste BFL ``AudioAnalysisResult`` / Fill JSON  (Comfy Cloud safe)
  - audio         : a ComfyUI AUDIO input            (numpy STFT band analysis)
  - video         : an MP4 path                      (OpenCV brightness/motion)
"""

from __future__ import annotations

from . import features, mapping


class KoshiAudioMotionSchedule:
    """Map audio/video features to a latent motion schedule + Deforum string."""

    COLOR = "#1a1a1a"
    BGCOLOR = "#2d2d2d"

    CATEGORY = "Koshi/Audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("KOSHI_MOTION_SCHEDULE", "STRING")
    RETURN_NAMES = ("motion_schedule", "zoom_schedule_string")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": (["analysis_json", "audio", "video"],),
                "num_frames": ("INT", {"default": 96, "min": 1, "max": 100000}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "feature": (["auto", "waveform", "markers"],),
                "base_zoom": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "zoom_gain": ("FLOAT", {"default": 0.12, "min": -1.0, "max": 1.0, "step": 0.01}),
                "translation_gain": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "angle_gain": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "base_strength": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength_gain": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "analysis_json": ("STRING", {"multiline": True, "default": ""}),
                "audio": ("AUDIO",),
                "video_path": ("STRING", {"default": ""}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1}),
                "duration_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1}),
            },
        }

    def generate(
        self,
        source,
        num_frames,
        fps,
        feature,
        base_zoom,
        zoom_gain,
        translation_gain,
        angle_gain,
        base_strength,
        strength_gain,
        smoothing,
        analysis_json="",
        audio=None,
        video_path="",
        start_seconds=0.0,
        duration_seconds=0.0,
    ):
        if source == "analysis_json":
            tracks = features.tracks_from_analysis_json(analysis_json)
        elif source == "audio":
            if audio is None:
                raise ValueError("source='audio' but no AUDIO input was connected.")
            tracks = features.tracks_from_audio(
                audio,
                start=start_seconds,
                duration=duration_seconds if duration_seconds > 0 else None,
            )
        elif source == "video":
            tracks = features.tracks_from_video(video_path)
        else:
            raise ValueError(f"Unknown source: {source!r}")

        schedule, zoom_values = mapping.build_motion_schedule(
            tracks,
            num_frames,
            fps,
            feature=feature,
            base_zoom=base_zoom,
            zoom_gain=zoom_gain,
            angle_gain=angle_gain,
            translation_gain=translation_gain,
            base_strength=base_strength,
            strength_gain=strength_gain,
            smoothing=smoothing,
        )
        return (schedule, mapping.to_deforum_string(zoom_values))


NODE_CLASS_MAPPINGS = {
    "Koshi_AudioMotionSchedule": KoshiAudioMotionSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Koshi_AudioMotionSchedule": "▄▀▄ KN Audio → Motion",
}

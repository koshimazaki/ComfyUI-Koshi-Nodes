"""AKUSPACE Reference Audio (aligned) — in-context reference for a2a IC-LoRAs.

WHY THIS EXISTS
ComfyUI core ships `LTXVReferenceAudio`, but that node implements the ID-LoRA
(speaker-identity) convention: the reference tokens are shifted to NEGATIVE
time so the reference ends just before t=0 and the target starts at t=0
(comfy/ldm/lightricks/av_model.py, LTXAVModel._process_input, "Compute negative
temporal positions matching ID-LoRA convention").

The LTX trainer's a2a IC-LoRA does something different. Both at training time
(ltx_trainer/training_strategies/flexible.py::_apply_reference_condition ->
_get_audio_positions(num_time_steps=cond_seq_len)) and at inference time
(ltx_trainer/validation_runner.py -> patchifier.get_patch_grid_bounds(ref_shape),
mirroring ltx_core AudioConditionByReferenceLatent) the reference tokens get the
SAME positions a target of that length would get, starting at t=0. Reference
token i sits at the same time as target token i. That time-aligned in-context
layout is what AKUSPACE learned, and it is why it preserves timing (16-24 ms).

Feeding AKUSPACE through the stock node therefore puts every reference token
T_ref+1 latents away from where the adapter learned to look. This node keeps the
trainer's convention. Everything else (clean reference, timestep 0, excluded
from the output, normalized latents from the audio VAE) matches the stock path.

MECHANISM
A small monkeypatch of LTXAVModel._process_input. When the conditioning's
`ref_audio` dict carries `akuspace_aligned: True`, the stock injection is
bypassed (ref_audio is removed before the original runs) and the reference is
prepended afterwards with positions from the very same AudioPatchifier the
target went through. Conditioning without the flag is untouched, so the stock
node keeps working side by side — that is the A/B this was written for.

Verified against ComfyUI master, 2026-08-18. If a future core changes
_process_input's return shape, the patch fails loudly at first use rather than
silently rendering the stock convention.

PACKAGING
This module must import cleanly OUTSIDE ComfyUI (the pack's test-suite, the
registry build). Everything that needs torch/comfy is behind `_COMFY`; without
it the class still registers (so workflow JSON resolves) and `apply()` raises a
clear error instead of an ImportError at pack load. The node id is deliberately
unprefixed — `AKUSPACEReferenceAudioAligned` — to match the standalone copy in
the AKUSPACE session kit, so graphs built for either resolve against the other.
"""

import logging

logger = logging.getLogger(__name__)

_FLAG = "akuspace_aligned"
_COMFY = False
_IMPORT_ERROR = None

try:  # pragma: no cover - exercised only inside ComfyUI
    import torch
    import torchaudio

    import comfy.samplers
    import node_helpers
    import comfy.ldm.lightricks.av_model as _avm

    _COMFY = True
except Exception as exc:  # ImportError outside ComfyUI; anything else = old core
    _IMPORT_ERROR = exc
    logger.debug("AKUSPACE aligned reference: ComfyUI core not importable here: %s", exc)


def _install_patch():  # pragma: no cover - exercised only inside ComfyUI
    """Patch LTXAVModel._process_input once, even if two copies of this node load."""

    current = _avm.LTXAVModel._process_input
    if getattr(current, "_akuspace_patched", False):
        return  # the session-kit copy (or an earlier import) already installed it

    _orig = current

    def _process_input_aligned(self, x, keyframe_idxs, denoise_mask, **kwargs):
        ref = kwargs.get("ref_audio", None)
        if not (isinstance(ref, dict) and ref.get(_FLAG)):
            return _orig(self, x, keyframe_idxs, denoise_mask, **kwargs)

        # Let the original do everything except the reference injection.
        kw = dict(kwargs)
        kw.pop("ref_audio", None)
        out = _orig(self, x, keyframe_idxs, denoise_mask, **kw)
        try:
            (vx, ax), (v_coords, a_coords), add = out
        except Exception as e:
            raise RuntimeError(
                "AKUSPACE aligned reference: unexpected _process_input return shape "
                f"({type(out)}); ComfyUI core changed, the patch needs updating"
            ) from e

        p = self.a_patchifier
        lat = ref["latents"].to(device=ax.device)  # (b, c, t, f), normalized
        ref_tokens, ref_pos = p.patchify(lat)  # (b, t, c*f), (b, 1, t, 2) from t=0
        ref_tokens = ref_tokens.to(dtype=ax.dtype)
        if ref_tokens.shape[0] < ax.shape[0]:
            ref_tokens = ref_tokens.expand(ax.shape[0], -1, -1)
            ref_pos = ref_pos.expand(ax.shape[0], -1, -1, -1)
        ref_proj = self.audio_patchify_proj(ref_tokens)  # same projection the target got

        n_ref = ref_proj.shape[1]
        ax = torch.cat([ref_proj, ax], dim=1)
        a_coords = torch.cat([ref_pos.to(a_coords), a_coords], dim=2)
        add["ref_audio_seq_len"] = n_ref
        add["target_audio_seq_len"] = ax.shape[1] - n_ref
        return [vx, ax], [v_coords, a_coords], add

    _process_input_aligned._akuspace_patched = True
    _avm.LTXAVModel._process_input = _process_input_aligned


if _COMFY:  # pragma: no cover
    _install_patch()


class AKUSPACEReferenceAudioAligned:
    """Reference audio as a time-aligned in-context condition (trainer convention)."""

    CATEGORY = "Koshi/Space"
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    DESCRIPTION = (
        "Time-aligned in-context reference audio for a2a IC-LoRAs trained with the LTX "
        "trainer (reference positions start at t=0, matching the target). Core's "
        "LTXVReferenceAudio uses the ID-LoRA negative-time convention instead. Wire: "
        "UNET → LoRA → this node → guider; feed the DRY recording, trimmed to the clip length."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference_audio": (
                    "AUDIO",
                    {
                        "tooltip": "The dry recording to transform. Same length as the clip, "
                        "starting at t=0 — token i of the reference sits at the same time "
                        "as token i of the target."
                    },
                ),
                "audio_vae": ("VAE", {"tooltip": "LTX audio VAE (returns normalized latents)."}),
                "reference_guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "0 = off (trainer inference had no such term). >0 adds an "
                        "extra no-reference pass per step and pushes the prediction away "
                        "from it, like the stock node's identity guidance.",
                    },
                ),
            }
        }

    def apply(self, model, positive, negative, reference_audio, audio_vae, reference_guidance_scale):
        if not _COMFY:  # pragma: no cover
            raise RuntimeError(
                "AKUSPACE Reference Audio (aligned) needs a ComfyUI core with LTXAVModel "
                f"(import failed: {_IMPORT_ERROR})"
            )
        sr = reference_audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 44100)
        wf = reference_audio["waveform"]
        if vae_sr != sr:
            wf = torchaudio.functional.resample(wf, sr, vae_sr)
        latents = audio_vae.encode(wf.movedim(1, -1))  # (b, c, t, f), normalized
        b, c, t, f = latents.shape
        tokens = latents.permute(0, 2, 1, 3).reshape(b, t, c * f)  # parity with the stock dict
        ref = {"tokens": tokens, "latents": latents, _FLAG: True}

        positive = node_helpers.conditioning_set_values(positive, {"ref_audio": ref})
        negative = node_helpers.conditioning_set_values(negative, {"ref_audio": ref})

        m = model.clone()
        scale = float(reference_guidance_scale)
        if scale > 0:

            def post_cfg_function(args):
                cond_pred = args["cond_denoised"]
                cond = args["cond"]
                cfg_result = args["denoised"]
                model_options = args["model_options"].copy()
                x = args["input"]
                sigma = args["sigma"]
                noref_cond = []
                for entry in cond:
                    new_entry = entry.copy()
                    mc = new_entry.get("model_conds", {}).copy()
                    mc.pop("ref_audio", None)
                    new_entry["model_conds"] = mc
                    noref_cond.append(new_entry)
                (pred_noref,) = comfy.samplers.calc_cond_batch(
                    args["model"], [noref_cond], x, sigma, model_options
                )
                return cfg_result + (cond_pred - pred_noref) * scale

            m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m, positive, negative)


NODE_CLASS_MAPPINGS = {"AKUSPACEReferenceAudioAligned": AKUSPACEReferenceAudioAligned}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AKUSPACEReferenceAudioAligned": "◉ AKUSPACE Reference Audio (aligned)",
}

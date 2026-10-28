# Klein + SAM region swap

Draft for a Comfy contest entry. One graph, three stages, each usable alone:

1. **Base image (Klein)** — FLUX 3 Image Klein generates a 1024 x 1024 frame from a prompt.
2. **Find the element (SAM)** — SAM 3.1 finds the named element (`drink`) from text and returns
   its mask and bounding box.
3. **Swap the region (Klein edit)** — Klein edits that region from an instruction
   (`hold a banana instead of the drink`), optionally guided by a reference image, and the
   result is composited back over the base through the SAM mask.

| file | what it is |
|---|---|
| `klein_sam_region_swap.json` | UI workflow: positioned nodes, three groups, two notes |
| `api/klein_sam_region_swap_api.json` | API prompt for `/prompt` (Genobi fills it) |
| `api/agent_slots.json` | machine-readable map of the agent slots below |
| `tools/graph_spec.py` | single source for both files: nodes, widgets, links, positions |
| `tools/build_klein_sam_region_swap.py` | regenerates both JSON files from the spec |
| `tools/validate_against_object_info.py` | checks both files against a live `/object_info` |

> **Status: structurally validated, never run.** No GPU was available. Everything below
> marked *unverified* needs the first Vast.ai run.

## Agent slots (Genobi)

| slot | node | input | default |
|---|---|---|---|
| base prompt | `4` CLIPTextEncode | `text` | cafe scene, glass of iced coffee in hand |
| element name | `21` CLIPTextEncode (SAM 3.1 encoder) | `text` | `drink` |
| instruction | `40` PrimitiveStringMultiline | `value` | `hold a banana instead of the drink` |
| base seed | `7` KSampler | `seed` | `20261028` |
| edit seed | `51` KSampler | `seed` | `20261029` |

Optional image inputs. Both are bypassed in the UI file and **absent from the API file**: a
`LoadImage` with a missing file fails prompt validation even behind a lazy switch, so Genobi
adds them only when it has a file.

| slot | node | wiring |
|---|---|---|
| external base image | `10` LoadImage | `11.on_true = ["10", 0]`, `11.switch = true` — stage 1 is skipped |
| reference image ("use this instead") | `49` LoadImage | `50["images.image1"] = ["49", 0]`; cite it as `<ref_image_1>` in the instruction |

Outputs: `9` base, `29` mask + box preview, `30` box as STRING, `46` final edit prompt,
`53` raw Klein edit, `55` swapped image.

The instruction is wrapped by two `StringConcatenate` nodes (`41`, `42`) into
`In <ref_image_0>, edit only the region <obj_1>: <instruction> Match the existing lighting… Keep
everything outside <obj_1> exactly as it is…`. The agent writes only the plain instruction.
SAM 3.1 also accepts `drink:2` to keep up to two detections (default is one per prompt).

## Klein variant: FLUX 3 Image Klein, distilled

ComfyUI 0.42.5 and templates 0.11.96 ship FLUX 3 Image Klein in core
(`minComfyUIVersion` 0.41.0, template date 2026-10-15, `openSource: true`), including an
edit path:

- `image_flux3_klein_distilled_t2i` / `image_flux3_klein_t2i` (base)
- `image_flux3_klein_distilled_image_edit` / `image_flux3_klein_image_edit` (base)

The edit path is `ReferenceMultiLatent` (core, `comfy_extras/nodes_edit_model.py`): the input
image becomes `<ref_image_0>`, further images `<ref_image_1>`, …, added to both positive and
negative; its `empty_latent` output sets the output size from the first reference. Region
targeting is `CreateBoundingBoxes` → `Flux3LayoutPrompt`, which appends a layout table so the
prompt can cite `<obj_1>`. This graph feeds the SAM box straight into that path, so the edit is
tied to the detected region by the model's own mechanism.

FLUX 3 was chosen over FLUX.2 Klein 4B/9B because it is the newest Klein with a core edit
path, and its layout table gives a native way to point the edit at the SAM box. The distilled
variant was chosen over base for speed on a rented box (16 steps, cfg 1, versus 50 steps).
One `UNETLoader` serves both stages.

Loader and sampler settings follow the distilled templates exactly: `UNETLoader` +
`CLIPLoader` (type `flux3`) + `VAELoader`, `EmptyFlux2LatentImage` 1024 x 1024,
`KSampler` 16 steps, cfg 1, `euler` / `simple`, empty negative (`CLIPTextEncode ""` for
text-to-image, `ConditioningZeroOut` for edit), `ReferenceMultiLatent` `resize` at 1.0 MP.
The template's optional prompt upsampler (`TextGenerate` on a Qwen 3.5 9B encoder) is left
out to save 9B of weights; Genobi's agent already writes full prompts.

### Weight files (do not download from here; fetch on the Vast box)

From [`black-forest-labs/flux-3-image-klein-comfy`](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy)
— **gated (licence click-through), licence "other"**: the box needs an HF token from an
account that has accepted it.

```
models/diffusion_models/flux3-image-klein-int8-convrot.safetensors   (distilled, int8)
models/text_encoders/qwen3vl_4b_int8_convrot.safetensors
models/vae/flux2-vae.safetensors
```

With more than 24 GB of VRAM the templates suggest `flux3-image-klein-bf16.safetensors`
instead (change node `1`). The repo also lists `-nvfp4` variants and the base model
(`flux3-image-klein-base-*`).

From [`Comfy-Org/sam3.1`](https://huggingface.co/Comfy-Org/sam3.1) (not gated):

```
models/checkpoints/sam3.1_multiplex_fp16.safetensors
```

File names come from the bundled templates' `properties.models` and were matched against the
two repos' file listings through the Hugging Face API (metadata only, nothing downloaded).

## SAM decision: core SAM 3.1, no custom pack

`SAM3_Detect` is a core node in ComfyUI 0.42.5 (`comfy_extras/nodes_sam3.py`), so nothing has
to be added to the deployment build. It does text-prompted detection: the element name is
encoded by `CLIPTextEncode` with the CLIP output of the SAM 3.1 checkpoint (exactly as the
bundled `utility_image_segment_sam3` template does), and the node returns:

- `masks` (MASK) — union of detections, refined by the SAM decoder (`refine_iterations` 2)
- `bboxes` (BOUNDING_BOX) — per frame, a list of `{x, y, width, height, score}` in pixels

The box is exposed as a STRING through `PreviewAny` (node `30`), which serialises any input.
Registry SAM 3 packs exist (`SAM3Segmentation`, `TBGSam3Segmentation`, …) but are not needed;
the text-prompt requirement is met in core, so no manual-mask fallback path was added. If one
is wanted later, a `LoadImage` MASK output can replace `22.masks` at node `23`.

Mask processing: `GrowMask` 24 px, then `MaskToImage` → `ImageBlur` (radius 15, sigma 6) →
`ImageToMask` for a soft edge. Core `FeatherMask` was considered and rejected: it fades the
mask towards the **canvas** border by fixed pixel counts, not along the mask's own edge.

## What was validated, and how

Validated on the local CPU ComfyUI 0.42.5 (frontend 1.55.18, templates 0.11.96):

1. **`/object_info`** (`tools/validate_against_object_info.py`, read-only GET): all 29 node
   classes exist; every API input name is declared, including the dynamic-combo child
   `resolution.megapixels` and the autogrow slot `images.image0`; every required input is
   present; every link hits an existing output slot with a compatible type; widget values are
   within range and in their combo lists; UI link tables agree with the node socket tables;
   UI `widgets_values` counts match each node's widget count.
2. **ComfyUI's own `execution.validate_prompt`**, run offline in the install's Python with
   core nodes only (no prompt queued on the server). The only errors are the four model files
   not installed on this Mac (`unet_name`, `clip_name`, `vae_name`, `ckpt_name`). This step
   caught one real bug, now fixed: list-valued widgets (`CreateBoundingBoxes.editor_state`)
   must be wrapped as `{"__value__": [...]}` in API format or they are read as links.
   The variant with both optional `LoadImage` nodes wired passes the same way (only
   file-not-found on the test image names).
3. **Source read** for behaviour that `object_info` cannot show: `SAM3_Detect` output format,
   `CreateBoundingBoxes` accepting a SAM `BOUNDING_BOX` (list of frames) as upstream boxes,
   `Flux3LayoutPrompt` numbering boxes as `obj_1…`, `ReferenceMultiLatent` sizing, and
   `ImageCompositeMasked` resizing the mask and (with `resize_source`) the edit.

Only read in templates or docs, not executed: the Klein sampler settings, the prompt token
conventions (`<ref_image_N>`, `<obj_N>`), and the SAM 3.1 `name:N` syntax.

Not checked: the UI file has not been opened in the frontend. It is generated in the same
format as the bundled templates (version 0.4, widget inputs carry `widget.name`), but a visual
load on the Vast box is the first test.

Rebuild and recheck after any change to `tools/graph_spec.py`:

```
python3 workflows/klein/tools/build_klein_sam_region_swap.py http://127.0.0.1:8189
python3 workflows/klein/tools/validate_against_object_info.py http://127.0.0.1:8189
```

## Open questions (unverified until a GPU run)

1. **Does Klein respect the region on its own?** The edit regenerates the whole frame from
   `<ref_image_0>`; the layout table and the "keep everything outside" suffix ask it to leave
   the rest alone. The composite (`54`) guarantees it does, at the cost of a possible seam.
   Compare `53` (raw) with `55` (composited). If the raw edit already keeps the rest intact,
   the composite can be dropped and the seam risk disappears.
2. **Is the mask big enough for the new object?** A banana is longer than a glass. The 24 px
   growth may clip it; the composite would then cut the new object off at the old outline.
   Options: raise `23.expand`, or composite through a mask of the SAM box instead of its shape.
3. **Empty detection.** If SAM finds nothing, the mask is empty (output equals the base) and
   the layout table is empty (the prompt loses `<obj_1>`). No guard yet; Genobi should read
   `30` and retry with another element name or a lower `22.threshold`.
4. **Non-square or external base images.** `ReferenceMultiLatent` resizes the reference to
   about 1 MP, so the edit can come back a few pixels off the base size; the composite stretches
   it back (`resize_source`). Fine for 1024 x 1024; check alignment for other sizes.
5. **Does FLUX 3 Klein have an edit path in 0.42.5?** Yes, per the bundled templates and core
   source; it has not been executed here.
6. **`CreateBoundingBoxes` canvas state in the UI.** In API runs the SAM box always wins
   (`last_incoming` is empty). In the UI the canvas keeps manual edits until the upstream box
   changes, which is useful for hand correction but can surprise on re-runs.
7. **Validation reaches stage 1 even when the switch selects an external base.** The switch is
   lazy, so Klein does not run, but the Klein model files must still be present on the box.

## How it plugs into the rooms story

The swapped frame (`55`) becomes the first frame of the next segment: Genobi passes it as the
start image of an LTX or MiniMax image-to-video segment, so an agent can change one object in
a room before the camera moves through it.

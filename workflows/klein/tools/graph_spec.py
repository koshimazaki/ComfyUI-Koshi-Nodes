"""Single source of truth for the Klein + SAM3 region-swap graph.

Both the UI workflow and the API prompt are generated from NODES below by
build_klein_sam_region_swap.py, so the two files can never drift apart.

Each node entry:
    id, type, pos, size, title, mode (0 = active, 4 = bypassed), group,
    widgets: ordered (name, value) pairs in object_info widget order.
             Names starting with "~" are UI-only widget slots (for example
             KSampler's control_after_generate) and are left out of the API.
    links:   {input_name: (source_node_id, source_output_slot)}
    api:     False to leave the node out of the API prompt (notes, bypassed
             optional loaders that Genobi adds only when it has a file).
"""

BASE_PROMPT = (
    "Cinematic medium shot of a young woman sitting at a sunlit cafe table by a "
    "window, holding a tall glass of iced coffee in her right hand at chest height, "
    "relaxed smile, soft natural light from the left, shallow depth of field, "
    "35mm film look. No text, no logos, no watermark."
)
ELEMENT = "drink"
INSTRUCTION = "hold a banana instead of the drink"
EDIT_PREFIX = "In <ref_image_0>, edit only the region <obj_1>:"
EDIT_SUFFIX = (
    "Match the existing lighting, shadows, skin tone and grip of the hand. "
    "Keep everything outside <obj_1> exactly as it is: same person, pose, "
    "framing and background."
)

# Klein files, as named in the bundled FLUX 3 Image Klein distilled templates.
UNET = "flux3-image-klein-int8-convrot.safetensors"
TEXT_ENCODER = "qwen3vl_4b_int8_convrot.safetensors"
VAE = "flux2-vae.safetensors"
SAM_CKPT = "sam3.1_multiplex_fp16.safetensors"

G1 = "1 · Base image (Klein)"
G2 = "2 · Find the element (SAM)"
G3 = "3 · Swap the region (Klein edit)"

GROUPS = [
    {"title": G1, "bounding": [0, 0, 1800, 640], "color": "#3f789e"},
    {"title": G2, "bounding": [0, 680, 1800, 600], "color": "#8e44ad"},
    {"title": G3, "bounding": [1840, 0, 1760, 1280], "color": "#27ae60"},
]

AGENT_COLOUR = {"color": "#232", "bgcolor": "#353"}
OPTIONAL_COLOUR = {"color": "#322", "bgcolor": "#533"}

BASE = (11, 0)  # ComfySwitchNode output: the image every later stage works on
SAM_BOXES = (22, 1)

NODES = [
    # ---------------- 1 · Base image (Klein) ----------------
    dict(id=1, type="UNETLoader", group=G1, pos=[20, 60], size=[340, 82],
         title="Klein diffusion model (distilled)",
         widgets=[("unet_name", UNET), ("weight_dtype", "default")]),
    dict(id=2, type="CLIPLoader", group=G1, pos=[20, 180], size=[340, 106],
         title="Klein text encoder",
         widgets=[("clip_name", TEXT_ENCODER), ("type", "flux3"), ("device", "default")]),
    dict(id=3, type="VAELoader", group=G1, pos=[20, 320], size=[340, 58],
         widgets=[("vae_name", VAE)]),
    dict(id=4, type="CLIPTextEncode", group=G1, pos=[390, 60], size=[420, 210],
         title="AGENT · base prompt", colour=AGENT_COLOUR,
         widgets=[("text", BASE_PROMPT)], links={"clip": (2, 0)}),
    dict(id=5, type="CLIPTextEncode", group=G1, pos=[390, 300], size=[420, 110],
         title="Negative (empty, as in the template)",
         widgets=[("text", "")], links={"clip": (2, 0)}),
    dict(id=6, type="EmptyFlux2LatentImage", group=G1, pos=[390, 440], size=[300, 106],
         widgets=[("width", 1024), ("height", 1024), ("batch_size", 1)]),
    dict(id=7, type="KSampler", group=G1, pos=[840, 60], size=[300, 262],
         title="Klein sampler · base (seed exposed)",
         widgets=[("seed", 20261028), ("~control_after_generate", "fixed"), ("steps", 16),
                  ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "simple"),
                  ("denoise", 1.0)],
         links={"model": (1, 0), "positive": (4, 0), "negative": (5, 0),
                "latent_image": (6, 0)}),
    dict(id=8, type="VAEDecode", group=G1, pos=[840, 360], size=[210, 46],
         links={"samples": (7, 0), "vae": (3, 0)}),
    dict(id=10, type="LoadImage", group=G1, pos=[1170, 60], size=[300, 330], mode=4,
         title="OPTIONAL · external base image", colour=OPTIONAL_COLOUR, api=False,
         widgets=[("image", "base.png"), ("~upload", "image")]),
    dict(id=11, type="ComfySwitchNode", group=G1, pos=[1170, 430], size=[270, 80],
         title="Base source · false = Klein, true = loaded image",
         widgets=[("switch", False)],
         links={"on_false": (8, 0)}, link_types={"on_false": "IMAGE", "on_true": "IMAGE"},
         out_types=["IMAGE"]),
    dict(id=9, type="SaveImage", group=G1, pos=[1500, 60], size=[280, 320],
         widgets=[("filename_prefix", "klein_sam/base")], links={"images": BASE}),

    # ---------------- 2 · Find the element (SAM) ----------------
    dict(id=20, type="CheckpointLoaderSimple", group=G2, pos=[20, 740], size=[340, 98],
         title="SAM 3.1 checkpoint", widgets=[("ckpt_name", SAM_CKPT)]),
    dict(id=21, type="CLIPTextEncode", group=G2, pos=[20, 870], size=[340, 120],
         title="AGENT · element name", colour=AGENT_COLOUR,
         widgets=[("text", ELEMENT)], links={"clip": (20, 1)}),
    dict(id=22, type="SAM3_Detect", group=G2, pos=[390, 740], size=[300, 190],
         widgets=[("threshold", 0.5), ("refine_iterations", 2), ("individual_masks", False)],
         links={"model": (20, 0), "image": BASE, "conditioning": (21, 0)}),
    dict(id=23, type="GrowMask", group=G2, pos=[720, 740], size=[260, 82],
         title="Grow mask (room for the new object)",
         widgets=[("expand", 24), ("tapered_corners", True)], links={"mask": (22, 0)}),
    dict(id=24, type="MaskToImage", group=G2, pos=[720, 860], size=[220, 26],
         links={"mask": (23, 0)}),
    dict(id=25, type="ImageBlur", group=G2, pos=[720, 920], size=[260, 82],
         title="Feather edge (blur)",
         widgets=[("blur_radius", 15), ("sigma", 6.0)], links={"image": (24, 0)}),
    dict(id=26, type="ImageToMask", group=G2, pos=[720, 1040], size=[260, 58],
         title="Feathered mask", widgets=[("channel", "red")], links={"image": (25, 0)}),
    dict(id=27, type="ImageBlend", group=G2, pos=[1010, 740], size=[260, 102],
         title="Mask over base",
         widgets=[("blend_factor", 0.45), ("blend_mode", "screen")],
         links={"image1": BASE, "image2": (24, 0)}),
    dict(id=28, type="DrawBBoxes", group=G2, pos=[1010, 880], size=[260, 46],
         links={"bboxes": SAM_BOXES, "image": (27, 0)}),
    dict(id=29, type="PreviewImage", group=G2, pos=[1300, 740], size=[480, 480],
         title="Preview · mask + box", links={"images": (28, 0)}),
    dict(id=30, type="PreviewAny", group=G2, pos=[1010, 970], size=[270, 250],
         title="Box as STRING (x, y, width, height, score)", links={"source": SAM_BOXES},
         link_types={"source": "*"}),

    # ---------------- 3 · Swap the region (Klein edit) ----------------
    dict(id=40, type="PrimitiveStringMultiline", group=G3, pos=[1860, 60], size=[400, 160],
         title="AGENT · instruction", colour=AGENT_COLOUR,
         widgets=[("value", INSTRUCTION)]),
    dict(id=41, type="StringConcatenate", group=G3, pos=[1860, 250], size=[400, 200],
         title="Prefix: bind the edit to <obj_1>",
         widgets=[("string_a", EDIT_PREFIX), ("string_b", ""), ("delimiter", " ")],
         links={"string_b": (40, 0)}),
    dict(id=42, type="StringConcatenate", group=G3, pos=[1860, 480], size=[400, 200],
         title="Suffix: keep the rest",
         widgets=[("string_a", ""), ("string_b", EDIT_SUFFIX), ("delimiter", " ")],
         links={"string_a": (41, 0)}),
    dict(id=43, type="GetImageSize", group=G3, pos=[1860, 720], size=[220, 70],
         links={"image": BASE}),
    dict(id=44, type="CreateBoundingBoxes", group=G3, pos=[2290, 60], size=[400, 560],
         title="SAM box → FLUX 3 layout element",
         widgets=[("width", 1024), ("height", 1024), ("editor_state", []),
                  ("last_incoming", [])],
         links={"background": BASE, "bboxes": SAM_BOXES, "width": (43, 0), "height": (43, 1)},
         link_types={"bboxes": "BOUNDING_BOX,ARRAY,STRING"}),
    dict(id=45, type="Flux3LayoutPrompt", group=G3, pos=[2290, 660], size=[400, 200],
         widgets=[("prompt", "")], links={"prompt": (42, 0), "element": (44, 2)}),
    dict(id=46, type="PreviewAny", group=G3, pos=[2290, 900], size=[400, 340],
         title="Final edit prompt", links={"source": (45, 0)}, link_types={"source": "*"}),
    dict(id=47, type="CLIPTextEncode", group=G3, pos=[2720, 60], size=[270, 90],
         title="Edit prompt encode", widgets=[("text", "")],
         links={"clip": (2, 0), "text": (45, 0)}),
    dict(id=48, type="ConditioningZeroOut", group=G3, pos=[2720, 190], size=[270, 26],
         links={"conditioning": (47, 0)}),
    dict(id=49, type="LoadImage", group=G3, pos=[2720, 260], size=[270, 330], mode=4,
         title="OPTIONAL · reference image (<ref_image_1>)", colour=OPTIONAL_COLOUR,
         api=False, widgets=[("image", "reference.png"), ("~upload", "image")]),
    dict(id=50, type="ReferenceMultiLatent", group=G3, pos=[3020, 60], size=[280, 170],
         title="Set Reference Multi Latent (base = <ref_image_0>)",
         widgets=[("resolution", "resize"), ("resolution.megapixels", 1.0)],
         links={"positive": (47, 0), "negative": (48, 0), "vae": (3, 0),
                "images.image0": BASE, "images.image1": (49, 0)},
         extra_sockets=[("images.image2", "IMAGE")]),
    dict(id=51, type="KSampler", group=G3, pos=[3020, 270], size=[280, 262],
         title="Klein sampler · edit",
         widgets=[("seed", 20261029), ("~control_after_generate", "fixed"), ("steps", 16),
                  ("cfg", 1.0), ("sampler_name", "euler"), ("scheduler", "simple"),
                  ("denoise", 1.0)],
         links={"model": (1, 0), "positive": (50, 0), "negative": (50, 1),
                "latent_image": (50, 2)}),
    dict(id=52, type="VAEDecode", group=G3, pos=[3020, 570], size=[210, 46],
         links={"samples": (51, 0), "vae": (3, 0)}),
    dict(id=53, type="PreviewImage", group=G3, pos=[3020, 660], size=[280, 320],
         title="Preview · raw Klein edit (before composite)", links={"images": (52, 0)}),
    dict(id=54, type="ImageCompositeMasked", group=G3, pos=[3330, 60], size=[250, 140],
         title="Composite edit over base with the SAM mask",
         widgets=[("x", 0), ("y", 0), ("resize_source", True)],
         links={"destination": BASE, "source": (52, 0), "mask": (26, 0)}),
    dict(id=55, type="SaveImage", group=G3, pos=[3330, 240], size=[250, 320],
         widgets=[("filename_prefix", "klein_sam/swap")], links={"images": (54, 0)}),
]

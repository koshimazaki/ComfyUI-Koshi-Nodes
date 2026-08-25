# Scripts

Setup scripts and public workflow tools live here so the package root stays
readable.

## Setup

| script | purpose |
|---|---|
| `setup_comfyui_flux.sh` | ComfyUI + Koshi Nodes + a selected FLUX model preset. |
| `akuspace/setup_ltx25_box.sh` | Vast.ai/RunPod setup for LTX-2.5, AKUSPACE, required custom nodes, inputs, and workflow staging. |

```bash
bash scripts/setup_comfyui_flux.sh --runpod --gguf
bash scripts/akuspace/setup_ltx25_box.sh --vast --token=hf_xxx --lora=/workspace/akuspace-ltx25-v0.5.safetensors
```

Run either script with `--help` before provisioning a paid box. LTX-2.5 is gated
on Hugging Face; accept its licence with the same account as the supplied token.

## AKUSPACE workflow tools

The three entry points use only the Python standard library and the API-format
graphs in [`workflows/akuspace/`](../workflows/akuspace/):

| tool | purpose |
|---|---|
| `verify_workflows.py` | Check graph format, links, LoRA/reference wiring, empty audio targets, CFG values, outputs, and reachability. |
| `run_batch.py` | Queue selected graphs sequentially and print a compact pass/fail summary. |
| `run_ab.py` | Run the same graph and seed with AKUSPACE strength `1` and `0`; output prefixes are isolated automatically. |

Start with the offline checks:

```bash
python3 scripts/akuspace/verify_workflows.py
python3 scripts/akuspace/run_batch.py --dry-run
python3 scripts/akuspace/run_ab.py --dry-run
```

Against the box created by `setup_ltx25_box.sh`:

```bash
export COMFY_URL=http://127.0.0.1:8189
python3 scripts/akuspace/run_batch.py
python3 scripts/akuspace/run_ab.py
```

Both runners accept `--set NODE_ID.INPUT=value` for input filenames, prompts,
seeds, or other graph values. Use `--help` for filtering and timeout options.

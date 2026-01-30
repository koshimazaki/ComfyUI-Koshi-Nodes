#!/bin/bash
# ComfyUI + FLUX + Koshi Nodes Setup Script
#
# Quick install (RunPod):
#   curl -sL https://raw.githubusercontent.com/koshimazaki/ComfyUI-Koshi-Nodes/main/setup_comfyui_flux.sh | bash -s -- --runpod --minimal
#
# Full install with Tailscale:
#   curl -sL https://raw.githubusercontent.com/koshimazaki/ComfyUI-Koshi-Nodes/main/setup_comfyui_flux.sh | bash -s -- --runpod --full --tailscale
#
# Options:
#   --runpod|--local    Environment (auto-detects if not specified)
#   --minimal           Schnell + FP8 T5 (~17GB)
#   --full              Schnell + Dev + FP16 T5 (~46GB)
#   --fp8               FP8 optimized (~17GB)
#   --gguf              Q4 quantized (~6GB)
#   --skip-models       Don't download models
#   --tailscale         Setup Tailscale SSH for remote access
#   --token=XXX         HuggingFace token
#   --authkey=XXX       Tailscale auth key (optional)
#
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parse arguments
MODEL_PRESET="menu"
INSTALL_MODE=""
SETUP_TAILSCALE=false
MODELS_ONLY=false
HF_TOKEN="${HF_TOKEN:-}"
TS_AUTHKEY="${TS_AUTHKEY:-}"

for arg in "$@"; do
    case $arg in
        --runpod) INSTALL_MODE="runpod" ;;
        --local) INSTALL_MODE="local" ;;
        --minimal) MODEL_PRESET="minimal" ;;
        --full) MODEL_PRESET="full" ;;
        --fp8) MODEL_PRESET="fp8" ;;
        --gguf) MODEL_PRESET="gguf" ;;
        --skip-models) MODEL_PRESET="skip" ;;
        --models-only) MODELS_ONLY=true ;;
        --tailscale) SETUP_TAILSCALE=true ;;
        --token=*) HF_TOKEN="${arg#*=}" ;;
        --authkey=*) TS_AUTHKEY="${arg#*=}" ;;
        --help)
            printf "Usage: ./setup_comfyui_flux.sh [OPTIONS]\n\n"
            printf "Options:\n"
            printf "  --runpod        RunPod environment (full install)\n"
            printf "  --local         Local environment (nodes only)\n"
            printf "  --minimal       Schnell + FP8 T5 (~17GB)\n"
            printf "  --full          Schnell + Dev + FP16 T5 (~46GB)\n"
            printf "  --fp8           FP8 optimized models (~17GB)\n"
            printf "  --gguf          Q4 quantized (~6GB)\n"
            printf "  --skip-models   Skip model downloads\n"
            printf "  --models-only   Only download models (skip ComfyUI/nodes)\n"
            printf "  --tailscale     Setup Tailscale SSH\n"
            printf "  --token=XXX     HuggingFace token\n"
            printf "  --authkey=XXX   Tailscale auth key\n"
            exit 0
            ;;
    esac
done

# Auto-detect if piped (non-interactive)
INTERACTIVE=true
if [ ! -t 0 ] || [ ! -t 1 ]; then
    INTERACTIVE=false
    # Running via curl pipe - need defaults
    [ -z "$INSTALL_MODE" ] && [ -d "/workspace" ] && INSTALL_MODE="runpod"
    [ "$MODEL_PRESET" = "menu" ] && MODEL_PRESET="minimal"
fi

# Early check: warn if HF token needed but missing in non-interactive mode
if [ "$INTERACTIVE" = false ] && [ "$MODEL_PRESET" != "skip" ] && [ -z "$HF_TOKEN" ]; then
    printf "${YELLOW}Warning:${NC} No HF token provided. FLUX models require authentication.\n"
    printf "Either:\n"
    printf "  1. Add --token=YOUR_TOKEN to the command\n"
    printf "  2. Set HF_TOKEN env var: export HF_TOKEN=xxx && curl ...\n"
    printf "  3. Use --skip-models to skip model downloads\n"
    printf "\nContinuing without token (some downloads may fail)...\n\n"
fi

# ASCII Art Banner
printf "\n"
printf "█▄▀ █▀█ █▀▀ █ █ █    █▄ █ █▀█ █▀▄ █▀▀ █▀▀\n"
printf "█ █ █▄█ ▄▄█ █▀█ █    █ ▀█ █▄█ █▄▀ ██▄ ▄▄█\n"
printf "░░█ ComfyUI + FLUX Setup █░░\n"
printf "\n"
printf "╭─────────────────────────────────────────────╮\n"
printf "│ github.com/koshimazaki/ComfyUI-Koshi-Nodes  │\n"
printf "╰─────────────────────────────────────────────╯\n\n"

# Installation mode selection
if [ -z "$INSTALL_MODE" ]; then
    # Auto-detect environment
    if [ -d "/workspace" ]; then
        printf "RunPod environment detected.\n\n"
    fi
    printf "Select installation mode:\n"
    printf "  1) RunPod   - Full install to /workspace/ComfyUI\n"
    printf "  2) Local    - Add nodes to existing ComfyUI\n"
    printf "\nChoice [1-2]: "
    read -r mode_choice
    case $mode_choice in
        1) INSTALL_MODE="runpod" ;;
        2) INSTALL_MODE="local" ;;
        *) [ -d "/workspace" ] && INSTALL_MODE="runpod" || INSTALL_MODE="local" ;;
    esac
fi

# Set paths based on mode
if [ "$INSTALL_MODE" = "runpod" ]; then
    COMFY_DIR="/workspace/ComfyUI"
    RUNPOD=true
    FULL_INSTALL=true
    printf "\n[Mode: RunPod] Install: %s\n\n" "$COMFY_DIR"
else
    RUNPOD=false
    FULL_INSTALL=false
    # Find existing ComfyUI installation
    if [ -n "$COMFY_PATH" ]; then
        COMFY_DIR="$COMFY_PATH"
    elif [ -d "$HOME/ComfyUI" ]; then
        COMFY_DIR="$HOME/ComfyUI"
    elif [ -d "/opt/ComfyUI" ]; then
        COMFY_DIR="/opt/ComfyUI"
    else
        printf "Enter path to your ComfyUI installation: "
        read -r COMFY_DIR
    fi
    printf "\n[Mode: Local] ComfyUI: %s\n\n" "$COMFY_DIR"

    # Verify ComfyUI exists
    if [ ! -f "$COMFY_DIR/main.py" ]; then
        printf "${RED}Error:${NC} ComfyUI not found at %s\n" "$COMFY_DIR"
        printf "Make sure ComfyUI is installed first.\n"
        exit 1
    fi
fi

# Models-only mode: set COMFY_DIR for model downloads
if [ "$MODELS_ONLY" = true ]; then
    if [ -d "/workspace/ComfyUI" ]; then
        COMFY_DIR="/workspace/ComfyUI"
    elif [ -n "$COMFY_PATH" ]; then
        COMFY_DIR="$COMFY_PATH"
    elif [ -d "$HOME/ComfyUI" ]; then
        COMFY_DIR="$HOME/ComfyUI"
    else
        printf "${RED}Error:${NC} ComfyUI not found. Specify COMFY_PATH or install ComfyUI first.\n"
        exit 1
    fi
    printf "[Models Only Mode] Using: %s\n\n" "$COMFY_DIR"
fi

if [ "$MODELS_ONLY" = false ]; then
    # Check prerequisites
    printf "[1/7] Checking prerequisites...\n"
    command -v python3 >/dev/null || { printf "Error: python3 required\n"; exit 1; }
    command -v pip3 >/dev/null || { printf "Error: pip3 required\n"; exit 1; }
    command -v git >/dev/null || { printf "Error: git required\n"; exit 1; }
    printf "  OK\n"

    # Install ComfyUI (RunPod only)
    printf "\n[2/7] Installing ComfyUI...\n"
    if [ "$FULL_INSTALL" = true ]; then
        mkdir -p "$(dirname "$COMFY_DIR")"
        if [ ! -d "$COMFY_DIR" ]; then
            git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
        fi
        cd "$COMFY_DIR"
        if [ "$RUNPOD" = true ]; then
            pip3 install --break-system-packages -r requirements.txt 2>/dev/null || pip3 install -r requirements.txt
        else
            pip3 install -r requirements.txt
        fi
        printf "  ${GREEN}Done${NC}\n"
    else
        printf "  Skipped (using existing ComfyUI)\n"
    fi

    # Install Koshi Nodes
    printf "\n[3/7] Installing Koshi Nodes...\n"
    mkdir -p "$COMFY_DIR/custom_nodes"
    cd "$COMFY_DIR/custom_nodes"
    if [ ! -d "Koshi-Nodes" ]; then
        git clone https://github.com/koshimazaki/ComfyUI-Koshi-Nodes.git Koshi-Nodes
    fi
    if [ "$RUNPOD" = true ]; then
        pip3 install --break-system-packages torch numpy Pillow scipy opencv-python moderngl 2>/dev/null || pip3 install torch numpy Pillow scipy opencv-python moderngl
    else
        pip3 install torch numpy Pillow scipy opencv-python moderngl
    fi
    printf "  ${GREEN}Done${NC}\n"
fi

# Download FLUX models
if [ "$MODELS_ONLY" = true ]; then
    printf "[1/1] Downloading FLUX models...\n"
else
    printf "\n[4/7] Downloading FLUX models...\n"
fi
cd "$COMFY_DIR/models"
mkdir -p checkpoints vae clip unet

# Model selection menu
if [ "$MODEL_PRESET" = "menu" ]; then
    printf "\nSelect model preset:\n"
    printf "  1) Minimal  - Schnell + FP8 T5 (~17GB) - Fast, low VRAM\n"
    printf "  2) Full     - Schnell + Dev + FP16 T5 (~46GB) - Best quality\n"
    printf "  3) FP8      - FP8 optimized models (~17GB) - Balanced\n"
    printf "  4) GGUF 4B  - Q4 quantized (~6GB) - Ultra low VRAM\n"
    printf "  5) Skip     - Don't download models\n"
    [ "$FULL_INSTALL" = false ] && printf "  (Recommended: Skip if you already have models)\n"
    printf "\nChoice [1-5]: "
    read -r choice
    case $choice in
        1) MODEL_PRESET="minimal" ;;
        2) MODEL_PRESET="full" ;;
        3) MODEL_PRESET="fp8" ;;
        4) MODEL_PRESET="gguf" ;;
        5) MODEL_PRESET="skip" ;;
        *) [ "$FULL_INSTALL" = true ] && MODEL_PRESET="minimal" || MODEL_PRESET="skip" ;;
    esac
fi

download_if_missing() {
    local file=$1 url=$2 desc=$3 auth=${4:-false}
    if [ ! -f "$file" ]; then
        printf "  ${YELLOW}Downloading${NC} %s...\n" "$desc"
        if [ "$auth" = true ] && [ -n "$HF_TOKEN" ]; then
            curl -L --progress-bar -H "Authorization: Bearer $HF_TOKEN" -o "$file" "$url"
        else
            curl -L --progress-bar -o "$file" "$url"
        fi
        # Check if download failed (got HTML error page)
        if [ -f "$file" ] && head -c 50 "$file" 2>/dev/null | grep -q "Access to model"; then
            printf "  ${RED}Auth required${NC}: %s\n" "$desc"
            rm -f "$file"
            return 1
        fi
        printf "  ${GREEN}Done${NC}: %s\n" "$desc"
    else
        printf "  ${GREEN}Exists${NC}: %s\n" "$desc"
    fi
}

HF_FLUX="https://huggingface.co/black-forest-labs"
HF_CLIP="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main"
HF_FP8="https://huggingface.co/Kijai/flux-fp8/resolve/main"
HF_GGUF="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main"

if [ "$MODEL_PRESET" != "skip" ]; then
    # Check for HF token (needed for FLUX models)
    if [ -z "$HF_TOKEN" ] && [ "$INTERACTIVE" = true ]; then
        printf "\n${YELLOW}FLUX models require HuggingFace authentication.${NC}\n"
        printf "Get token at: ${CYAN}https://huggingface.co/settings/tokens${NC}\n"
        printf "Enter HF token (or press Enter to skip FLUX checkpoints): "
        read -r HF_TOKEN
    fi
    # Always download VAE and CLIP-L (VAE needs auth, CLIP doesn't)
    download_if_missing "vae/ae.safetensors" \
        "$HF_FLUX/FLUX.1-schnell/resolve/main/ae.safetensors" "FLUX VAE (335MB)" true
    download_if_missing "clip/clip_l.safetensors" \
        "$HF_CLIP/clip_l.safetensors" "CLIP-L (246MB)"

    case $MODEL_PRESET in
        minimal)
            download_if_missing "checkpoints/flux1-schnell.safetensors" \
                "$HF_FLUX/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" "FLUX.1 Schnell (12GB)" true
            download_if_missing "clip/t5xxl_fp8_e4m3fn.safetensors" \
                "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8 (4.9GB)"
            ;;
        full)
            download_if_missing "checkpoints/flux1-schnell.safetensors" \
                "$HF_FLUX/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" "FLUX.1 Schnell (12GB)" true
            download_if_missing "checkpoints/flux1-dev.safetensors" \
                "$HF_FLUX/FLUX.1-dev/resolve/main/flux1-dev.safetensors" "FLUX.1 Dev (24GB)" true
            download_if_missing "clip/t5xxl_fp16.safetensors" \
                "$HF_CLIP/t5xxl_fp16.safetensors" "T5-XXL FP16 (9.8GB)"
            ;;
        fp8)
            download_if_missing "unet/flux1-dev-fp8.safetensors" \
                "$HF_FP8/flux1-dev-fp8.safetensors" "FLUX.1 Dev FP8 (12GB)"
            download_if_missing "clip/t5xxl_fp8_e4m3fn.safetensors" \
                "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8 (4.9GB)"
            ;;
        gguf)
            download_if_missing "unet/flux1-dev-Q4_K_S.gguf" \
                "$HF_GGUF/flux1-dev-Q4_K_S.gguf" "FLUX.1 Dev Q4 GGUF (5.6GB)"
            download_if_missing "clip/t5xxl_fp8_e4m3fn.safetensors" \
                "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8 (4.9GB)"
            ;;
    esac
    printf "  ${GREEN}Model downloads complete${NC}\n"
else
    printf "  ${YELLOW}Skipping model downloads${NC}\n"
fi

if [ "$MODELS_ONLY" = false ]; then
    # Install extra nodes
    printf "\n[5/7] Installing extra nodes...\n"
    cd "$COMFY_DIR/custom_nodes"
    [ ! -d "ComfyUI-Manager" ] && git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    [ ! -d "ComfyUI-VideoHelperSuite" ] && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
    # Install GGUF support if using GGUF models
    if [ "$MODEL_PRESET" = "gguf" ]; then
        [ ! -d "ComfyUI-GGUF" ] && git clone https://github.com/city96/ComfyUI-GGUF.git
        if [ "$RUNPOD" = true ]; then
            pip3 install --break-system-packages gguf 2>/dev/null || pip3 install gguf
        else
            pip3 install gguf
        fi
    fi
    if [ "$RUNPOD" = true ]; then
        pip3 install --break-system-packages gitpython 2>/dev/null || true
    fi
    printf "  ${GREEN}Done${NC}\n"

    # Create launcher
    printf "\n[6/7] Creating launcher...\n"
    cat > "$COMFY_DIR/run.sh" << EOF
#!/bin/bash
cd "$COMFY_DIR"
python3 main.py --listen 0.0.0.0 --port \${PORT:-8188} "\$@"
EOF
    chmod +x "$COMFY_DIR/run.sh"
    printf "  ${GREEN}Done${NC}\n"

    # Tailscale setup (optional)
    if [ "$SETUP_TAILSCALE" = true ]; then
        printf "\n[7/7] Setting up Tailscale...\n"

        # Check if already installed
        if command -v tailscale >/dev/null 2>&1; then
            printf "  Tailscale already installed\n"
        else
            printf "  Installing Tailscale...\n"
            curl -fsSL https://tailscale.com/install.sh | sh
        fi

        # Start daemon (userspace mode for containers)
        if ! pgrep -x tailscaled >/dev/null; then
            printf "  Starting tailscaled daemon...\n"
            tailscaled --tun=userspace-networking --state=/workspace/tailscale.state &
            sleep 2
        fi

        # Connect with SSH enabled
        if [ -n "$TS_AUTHKEY" ]; then
            printf "  Authenticating with auth key...\n"
            tailscale up --ssh --authkey="$TS_AUTHKEY"
        else
            printf "  ${YELLOW}Run this to authenticate:${NC}\n"
            printf "    tailscale up --ssh\n"
            printf "  Then click the URL to login.\n"
        fi

        # Show IP if connected
        if tailscale status >/dev/null 2>&1; then
            TS_IP=$(tailscale ip -4 2>/dev/null || echo "pending")
            printf "  ${GREEN}Tailscale IP:${NC} %s\n" "$TS_IP"
        fi
    else
        printf "\n[7/7] Skipping Tailscale (use --tailscale to enable)\n"
    fi
fi

printf "\n"
printf "╭─────────────────────────────────────────────╮\n"
printf "│  Installation Complete!                     │\n"
printf "╰─────────────────────────────────────────────╯\n"
printf "\n"
if [ "$MODELS_ONLY" = true ]; then
    printf "Models downloaded to:\n"
    printf "  %s/models/\n" "$COMFY_DIR"
elif [ "$FULL_INSTALL" = true ]; then
    printf "Start ComfyUI:\n"
    printf "  %s/run.sh\n\n" "$COMFY_DIR"
    printf "Access:\n"
    printf "  Local:     http://127.0.0.1:8188\n"
    if [ "$SETUP_TAILSCALE" = true ]; then
        TS_IP=$(tailscale ip -4 2>/dev/null || echo "<tailscale-ip>")
        printf "  Tailscale: http://%s:8188\n" "$TS_IP"
    fi
else
    printf "Koshi Nodes installed to:\n"
    printf "  %s/custom_nodes/Koshi-Nodes\n" "$COMFY_DIR"
    printf "\nRestart ComfyUI to load the new nodes.\n"
fi
printf "\n"

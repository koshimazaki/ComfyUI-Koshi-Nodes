#!/bin/bash
# ComfyUI + FLUX + Koshi Nodes Setup Script
#
# Interactive: ./setup_comfyui_flux.sh
# Non-interactive: ./setup_comfyui_flux.sh --runpod --klein --token=hf_xxx
#
set -e

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

print_banner() {
    clear 2>/dev/null || true
    printf "\n"
    printf "${CYAN}"
    printf "    ██╗  ██╗ ██████╗ ███████╗██╗  ██╗██╗    ███╗   ██╗ ██████╗ ██████╗ ███████╗███████╗\n"
    printf "    ██║ ██╔╝██╔═══██╗██╔════╝██║  ██║██║    ████╗  ██║██╔═══██╗██╔══██╗██╔════╝██╔════╝\n"
    printf "    █████╔╝ ██║   ██║███████╗███████║██║    ██╔██╗ ██║██║   ██║██║  ██║█████╗  ███████╗\n"
    printf "    ██╔═██╗ ██║   ██║╚════██║██╔══██║██║    ██║╚██╗██║██║   ██║██║  ██║██╔══╝  ╚════██║\n"
    printf "    ██║  ██╗╚██████╔╝███████║██║  ██║██║    ██║ ╚████║╚██████╔╝██████╔╝███████╗███████║\n"
    printf "    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝    ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝\n"
    printf "${NC}\n"
    printf "    ${DIM}ComfyUI + FLUX Setup${NC}\n"
    printf "    ${DIM}github.com/koshimazaki/ComfyUI-Koshi-Nodes${NC}\n"
    printf "\n"
    printf "    ────────────────────────────────────────────────────────────────────────────────\n\n"
}

print_done() {
    printf "\n"
    printf "${GREEN}"
    printf "    ██████╗  ██████╗ ███╗   ██╗███████╗██╗\n"
    printf "    ██╔══██╗██╔═══██╗████╗  ██║██╔════╝██║\n"
    printf "    ██║  ██║██║   ██║██╔██╗ ██║█████╗  ██║\n"
    printf "    ██║  ██║██║   ██║██║╚██╗██║██╔══╝  ╚═╝\n"
    printf "    ██████╔╝╚██████╔╝██║ ╚████║███████╗██╗\n"
    printf "    ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝\n"
    printf "${NC}\n"
}

print_step() {
    printf "\n${BOLD}[$1]${NC} $2\n"
}

print_ok() {
    printf "    ${GREEN}✓${NC} $1\n"
}

print_skip() {
    printf "    ${DIM}○ $1${NC}\n"
}

print_warn() {
    printf "    ${YELLOW}!${NC} $1\n"
}

print_err() {
    printf "    ${RED}✗${NC} $1\n"
}

# ═══════════════════════════════════════════════════════════════════════════════
# PARSE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
INSTALL_COMFY=true
INSTALL_NODES=true
INSTALL_MODELS=true
INSTALL_TAILSCALE=false
MODEL_PRESET=""
INSTALL_MODE=""
HF_TOKEN="${HF_TOKEN:-}"
TS_AUTHKEY="${TS_AUTHKEY:-}"

for arg in "$@"; do
    case $arg in
        --runpod) INSTALL_MODE="runpod" ;;
        --local) INSTALL_MODE="local" ;;
        --klein) MODEL_PRESET="klein" ;;
        --minimal) MODEL_PRESET="minimal" ;;
        --full) MODEL_PRESET="full" ;;
        --fp8) MODEL_PRESET="fp8" ;;
        --gguf) MODEL_PRESET="gguf" ;;
        --skip-models) INSTALL_MODELS=false ;;
        --models-only) INSTALL_COMFY=false; INSTALL_NODES=false ;;
        --nodes-only) INSTALL_COMFY=false; INSTALL_MODELS=false ;;
        --tailscale) INSTALL_TAILSCALE=true ;;
        --token=*) HF_TOKEN="${arg#*=}" ;;
        --authkey=*) TS_AUTHKEY="${arg#*=}" ;;
        --help)
            printf "Usage: ./setup_comfyui_flux.sh [OPTIONS]\n\n"
            printf "Presets:\n"
            printf "  --klein         FLUX.2-klein-4B (~13GB)\n"
            printf "  --minimal       Schnell + FP8 T5 (~17GB)\n"
            printf "  --full          Schnell + Dev + FP16 T5 (~46GB)\n"
            printf "  --fp8           FP8 optimized (~17GB)\n"
            printf "  --gguf          Q4 quantized (~6GB)\n\n"
            printf "Options:\n"
            printf "  --runpod        RunPod environment\n"
            printf "  --local         Local environment\n"
            printf "  --skip-models   Skip model downloads\n"
            printf "  --models-only   Only download models\n"
            printf "  --nodes-only    Only install nodes\n"
            printf "  --tailscale     Setup Tailscale SSH\n"
            printf "  --token=XXX     HuggingFace token\n"
            printf "  --authkey=XXX   Tailscale auth key\n"
            exit 0
            ;;
    esac
done

# Auto-detect interactive mode
INTERACTIVE=true
[ ! -t 0 ] || [ ! -t 1 ] && INTERACTIVE=false

# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE SETUP
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INTERACTIVE" = true ]; then
    print_banner

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: HuggingFace Token
    # ─────────────────────────────────────────────────────────────────────────────
    printf "${BOLD}STEP 1: HuggingFace Token${NC}\n\n"
    printf "    FLUX models require authentication.\n"
    printf "    Get token: ${CYAN}https://huggingface.co/settings/tokens${NC}\n\n"

    if [ -n "$HF_TOKEN" ]; then
        printf "    ${GREEN}✓${NC} Token found in environment\n\n"
    else
        printf "    Enter HF token (or press Enter to skip): "
        read -r HF_TOKEN
        printf "\n"
    fi

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: What to Install
    # ─────────────────────────────────────────────────────────────────────────────
    printf "${BOLD}STEP 2: What do you want to install?${NC}\n\n"
    printf "    ${BOLD}1)${NC} Full Setup      ${DIM}- ComfyUI + Koshi Nodes + Models${NC}\n"
    printf "    ${BOLD}2)${NC} Models Only     ${DIM}- Just download FLUX models${NC}\n"
    printf "    ${BOLD}3)${NC} Nodes Only      ${DIM}- Just install Koshi Nodes${NC}\n"
    printf "    ${BOLD}4)${NC} ComfyUI Only    ${DIM}- Just install ComfyUI${NC}\n"
    printf "\n    Choice [1-4]: "
    read -r install_choice
    printf "\n"

    case $install_choice in
        1) INSTALL_COMFY=true; INSTALL_NODES=true; INSTALL_MODELS=true ;;
        2) INSTALL_COMFY=false; INSTALL_NODES=false; INSTALL_MODELS=true ;;
        3) INSTALL_COMFY=false; INSTALL_NODES=true; INSTALL_MODELS=false ;;
        4) INSTALL_COMFY=true; INSTALL_NODES=false; INSTALL_MODELS=false ;;
        *) INSTALL_COMFY=true; INSTALL_NODES=true; INSTALL_MODELS=true ;;
    esac

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: Model Selection (if installing models)
    # ─────────────────────────────────────────────────────────────────────────────
    if [ "$INSTALL_MODELS" = true ]; then
        printf "${BOLD}STEP 3: Select Model Preset${NC}\n\n"
        printf "    ${BOLD}1)${NC} Klein     ${DIM}- FLUX.2-klein-4B (~13GB) - Recommended${NC}\n"
        printf "    ${BOLD}2)${NC} Minimal   ${DIM}- Schnell + FP8 T5 (~17GB)${NC}\n"
        printf "    ${BOLD}3)${NC} Full      ${DIM}- Schnell + Dev + FP16 T5 (~46GB)${NC}\n"
        printf "    ${BOLD}4)${NC} FP8       ${DIM}- FP8 optimized (~17GB)${NC}\n"
        printf "    ${BOLD}5)${NC} GGUF      ${DIM}- Q4 quantized (~6GB) - Low VRAM${NC}\n"
        printf "\n    Choice [1-5]: "
        read -r model_choice
        printf "\n"

        case $model_choice in
            1) MODEL_PRESET="klein" ;;
            2) MODEL_PRESET="minimal" ;;
            3) MODEL_PRESET="full" ;;
            4) MODEL_PRESET="fp8" ;;
            5) MODEL_PRESET="gguf" ;;
            *) MODEL_PRESET="klein" ;;
        esac
    fi

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4: Environment
    # ─────────────────────────────────────────────────────────────────────────────
    if [ "$INSTALL_COMFY" = true ] || [ "$INSTALL_NODES" = true ]; then
        if [ -z "$INSTALL_MODE" ]; then
            printf "${BOLD}STEP 4: Environment${NC}\n\n"
            if [ -d "/workspace" ]; then
                printf "    ${GREEN}✓${NC} RunPod detected\n\n"
                INSTALL_MODE="runpod"
            else
                printf "    ${BOLD}1)${NC} RunPod    ${DIM}- Full install to /workspace/ComfyUI${NC}\n"
                printf "    ${BOLD}2)${NC} Local     ${DIM}- Add to existing ComfyUI${NC}\n"
                printf "\n    Choice [1-2]: "
                read -r env_choice
                printf "\n"
                case $env_choice in
                    1) INSTALL_MODE="runpod" ;;
                    2) INSTALL_MODE="local" ;;
                    *) INSTALL_MODE="runpod" ;;
                esac
            fi
        fi
    fi

    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5: Tailscale
    # ─────────────────────────────────────────────────────────────────────────────
    if [ "$INSTALL_MODE" = "runpod" ]; then
        printf "${BOLD}STEP 5: Setup Tailscale SSH?${NC}\n\n"
        printf "    ${BOLD}1)${NC} Yes   ${DIM}- Enable remote SSH via Tailscale${NC}\n"
        printf "    ${BOLD}2)${NC} No    ${DIM}- Skip Tailscale setup${NC}\n"
        printf "\n    Choice [1-2]: "
        read -r ts_choice
        printf "\n"
        [ "$ts_choice" = "1" ] && INSTALL_TAILSCALE=true
    fi

    printf "    ─────────────────────────────────────────\n"
    printf "    ${BOLD}Starting installation...${NC}\n"
    printf "    ─────────────────────────────────────────\n"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SET PATHS
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_MODE" = "runpod" ] || [ -d "/workspace" ]; then
    COMFY_DIR="/workspace/ComfyUI"
    RUNPOD=true
else
    RUNPOD=false
    if [ -n "$COMFY_PATH" ]; then
        COMFY_DIR="$COMFY_PATH"
    elif [ -d "$HOME/ComfyUI" ]; then
        COMFY_DIR="$HOME/ComfyUI"
    elif [ -d "/opt/ComfyUI" ]; then
        COMFY_DIR="/opt/ComfyUI"
    else
        COMFY_DIR="$HOME/ComfyUI"
    fi
fi

# Track what was installed for summary
INSTALLED_COMFY=false
INSTALLED_NODES=false
INSTALLED_MODELS=false
INSTALLED_TS=false
MODEL_NAME=""

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL COMFYUI
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_COMFY" = true ]; then
    print_step "1" "Installing ComfyUI"

    mkdir -p "$(dirname "$COMFY_DIR")"
    if [ ! -d "$COMFY_DIR" ]; then
        git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" 2>/dev/null
        print_ok "Cloned ComfyUI"
    else
        print_skip "ComfyUI already exists"
    fi

    cd "$COMFY_DIR"
    if [ "$RUNPOD" = true ]; then
        pip3 install --break-system-packages -q -r requirements.txt 2>/dev/null || pip3 install -q -r requirements.txt
    else
        pip3 install -q -r requirements.txt
    fi
    print_ok "Dependencies installed"
    INSTALLED_COMFY=true
fi

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL KOSHI NODES
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_NODES" = true ]; then
    print_step "2" "Installing Koshi Nodes"

    mkdir -p "$COMFY_DIR/custom_nodes"
    cd "$COMFY_DIR/custom_nodes"

    if [ ! -d "Koshi-Nodes" ]; then
        git clone https://github.com/koshimazaki/ComfyUI-Koshi-Nodes.git Koshi-Nodes 2>/dev/null
        print_ok "Cloned Koshi-Nodes"
    else
        cd Koshi-Nodes && git pull 2>/dev/null && cd ..
        print_skip "Koshi-Nodes updated"
    fi

    # Install dependencies
    if [ "$RUNPOD" = true ]; then
        pip3 install --break-system-packages -q torch numpy Pillow scipy opencv-python moderngl 2>/dev/null || true
    else
        pip3 install -q torch numpy Pillow scipy opencv-python moderngl 2>/dev/null || true
    fi

    # Install extra nodes
    [ ! -d "ComfyUI-Manager" ] && git clone -q https://github.com/ltdrdata/ComfyUI-Manager.git 2>/dev/null && print_ok "ComfyUI-Manager"
    [ ! -d "ComfyUI-VideoHelperSuite" ] && git clone -q https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 2>/dev/null && print_ok "VideoHelperSuite"

    INSTALLED_NODES=true
fi

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_MODELS" = true ] && [ -n "$MODEL_PRESET" ]; then
    print_step "3" "Downloading Models ($MODEL_PRESET)"

    cd "$COMFY_DIR/models"
    mkdir -p checkpoints vae clip unet

    download_model() {
        local file=$1 url=$2 desc=$3 auth=${4:-false}
        if [ ! -f "$file" ]; then
            printf "    ${YELLOW}↓${NC} $desc\n"
            if [ "$auth" = true ] && [ -n "$HF_TOKEN" ]; then
                curl -L --progress-bar -H "Authorization: Bearer $HF_TOKEN" -o "$file" "$url"
            else
                curl -L --progress-bar -o "$file" "$url"
            fi
            if [ -f "$file" ] && head -c 50 "$file" 2>/dev/null | grep -q "Access to model"; then
                print_err "Auth required: $desc"
                rm -f "$file"
                return 1
            fi
        else
            print_skip "$desc (exists)"
        fi
    }

    HF_KLEIN="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main"
    HF_FLUX="https://huggingface.co/black-forest-labs"
    HF_CLIP="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main"
    HF_FP8="https://huggingface.co/Kijai/flux-fp8/resolve/main"
    HF_GGUF="https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main"
    HF_VAE="https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main"

    case $MODEL_PRESET in
        klein)
            MODEL_NAME="FLUX.2-klein-4B"
            download_model "checkpoints/flux2-klein-4b.safetensors" "$HF_KLEIN/flux2-klein-4b.safetensors" "Klein 4B (8GB)" true
            download_model "vae/klein_vae.safetensors" "$HF_KLEIN/vae/diffusion_pytorch_model.safetensors" "Klein VAE" true
            download_model "clip/t5xxl_fp8_e4m3fn.safetensors" "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8"
            download_model "clip/clip_l.safetensors" "$HF_CLIP/clip_l.safetensors" "CLIP-L"
            ;;
        minimal)
            MODEL_NAME="FLUX.1-schnell"
            download_model "vae/ae.safetensors" "$HF_VAE/ae.safetensors" "FLUX VAE"
            download_model "checkpoints/flux1-schnell.safetensors" "$HF_FLUX/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" "Schnell (12GB)" true
            download_model "clip/t5xxl_fp8_e4m3fn.safetensors" "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8"
            download_model "clip/clip_l.safetensors" "$HF_CLIP/clip_l.safetensors" "CLIP-L"
            ;;
        full)
            MODEL_NAME="FLUX.1-schnell + dev"
            download_model "vae/ae.safetensors" "$HF_VAE/ae.safetensors" "FLUX VAE"
            download_model "checkpoints/flux1-schnell.safetensors" "$HF_FLUX/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors" "Schnell (12GB)" true
            download_model "checkpoints/flux1-dev.safetensors" "$HF_FLUX/FLUX.1-dev/resolve/main/flux1-dev.safetensors" "Dev (24GB)" true
            download_model "clip/t5xxl_fp16.safetensors" "$HF_CLIP/t5xxl_fp16.safetensors" "T5-XXL FP16"
            download_model "clip/clip_l.safetensors" "$HF_CLIP/clip_l.safetensors" "CLIP-L"
            ;;
        fp8)
            MODEL_NAME="FLUX.1-dev FP8"
            download_model "vae/ae.safetensors" "$HF_VAE/ae.safetensors" "FLUX VAE"
            download_model "unet/flux1-dev-fp8.safetensors" "$HF_FP8/flux1-dev-fp8.safetensors" "Dev FP8 (12GB)"
            download_model "clip/t5xxl_fp8_e4m3fn.safetensors" "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8"
            download_model "clip/clip_l.safetensors" "$HF_CLIP/clip_l.safetensors" "CLIP-L"
            ;;
        gguf)
            MODEL_NAME="FLUX.1-dev GGUF Q4"
            download_model "vae/ae.safetensors" "$HF_VAE/ae.safetensors" "FLUX VAE"
            download_model "unet/flux1-dev-Q4_K_S.gguf" "$HF_GGUF/flux1-dev-Q4_K_S.gguf" "Dev Q4 GGUF (5.6GB)"
            download_model "clip/t5xxl_fp8_e4m3fn.safetensors" "$HF_CLIP/t5xxl_fp8_e4m3fn.safetensors" "T5-XXL FP8"
            download_model "clip/clip_l.safetensors" "$HF_CLIP/clip_l.safetensors" "CLIP-L"
            # Install GGUF support
            cd "$COMFY_DIR/custom_nodes"
            [ ! -d "ComfyUI-GGUF" ] && git clone -q https://github.com/city96/ComfyUI-GGUF.git 2>/dev/null
            pip3 install -q gguf 2>/dev/null || true
            ;;
    esac
    INSTALLED_MODELS=true
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP TAILSCALE
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_TAILSCALE" = true ]; then
    print_step "4" "Setting up Tailscale"

    if ! command -v tailscale >/dev/null 2>&1; then
        curl -fsSL https://tailscale.com/install.sh | sh 2>/dev/null
        print_ok "Tailscale installed"
    fi

    if ! pgrep -x tailscaled >/dev/null; then
        tailscaled --tun=userspace-networking --state=/workspace/tailscale.state &
        sleep 2
        print_ok "Daemon started"
    fi

    if [ -n "$TS_AUTHKEY" ]; then
        tailscale up --ssh --authkey="$TS_AUTHKEY"
        print_ok "Authenticated"
    else
        print_warn "Run: tailscale up --ssh"
    fi

    INSTALLED_TS=true
fi

# ═══════════════════════════════════════════════════════════════════════════════
# CREATE LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_COMFY" = true ]; then
    cat > "$COMFY_DIR/run.sh" << 'RUNEOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 main.py --listen 0.0.0.0 --port ${PORT:-8188} "$@"
RUNEOF
    chmod +x "$COMFY_DIR/run.sh"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print_done

printf "    ─────────────────────────────────────────\n\n"

[ "$INSTALLED_COMFY" = true ] && printf "    ${GREEN}✓${NC} ComfyUI installed\n"
[ "$INSTALLED_NODES" = true ] && printf "    ${GREEN}✓${NC} Koshi Nodes installed\n"
[ "$INSTALLED_MODELS" = true ] && printf "    ${GREEN}✓${NC} Models: $MODEL_NAME\n"
[ "$INSTALLED_TS" = true ] && printf "    ${GREEN}✓${NC} Tailscale ready\n"

printf "\n    ${BOLD}LOCATIONS${NC}\n"
printf "    ComfyUI: $COMFY_DIR\n"
[ "$INSTALLED_NODES" = true ] && printf "    Nodes:   $COMFY_DIR/custom_nodes/Koshi-Nodes\n"
[ "$INSTALLED_MODELS" = true ] && printf "    Models:  $COMFY_DIR/models/\n"

if [ "$INSTALL_COMFY" = true ]; then
    printf "\n    ${BOLD}START COMFYUI${NC}\n"
    printf "    $COMFY_DIR/run.sh\n"
    printf "\n    ${BOLD}ACCESS${NC}\n"
    printf "    http://localhost:8188\n"
    if [ "$INSTALLED_TS" = true ]; then
        TS_IP=$(tailscale ip -4 2>/dev/null || echo "<tailscale-ip>")
        printf "    http://$TS_IP:8188\n"
    fi
fi

printf "\n    ─────────────────────────────────────────\n\n"

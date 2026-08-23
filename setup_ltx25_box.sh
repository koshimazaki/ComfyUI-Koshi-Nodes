#!/bin/bash
# ComfyUI + LTX-2.5 + AKUSPACE Setup Script  (Vast.ai / RunPod session box)
#
# Interactive: ./setup_ltx25_box.sh
# Non-interactive: ./setup_ltx25_box.sh --vast --token=hf_xxx --lora=/workspace/akuspace-ltx25-v0.5.safetensors
#
# Provisions the box the AKUSPACE A/B actually runs on. The paths below are the
# ones run_batch.sh and run_ab.sh already default to (COMFY=/workspace/ComfyUI-ltx,
# GRAPHS=/workspace/graphs, PORT=8189) -- change them here and you change them
# there too, so don't drift them apart.
#
# Two things this script exists to get right, because both fail SILENTLY:
#
#   1. AKUSPACEReferenceAudioAligned must actually register. It is imported in
#      nodes/audio/__init__.py behind try/except ImportError -> logger.debug, so
#      a half-cloned or stale pack loads perfectly and the node is just absent.
#      The aligned arm then dies on "unknown node type" with nothing to read.
#      --verify greps the running instance's object_info and fails loudly.
#
#   2. LTX-2.5 is a GATED repo. Accept the licence at
#      https://huggingface.co/Lightricks/LTX-2.5 with the same account the token
#      belongs to, or the downloads return a 401 HTML error page that lands on
#      disk as a .safetensors and only explodes at load time.
#
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
    printf "    █▄▀ █▀█ █▀▀ █ █ █    █   ▀█▀ ▀█▀ █ █▀█ █▀▀\n"
    printf "    █ █ █▄█ ▄▄█ █▀█ █    █▄▄  █   █  ▄▀▄ ▄▄█ ▄▄█\n"
    printf "${NC}\n"
    printf "    ${DIM}ComfyUI + LTX-2.5 + AKUSPACE${NC}\n"
    printf "    ${DIM}github.com/koshimazaki/ComfyUI-Koshi-Nodes${NC}\n"
    printf "\n"
    printf "    ─────────────────────────────────────────\n\n"
}

print_done() {
    printf "\n"
    printf "${CYAN}"
    printf "    █▀▄ █▀█ █▄ █ █▀▀ █\n"
    printf "    █▄▀ █▄█ █ ▀█ ██▄ ▄\n"
    printf "${NC}\n"
}

print_step() { printf "\n${BOLD}[$1]${NC} $2\n"; }
print_ok()   { printf "    ${GREEN}✓${NC} $1\n"; }
print_skip() { printf "    ${DIM}○ $1${NC}\n"; }
print_warn() { printf "    ${YELLOW}!${NC} $1\n"; }
print_err()  { printf "    ${RED}✗${NC} $1\n"; }

die() { print_err "$1"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════════════
# PARSE ARGUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
INSTALL_MODE="auto"
INSTALL_COMFY=true
INSTALL_NODES=true
INSTALL_MODELS=true
VERIFY_ONLY=false
HF_TOKEN=""
LORA_SRC=""
WORKSPACE=""

for arg in "$@"; do
    case $arg in
        --vast)         INSTALL_MODE="vast" ;;
        --runpod)       INSTALL_MODE="runpod" ;;
        --token=*)      HF_TOKEN="${arg#*=}" ;;
        --lora=*)       LORA_SRC="${arg#*=}" ;;
        --workspace=*)  WORKSPACE="${arg#*=}" ;;
        --skip-models)  INSTALL_MODELS=false ;;
        --nodes-only)   INSTALL_COMFY=false; INSTALL_MODELS=false ;;
        --verify)       VERIFY_ONLY=true ;;
        --help|-h)
            printf "\n${BOLD}setup_ltx25_box.sh${NC} — provision a session box for the AKUSPACE A/B\n\n"
            printf "  --vast          Vast.ai environment\n"
            printf "  --runpod        RunPod environment\n"
            printf "  --token=XXX     HuggingFace token (LTX-2.5 is gated)\n"
            printf "  --lora=PATH     AKUSPACE LoRA: local file, or hf:REPO/FILE\n"
            printf "  --workspace=DIR Override /workspace\n"
            printf "  --skip-models   Nodes + ComfyUI only\n"
            printf "  --nodes-only    Custom nodes only\n"
            printf "  --verify        Check a RUNNING instance, install nothing\n\n"
            printf "  Get token: ${CYAN}https://huggingface.co/settings/tokens${NC}\n"
            printf "  Accept licence: ${CYAN}https://huggingface.co/Lightricks/LTX-2.5${NC}\n\n"
            exit 0
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════════
# SET PATHS  — these mirror run_batch.sh / run_ab.sh defaults
# ═══════════════════════════════════════════════════════════════════════════════
if [ -z "$WORKSPACE" ]; then
    if [ -d /workspace ]; then WORKSPACE="/workspace"; else WORKSPACE="$HOME/workspace"; fi
fi

COMFY_DIR="$WORKSPACE/ComfyUI-ltx"       # COMFY= in run_batch.sh
GRAPHS_DIR="$WORKSPACE/graphs"           # GRAPHS=
REFS_DIR="$WORKSPACE/refs"               # REFS=
KIT_DIR="$WORKSPACE/kit"                 # KIT=
TRUTH_DIR="$WORKSPACE/truth"             # TRUTH=
PORT="${PORT:-8189}"                     # PORT=
PY="$COMFY_DIR/.venv/bin/python"

LTX_REPO="Lightricks/LTX-2.5"
LTX_BASE="https://huggingface.co/$LTX_REPO/resolve/main"

# path-in-repo  ->  models/ subdir
LTX_FILES=(
    "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|diffusion_models"
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|text_encoders"
    "vae/ltx-2.5-video-vae-bf16.safetensors|vae"
    "vae/ltx-2.5-audio-vae-bf16.safetensors|vae"
)

# Every pack the AKUSPACE graphs resolve against, read off their class_types.
NODE_REPOS=(
    "https://github.com/koshimazaki/ComfyUI-Koshi-Nodes.git|Koshi-Nodes"
    "https://github.com/Lightricks/ComfyUI-LTXVideo.git|ComfyUI-LTXVideo"
    "https://github.com/kijai/ComfyUI-KJNodes.git|ComfyUI-KJNodes"
    "https://github.com/evanspearman/ComfyMath.git|ComfyMath"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git|ComfyUI-VideoHelperSuite"
)

# The two nodes the aligned arm cannot run without.
REQUIRED_NODES=("AKUSPACEReferenceAudioAligned" "Koshi_AKUSPACEPrompt")

print_banner
printf "    workspace  ${BOLD}%s${NC}\n" "$WORKSPACE"
printf "    comfy      ${BOLD}%s${NC}\n" "$COMFY_DIR"
printf "    port       ${BOLD}%s${NC}\n\n" "$PORT"

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY  — the whole point: catch the silent-absence failure
# ═══════════════════════════════════════════════════════════════════════════════
verify_nodes() {
    print_step "VERIFY" "Checking the running instance on :$PORT"

    local info
    info=$(curl -s --max-time 15 "http://127.0.0.1:$PORT/object_info" 2>/dev/null)
    if [ -z "$info" ]; then
        print_err "No response from :$PORT — is ComfyUI running?"
        printf "    ${DIM}start it: %s/run.sh${NC}\n" "$COMFY_DIR"
        return 1
    fi

    local missing=0
    for n in "${REQUIRED_NODES[@]}"; do
        if printf '%s' "$info" | grep -q "\"$n\""; then
            print_ok "$n registered"
        else
            print_err "$n MISSING"
            missing=$((missing + 1))
        fi
    done

    if [ "$missing" -gt 0 ]; then
        printf "\n"
        print_err "The pack loaded but $missing node(s) did not register."
        printf "    ${DIM}This is the silent failure: __init__.py swallows the ImportError${NC}\n"
        printf "    ${DIM}to logger.debug. Check the real reason with:${NC}\n\n"
        printf "      %s -c 'import nodes.audio.aligned_ref' \n" "$PY"
        printf "      cd %s/custom_nodes/Koshi-Nodes && git log --oneline -1\n\n" "$COMFY_DIR"
        printf "    ${DIM}A clone predating commit befde3f has no aligned_ref.py at all.${NC}\n\n"
        return 1
    fi

    # Models are checked by size: a gated-repo 401 lands as a small HTML page
    # with a .safetensors name and only explodes at load time.
    local badfiles=0
    for entry in "${LTX_FILES[@]}"; do
        local rel="${entry%%|*}" sub="${entry##*|}"
        local f="$COMFY_DIR/models/$sub/$(basename "$rel")"
        if [ ! -f "$f" ]; then
            print_err "missing: $sub/$(basename "$rel")"; badfiles=$((badfiles + 1))
        elif [ "$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null)" -lt 1048576 ]; then
            print_err "truncated (<1MB, likely a 401 page): $(basename "$rel")"; badfiles=$((badfiles + 1))
        else
            print_ok "$(basename "$rel")"
        fi
    done
    [ "$badfiles" -gt 0 ] && return 1

    printf "\n"
    print_ok "Box is ready for run_ab.sh"
    return 0
}

if [ "$VERIFY_ONLY" = true ]; then
    verify_nodes; exit $?
fi

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL COMFYUI
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_COMFY" = true ]; then
    print_step "1/4" "ComfyUI"
    mkdir -p "$WORKSPACE" "$GRAPHS_DIR" "$REFS_DIR" "$KIT_DIR" "$TRUTH_DIR"

    if [ ! -d "$COMFY_DIR" ]; then
        git clone -q https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR" \
            || die "clone failed"
        print_ok "cloned to $COMFY_DIR"
    else
        print_skip "already at $COMFY_DIR"
    fi

    cd "$COMFY_DIR" || die "cannot enter $COMFY_DIR"
    if [ ! -d .venv ]; then
        python3 -m venv .venv || die "venv failed"
        print_ok "venv created"
    else
        print_skip "venv exists"
    fi

    "$PY" -m pip install -q --upgrade pip
    "$PY" -m pip install -q -r requirements.txt || print_warn "some requirements failed"
    print_ok "requirements installed"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALL CUSTOM NODES
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$INSTALL_NODES" = true ]; then
    print_step "2/4" "Custom nodes"
    mkdir -p "$COMFY_DIR/custom_nodes"
    cd "$COMFY_DIR/custom_nodes" || die "no custom_nodes dir"

    for entry in "${NODE_REPOS[@]}"; do
        url="${entry%%|*}"; dir="${entry##*|}"
        if [ -d "$dir" ]; then
            git -C "$dir" pull -q 2>/dev/null && print_ok "$dir (updated)" || print_skip "$dir (exists)"
        else
            git clone -q "$url" "$dir" 2>/dev/null && print_ok "$dir" || print_err "$dir failed"
        fi
        [ -f "$dir/requirements.txt" ] && "$PY" -m pip install -q -r "$dir/requirements.txt" 2>/dev/null
    done

    # Fail early rather than at render time: the aligned node must be present in
    # the clone. Anything before befde3f simply does not contain the file.
    if [ ! -f "Koshi-Nodes/nodes/audio/aligned_ref.py" ]; then
        print_err "Koshi-Nodes has no nodes/audio/aligned_ref.py"
        printf "    ${DIM}The clone predates commit befde3f. The aligned arm will not run.${NC}\n"
        printf "    ${DIM}Fix: git -C Koshi-Nodes pull origin main${NC}\n"
    else
        print_ok "aligned_ref.py present"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════
fetch() {  # fetch <url> <dest>
    local url="$1" dest="$2"
    if [ -f "$dest" ] && [ "$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest" 2>/dev/null)" -gt 1048576 ]; then
        print_skip "$(basename "$dest") (have it)"; return 0
    fi
    mkdir -p "$(dirname "$dest")"
    if [ -n "$HF_TOKEN" ]; then
        curl -fL --progress-bar -H "Authorization: Bearer $HF_TOKEN" -o "$dest" "$url"
    else
        curl -fL --progress-bar -o "$dest" "$url"
    fi
    if [ $? -ne 0 ]; then
        rm -f "$dest"
        print_err "$(basename "$dest") failed — gated repo needs --token and an accepted licence"
        return 1
    fi
    print_ok "$(basename "$dest")"
}

if [ "$INSTALL_MODELS" = true ]; then
    print_step "3/4" "LTX-2.5 models (gated — $LTX_REPO)"
    if [ -z "$HF_TOKEN" ]; then
        print_warn "no --token given; gated downloads will 401"
        printf "    ${DIM}Accept the licence first: https://huggingface.co/%s${NC}\n" "$LTX_REPO"
    fi

    for entry in "${LTX_FILES[@]}"; do
        rel="${entry%%|*}"; sub="${entry##*|}"
        fetch "$LTX_BASE/$rel" "$COMFY_DIR/models/$sub/$(basename "$rel")"
    done

    # AKUSPACE LoRA. The HF repo is private (401 without a token), so a local
    # file copied up with scp/rsync is the usual path on a rented box.
    print_step "4/4" "AKUSPACE LoRA"
    LORA_DEST="$COMFY_DIR/models/loras/akuspace-ltx25-v0.5.safetensors"
    mkdir -p "$(dirname "$LORA_DEST")"
    if [ -z "$LORA_SRC" ]; then
        print_warn "no --lora given — the aligned and stock arms need it"
        printf "    ${DIM}scp it up:  scp akuspace-ltx25-v0.5.safetensors box:%s${NC}\n" "$LORA_DEST"
        printf "    ${DIM}or:         --lora=hf:KoshiMazaki/akuspace-ltx25/akuspace-ltx25-v0.5.safetensors${NC}\n"
    elif [ -f "$LORA_SRC" ]; then
        cp "$LORA_SRC" "$LORA_DEST" && print_ok "copied from $LORA_SRC"
    elif [[ "$LORA_SRC" == hf:* ]]; then
        spec="${LORA_SRC#hf:}"
        repo="$(printf '%s' "$spec" | cut -d/ -f1-2)"
        file="$(printf '%s' "$spec" | cut -d/ -f3-)"
        fetch "https://huggingface.co/$repo/resolve/main/$file" "$LORA_DEST"
    else
        print_err "--lora=$LORA_SRC is neither a file nor hf:REPO/FILE"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
cat > "$COMFY_DIR/run.sh" << RUNEOF
#!/bin/bash
# Port $PORT is what run_batch.sh and run_ab.sh talk to.
cd "$COMFY_DIR"
exec .venv/bin/python main.py --listen 0.0.0.0 --port $PORT "\$@"
RUNEOF
chmod +x "$COMFY_DIR/run.sh"

print_done
printf "    ComfyUI: $COMFY_DIR\n"
printf "    Graphs:  $GRAPHS_DIR ${DIM}(copy workflows/api/*.json here)${NC}\n"
printf "    Models:  $COMFY_DIR/models/\n\n"
printf "    ${BOLD}Start it${NC}\n"
printf "      $COMFY_DIR/run.sh\n\n"
printf "    ${BOLD}Then confirm the aligned node actually registered${NC}\n"
printf "      ./setup_ltx25_box.sh --verify\n\n"
printf "    ${DIM}Skipping --verify is how a broken aligned arm reaches a paid GPU hour.${NC}\n\n"

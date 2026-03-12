#!/bin/bash
# ============================================================
# VaceSamhera Provisioning Script
# github.com/HeraKang000/VaceSamhera
#
# Pipeline: Person image → SAM3 mask → VACE MV2V/R2V → edited video
# Modes: R2V (ref image → new video) | MV2V (masked region swap)
# ============================================================

set -euo pipefail

COMFY_ROOT="/workspace/ComfyUI"
CUSTOM_NODES="$COMFY_ROOT/custom_nodes"
MODELS="$COMFY_ROOT/models"
LOG="/workspace/provisioning.log"

exec > >(tee -a "$LOG") 2>&1
echo ""
echo "======================================================"
echo " VaceSamhera Provisioning — $(date)"
echo "======================================================"

# ── helpers ──────────────────────────────────────────────────
green()  { echo -e "\033[32m[OK]\033[0m  $*"; }
yellow() { echo -e "\033[33m[--]\033[0m  $*"; }
red()    { echo -e "\033[31m[ERR]\033[0m $*"; }

pip_quiet() { pip install -q --no-warn-script-location "$@"; }

clone_or_update() {
    local NAME=$1 URL=$2
    local DIR="$CUSTOM_NODES/$NAME"
    if [ ! -d "$DIR/.git" ]; then
        echo "  Cloning $NAME..."
        git clone --depth 1 "$URL" "$DIR"
        [ -f "$DIR/requirements.txt" ] && pip_quiet -r "$DIR/requirements.txt"
        green "$NAME installed"
    else
        yellow "$NAME exists — pulling"
        git -C "$DIR" pull --ff-only 2>/dev/null || true
    fi
}

dl_hf() {
    # dl_hf <repo_id> <filename> <local_dir>
    local REPO=$1 FILE=$2 DIR=$3
    local TARGET="$DIR/$(basename "$FILE")"
    if [ -f "$TARGET" ]; then
        yellow "EXISTS  $(basename "$FILE")"
        return
    fi
    echo "  Downloading $(basename "$FILE") from $REPO ..."
    python3 - <<PYEOF
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="$REPO", filename="$FILE", local_dir="$DIR")
PYEOF
    green "$(basename "$FILE")"
}

# ── 1. System packages ────────────────────────────────────────
echo ""
echo "── 1. System packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    git ffmpeg libgl1 libglib2.0-0 wget curl > /dev/null
green "System packages ready"

# ── 2. Python packages ────────────────────────────────────────
echo ""
echo "── 2. Python packages"
export HF_HUB_ENABLE_HF_TRANSFER=1
pip_quiet huggingface_hub hf_transfer
pip_quiet decord opencv-python-headless imageio[ffmpeg]
pip_quiet einops omegaconf timm
green "Python packages ready"

# ── 3. ComfyUI ───────────────────────────────────────────────
echo ""
echo "── 3. ComfyUI"
if [ ! -d "$COMFY_ROOT/.git" ]; then
    echo "  Cloning ComfyUI..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI "$COMFY_ROOT"
    pip_quiet -r "$COMFY_ROOT/requirements.txt"
    green "ComfyUI installed"
else
    yellow "ComfyUI exists — pulling"
    git -C "$COMFY_ROOT" pull --ff-only 2>/dev/null || true
fi

mkdir -p \
    "$MODELS/diffusion_models" \
    "$MODELS/vae" \
    "$MODELS/text_encoders" \
    "$MODELS/clip_vision" \
    "$MODELS/sam3" \
    "$MODELS/controlnet" \
    "$MODELS/upscale_models"

# ── 4. Custom nodes ──────────────────────────────────────────
echo ""
echo "── 4. Custom nodes"

# Core video/VACE
clone_or_update "ComfyUI-VideoHelperSuite" \
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"

clone_or_update "ComfyUI-WanVideoWrapper" \
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"

# ControlNet (DWPose + DepthAnything for pose/depth preprocessing)
clone_or_update "comfyui_controlnet_aux" \
    "https://github.com/Fannovel16/comfyui_controlnet_aux"

# SAMhera — custom nodes (VLM + SAM3 segmentation)
clone_or_update "SAMhera" \
    "https://github.com/HeraKang000/SAMhera"

green "All custom nodes ready"

# ── 5. Models ────────────────────────────────────────────────
echo ""
echo "── 5. Models"

# ── VACE 14B fp8_scaled ──────────────────────────────────────
# Source: Kijai/WanVideo_comfy_fp8_scaled
# Structure: T2V base model + VACE module (both required, loaded separately in ComfyUI)
# Total: ~15GB + ~3GB = ~18GB  (vs 63GB bf16 sharded — same visual quality)

# T2V base (fp8_scaled) — ~15GB single file
echo "  [VACE 14B base — fp8_scaled ~15GB]"
dl_hf "Kijai/WanVideo_comfy_fp8_scaled" \
    "T2V/Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
    "$MODELS/diffusion_models"

# VACE module (fp8_scaled) — ~3GB single file
echo "  [VACE module — fp8_scaled ~3GB]"
dl_hf "Kijai/WanVideo_comfy_fp8_scaled" \
    "VACE/Wan2_1-VACE-module-14B_fp8_e4m3fn_scaled_KJ.safetensors" \
    "$MODELS/diffusion_models"

# VAE — from official Wan2.1-VACE-14B repo (508MB)
echo "  [Wan VAE — 508MB]"
dl_hf "Wan-AI/Wan2.1-VACE-14B" \
    "Wan2.1_VAE.pth" \
    "$MODELS/vae"

# Text encoder — fp8 safetensors (NOT the .pth pickle in VACE repo)
echo "  [UMT5 text encoder — fp8 safetensors]"
dl_hf "Comfy-Org/mochi_preview_repackaged" \
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    "$MODELS/text_encoders"

# CLIP vision (for image conditioning in R2V)
echo "  [CLIP vision]"
dl_hf "Comfy-Org/sigclip_vision_384" \
    "sigclip_vision_patch14_384.safetensors" \
    "$MODELS/clip_vision"

# SAM3 checkpoint
echo "  [SAM3 checkpoint]"
dl_hf "1038lab/sam3" \
    "sam3.pt" \
    "$MODELS/sam3"

green "All models downloaded"

# ── 6. extra_model_paths.yaml ────────────────────────────────
echo ""
echo "── 6. Patching extra_model_paths.yaml"
cat > "$COMFY_ROOT/extra_model_paths.yaml" << EOF
# VaceSamhera — auto-generated by provisioning.sh
vacesamhera:
    base_path: ${MODELS}
    checkpoints: diffusion_models
    diffusion_models: diffusion_models
    vae: vae
    text_encoders: text_encoders
    clip_vision: clip_vision
    controlnet: controlnet
    upscale_models: upscale_models
    sam3: sam3
EOF
green "extra_model_paths.yaml written"

# ── 7. Launch ComfyUI ────────────────────────────────────────
echo ""
echo "── 7. Launching ComfyUI"
pkill -f "python.*main.py" 2>/dev/null || true
sleep 1

nohup python "$COMFY_ROOT/main.py" \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    >> /workspace/comfyui.log 2>&1 &

echo "  PID: $!"
echo "  Log: /workspace/comfyui.log"
echo ""
echo "======================================================"
green "Provisioning complete — $(date)"
echo "======================================================"
echo ""
echo "  Models ready for:"
echo "    R2V  — person photo -> animated video"
echo "    MV2V — SAM3 mask + ref image -> face/clothing swap"
echo "    V2V  — DWPose + DepthAnything -> motion transfer"
echo ""

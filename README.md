# VaceSamhera

vAST AI auto-setup for **VACE + SAM3** ComfyUI pipeline.

**Goal:** Person photo → SAM3 segmentation mask → VACE video editing (face, clothing, expression swap)

---

## Pipeline Overview

```
Person photo (1 image)
        │
        ├─► [SAM3] ──── segmentation masks (face / clothing / expression regions)
        │                        │
        │                        ▼
        └─► [VACE MV2V] ── masked region swap into target video
                │
                └─► [VACE R2V] ── generate new video preserving identity
```

### VACE Modes Used

| Mode | Input | Use case |
|------|-------|----------|
| **R2V** | ref image → video | Animate person photo, preserve face/clothing identity |
| **MV2V** | mask + ref → video | Replace specific region (face, outfit) in existing video |
| **V2V** | pose/depth + ref → video | Motion transfer with DWPose + DepthAnything |

---

## vAST AI Setup

### On-start Script Field
Paste this single line:
```bash
curl -fsSL https://raw.githubusercontent.com/HeraKang000/VaceSamhera/main/entrypoint.sh | bash
```

### What Gets Installed Automatically

**Custom nodes:**
- ComfyUI-VideoHelperSuite
- ComfyUI-WanVideoWrapper
- comfyui_controlnet_aux (DWPose, DepthAnything)
- ComfyUI-BRIA-AI-RMBG (background removal for R2V)
- ComfyUI-SAM3 (this repo)

**Models (from HuggingFace):**
- `wan2.1_vace_14B` — VACE diffusion model
- `wan_2.1_vae` — VAE
- `umt5_xxl_fp8` — text encoder
- `sigclip_vision_patch14_384` — CLIP vision (image conditioning)
- `sam3.pt` — SAM3 segmentation checkpoint

---

## Repo Structure

```
VaceSamhera/
├── entrypoint.sh          ← vAST "On-start script" field (1 line)
├── provisioning.sh        ← full install + model download logic
├── workflows/             ← ComfyUI workflow JSONs
│   ├── r2v_person.json    ← R2V: photo → video
│   ├── mv2v_sam_swap.json ← MV2V: SAM mask + ref → swap
│   └── v2v_pose.json      ← V2V: pose transfer
└── README.md
```

---

## VRAM Requirements

| Mode | Min VRAM | Recommended |
|------|----------|-------------|
| VACE 1.3B | 12 GB | 16 GB |
| VACE 14B (R2V/MV2V) | 20 GB | 40 GB |
| + SAM3 | +4 GB | — |

For VRAM < 20GB: edit `provisioning.sh` line and replace `Wan2.1-VACE-14B` with `Wan2.1-VACE-1.3B`.

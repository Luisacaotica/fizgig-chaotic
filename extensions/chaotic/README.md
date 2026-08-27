# Fizgig Chaotic Fork — Luisa Caotica Edition

Extension that makes Fizgig behave like AI-Toolkit where it matters, without forking core files (so `update_fizgig.bat` / `git pull` never overwrites it).

## What it does

- **Steps instead of Epochs** (AI-Toolkit parity): Training tab shows `Max Steps` toggle. When ON, you type `1000` steps → Chaotic converts to epochs (`steps // steps_per_epoch`) before launching trainer. Works for Klein / Krea2 / MiniMax H3. 4060 Ti 8GB safe — respects batch_size=1, repeats, bucket count.
- **Control + Target (paired training)**: Exposes Fizgig's existing `control_directory` (hidden before). Two folders:
  - `Dataset folder` = TARGET (what model learns to generate)
  - `Control folder` = CONDITION (depth, pose, canny, edit base image — Flux Klein Edit, Krea2, H3 video controls)
  Energies are encoded together via `pack_control_latent` → `ref_tokens` concatenated to `img_input` in `trainer.py:1035`. Compatible with Klein, Krea2 (Qwen3-VL ref image path), and H3 video (clip controls via same pipe). Matches AI-Toolkit's `control_path` / `control_lora`.
- **LoRA Adapter — already exists, now visible**: Fizgig trains LoRA (rank/alpha) + LoKR (Kronecker) on all three families (see `src/fizgig/networks/lora.py:23`). Check `Network Type` → `LoRA (standard)` vs `LoKR`. Chaotic adds:
  - Auto LoRA rank suggestion for 8GB (rank 8-16 for 8GB portraits)
  - `Context LoRA` = AI-Toolkit's `control_lora` — frozen + active base LoRA so new LoRA learns to coexist (Klein & Krea2).
  - H3 `Adapter-relative LR` ramp for stable low-rank fine-tunes.
- **Survives updates**: Lives in `extensions/chaotic/`. Loaded via `launch_chaotic.pyw` / `launch_chaotic.bat` which patches `lora_trainer_gui.py` at runtime (monkey-patch). No core files edited → `git pull` safe.

## Install (fork-safe)

1. This folder is already at `F:\Fizgig\extensions\chaotic`
2. Launch with: `launch_chaotic.bat` (double-click) or `python launch_chaotic.pyw`
3. Don't use `run_fizgig.bat` when you want chaotic features — that launches vanilla Fizgig.

## Git fork (optional, if you want GitHub)

```bash
cd F:\Fizgig
git remote rename origin upstream
git remote add origin https://github.com/luisa-caotica/fizgig-chaotic.git
git push -u origin master
# to sync upstream without losing chaotic:
git fetch upstream
git merge upstream/master
# extensions/chaotic never conflicts
```

## VRAM guide for RTX 4060 Ti 8GB

- Krea2: Auto → INT8, batch 1, 0.25 MP (fits 8GB) — Chaotic defaults to this when family=Krea2.
- Klein 9B: Auto → fp8 + block_swap 12-16, 768px.
- H3: Train on 0.25 MP, 22-frame clips max (see `docs` table). Don't enable full 1 MP video on 8GB.

## AI-Toolkit config translation

AI-Toolkit JSON → Fizgig Chaotic:

- `train.steps = 1000` → toggle Steps ON, enter 1000 (converted to epochs).
- `datasets[0].control_path` → `Control folder` picker (same basenames matched in `dataset/image_dataset.py:584`).
- `network.type = "lora"` / `"lokr"` → `Network Type` dropdown (identical).
- `network.lora_config` / `lokr_factor` → same fields.

Questions: open issue in chaotic fork or ping Luisa Caotica.

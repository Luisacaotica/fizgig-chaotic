# Fizgig Chaotic Fork — Luisa Caotica Edition

Extension that makes Fizgig behave like AI-Toolkit where it matters, without forking core files (so `update_fizgig.bat` / `git pull` never overwrites it).

> **Repo:** `Luisacaotica/fizgig-chaotic` — based on `shootthesound/Fizgig` + Chaotic extension in `extensions/chaotic/`. Launch with `launch_chaotic.bat` (title shows `[CHAOTIC]`).

## What's new (vs vanilla Fizgig)

### 1 — Steps instead of Epochs (AI-Toolkit parity)
- **Where:** Training tab → `Chaotic — Advanced Training` → *Use Steps instead of Epochs* (top of the card)
- **What:** Type `train.steps = 1000` like AI-Toolkit (`toolkit/config_modules.py:380`). Chaotic converts to Fizgig epochs via `epochs = ceil(steps / num_images)` (`extensions/chaotic/patch.py: _epochs_from_steps`) and swaps `MAX_TRAIN_EPOCHS` at `start_training()` time. Live label shows `≈ X epochs @ Y imgs/epoch`.
- **Files:** `src/fizgig/training/train_utils.py`, `lora_trainer_gui.py:4059` (vanilla epoch-based) vs `toolkit/config_modules.py:380` (AI-Toolkit steps)

### 2 — Control + Target (paired training)
- **Where:** Same Chaotic card → *Control + Target* → `Control folder` picker
- **What:** Exposes Fizgig's existing `control_directory` (`src/fizgig/dataset/image_dataset.py:528`, `src/fizgig/dataset/config.py:71` `ImageDatasetParams.control_directory`). Same-basename pairing (`image_dataset.py:584` `glob_images`) → `latent_control_{i}` cached in `cache_latents.py:51` → `ref_tokens` via `pack_control_latent` → DiT `img_input` (`src/fizgig/training/trainer.py:998-1035`). Works for:
  - **Flux2 Klein Edit** (base edit model — control = edit base image)
  - **Krea2** (Qwen3-VL vision path)
  - **MiniMax H3** video (clip controls, `latent_TxHxW`)
- **Pairing check:** live `✓ N/N paired` / `⚠ N/M missing` label. Unpaired = vanilla single-folder training.
- **AI-Toolkit map:** `datasets[0].control_path` / `AdapterConfig.type=control_lora` (`toolkit/config_modules.py:230`) → Chaotic `Control folder`

### 3 — Advanced Training card (optimizer / scheduler / saves / dropout)
New `CollapsibleFrame` **"Chaotic — Advanced Training (AI-Toolkit parity)"** packed after vanilla Training section (`extensions/chaotic/patch.py: _inject_chaotic_ui`). Contains synced duplicates of vanilla hidden fields so you don't dig through collapsed `Optimizer`/`Other Options`:

| Chaotic field | Real entry synced at launch | AI-Toolkit equivalent |
|---|---|---|
| `Optimizer` (adamw, adamw8bit, lion, prodigy, adafactor) | `OPTIMIZER_TYPE` | `train.optimizer` |
| `LR Scheduler` (constant, cosine, linear, polynomial, rex) | `LR_SCHEDULER` | `train.lr_scheduler` |
| `Warmup steps` | `LR_WARMUP_STEPS` | `train.lr_warmup_steps` |
| `Grad Accum` | `GRADIENT_ACCUMULATION` | `train.gradient_accumulation` |
| `Max Grad Norm` | `MAX_GRAD_NORM` | `train.max_grad_norm` |
| `Save Every N epochs` / `Save Every N steps` (toggle) | `SAVE_EVERY_N_EPOCHS` | `save.save_every` |
| `Network Dropout` | `NETWORK_DROPOUT` | `network.dropout` |
| `Caption Dropout` (0.05 default) | new `CHAOTIC_CAPTION_DROPOUT` persisted, maps to dataset `caption_dropout_rate` | `datasets[].caption_dropout_rate` |

Deduped: original `Save Every N Epochs` row greyed/disarmed (`_sync_chaotic_to_real`), Chaotic card is single source. When `Use Steps instead of Epochs` is ON, Chaotic `Save Every` is interpreted as **steps** → `epochs = ceil(steps / spe)` (`spe = num_images/clips` `patch.py:_steps_per_epoch`), e.g. 9 clips, `Save Every 30 steps` → `4 epochs`. When OFF, interpreted as epochs. Label + hint toggle via `_on_steps_toggle`.

### 4 — LoRA adapter (not missing — now visible)
- Fizgig already trains LoRA + LoKR (`src/fizgig/networks/lora.py:23` `LoRAModule`, `LoKRModule` kronecker) → Kohya `.safetensors`, ComfyUI-ready.
- Chaotic surfaces: `Training Parameters → Network Type: LoRA (standard) vs LoKR (Kronecker)` + `LoKR Factor 8` + `Context LoRA` (`CONTEXT_LORA_PATH` at `lora_trainer_gui.py:4183`) = AI-Toolkit `network.type=lokr` / `control_lora`.
- 8GB tip: rank 8-16, 0.25 MP, Auto VRAM for RTX 4060 Ti 8GB.

### 5 — Krea2 Ostris Edit (Sampling tab, English only)
- **Where:** **Samples tab → Prompt & Dimensions card → bottom** — `Chaotic — Krea2 Ostris Edit (test edit in sampling)` (previously in Training, moved per request for live testing). Injected directly into `prompt_card` (`lora_trainer_gui.py:10384`) so it sits *below* Width/Height/Steps/Seed/Reference, *together* with Prompt & Dimensions.
- **What:** 3 reference pickers `Image 1..3` (`CHAOTIC_KREA2_IMG1..3`) + `kv_cache` toggle (`CHAOTIC_KV_CACHE`). Mirrors `comfyui-krea2-ostris-edit/nodes.py:64` `TextEncodeKrea2OstrisEdit`: Qwen3-VL vision `384*384` (`VLM_MAX_PIXELS`) + VAE latents `1MP snap 16` (`REF_LATENT_MAX_PIXELS`) → `index_timestep_zero` (`nodes.py:227` `_forward_with_refs` at `t=0`, split modulation `_block_ref_forward:195`). `kv_cache` = one-pass `K/V` precompute (`nodes.py:307` `_precompute_ref_kv`) reused per denoising step.
- **How to test:** select 1-3 refs, keep VAE connected, prompt with trigger, Generate Sample. Refs are encoded via Qwen (semantic) + VAE (t=0). Training card shows `moved → Samples tab` note.

### 6 — Automagic optimizers (v1/v2/v3)
- **Where:** Training → `Chaotic — Advanced Training` → `Optimizer` dropdown now includes `automagic`, `automagic2`, `automagic3`
- **What:** Ported from `F:\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\optimizers\automagic3.py:1` (`Automagic3: polarity_history` fused adaptive LR) into `extensions/chaotic/optimizers/automagic{,2,3}.py`. Injected into `src/fizgig/training/optimizers.py:39` `_CATALOG` via `_patch_automagic_catalog()` (`extensions/chaotic/patch.py`). Previously removed in `optimizers.py:33` (prodigy/came/adafactor), now available as `automagic3` (recommended v3).

### 7 — Black & Orange theme
- **Where:** `launch_chaotic.*` only (vanilla stays blue). `CHAOTIC_COLORS` (`patch.py:43` `bg_deep #0A0A0A`, `accent #FF6B00`) overrides `lora_trainer_gui.py:55` `COLORS` before widgets, plus `clam` ttk style (`TButton`, `TNotebook.Tab`, `Progressbar`). Title gets `[CHAOTIC]` / `[CHAOTIC - ORANGE]`.

### 8 — Grouped hub (Chaotic tab — fix for collapsed params)
- **Where:** `✦ Chaotic` first tab — `lora_trainer_gui.py:1830` `notebook` hub with `Data • Training • Misc • Modify • Options` jump buttons (`extensions/chaotic/patch.py:_inject_chaotic_full_tab`)
- **What:** Previous LLM reorg hid `1.Start..5.Training` but failed to move `Canvas window` cards (`card.master = outer` illegal + `outer_src` 2-level search missed `canvas.create_window` `lora_trainer_gui.py:3056`), leaving `Chaotic` empty. Fixed: hub keeps all 10 tabs flat and jumps via `notebook.select()` — no `pack(in_=)` reparent. `CHAOTIC_GROUPED=1` env tries experimental `place(in_=sub)` grouping; default `0` is flat+hub (stable).
- **Hub layout:** `Data` → `1.Start/2.Image Prep/3.Captions`, `Training` → `4.Samples/5.Training`, `Misc` → `Profiler/Repair/Explorer/Royale/Extract/Metadata`, `Modify` → `Training → Training Parameters`, `Options` → `Preferences + Training → Memory & Seed`

## How it survives updates
- Lives in `extensions/chaotic/` (`patch.py`, `README.md`, `fork_guide.md`). Loaded via `launch_chaotic.pyw` which imports `lora_trainer_gui`, patches `LoRATrainerGUI` class (`apply_chaotic_patches`), then instantiates it — no core file edited → `git pull` / `update_fizgig.bat` safe.
- Title proves it: window title gets ` [CHAOTIC]` / `CHAOTIC (Luisa Caotica)` (`launch_chaotic.pyw`).

## Install & Launch (fork-safe)

1. Clone Chaotic fork (or copy `extensions/chaotic/` + `launch_chaotic.*` into vanilla Fizgig)
2. **Chaotic:** double-click `launch_chaotic.bat` (silent, via `wscript run_chaotic_silent.vbs`) or `run_chaotic_console.bat` (console logs, recommended to test) — `python launch_chaotic.pyw` also works
3. **Vanilla:** `run_fizgig.bat` / `launch.pyw` still works unchanged
4. Verify: console shows `[chaotic] patched LoRATrainerGUI` + `[chaotic] injecting UI...` and title `CHAOTIC`

## Git fork workflow

```bash
cd F:\Fizgig
git remote rename origin upstream
git remote add origin https://github.com/Luisacaotica/fizgig-chaotic.git
git push -u origin master
# sync upstream without losing Chaotic:
git fetch upstream
git merge upstream/master  # extensions/chaotic/ never conflicts
git push
```

See `extensions/chaotic/fork_guide.md` for details.

## VRAM guide for RTX 4060 Ti 8GB + 32 RAM

- **Krea2:** Auto → INT8, batch 1, 0.25 MP (fits 8GB per `README.md:83`)
- **Klein 9B:** Auto → fp8 + block_swap 12-16, 768px, muestre s 40 pasos
- **MiniMax H3:** 0.25 MP, 22-frame clips max; 56+ frames → 24GB; 107-124 frames → 0.25 MP only. See `docs` VRAM table.

## AI-Toolkit → Chaotic translation cheat-sheet

- `train.steps = 1000` → toggle Steps ON, enter 1000
- `datasets[0].control_path` → `Control folder` picker (same basenames)
- `network.type = "lora"` / `"lokr"` + `lokr_factor` → `Network Type` dropdown
- `train.optimizer` / `lr_scheduler` / `caption_dropout_rate` → Chaotic Advanced card (same keys, synced)

## Files

- `launch_chaotic.pyw:1` — Chaotic launcher (patched, correct `LoRATrainerGUI` instantiation, avoids `runpy` lost-patch bug)
- `launch_chaotic.bat:1` — ASCII, wscript silent
- `run_chaotic_silent.vbs:1` / `run_chaotic_console.bat:1` — launch helpers
- `extensions/chaotic/patch.py:1` — monkey-patches: `__init__`, `start_training`, TOML, Chaotic card (below Presets→above Output), Prompt & Dimensions injection, automagic catalog, orange theme
- `extensions/chaotic/krea2_ostris_edit.py:1` — Ostris port: `VLM 384px + VAE 1MP t=0` + `kv_cache` (`_pack_refs_fizgig`, `apply_krea2_ostris_patch`)
- `extensions/chaotic/optimizers/automagic{,2,3}.py:1` — Automagic v1/v2/v3 ported from `toolkit/optimizers/` (MIT)
- `presets/chaotic.json` — remembers Steps toggle

Questions: open issue in `Luisacaotica/fizgig-chaotic` or ping Luisa Caotica.

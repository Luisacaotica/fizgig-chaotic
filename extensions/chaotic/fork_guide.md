# Chaotic Fork — Git Setup (so updates never overwrite)

## 1) Keep vanilla Fizgig as upstream
```bat
cd /d F:\Fizgig
git remote rename origin upstream
git remote add origin https://github.com/YOURUSER/fizgig-chaotic.git
git push -u origin master
```

## 2) Daily use
- Train with Chaotic: double-click `launch_chaotic.bat`
- Vanilla fallback: `run_fizgig.bat` / `launch.pyw`

## 3) Pull upstream Fizgig updates without losing Chaotic
```bat
git fetch upstream
git merge upstream/master --no-edit
:: extensions/chaotic/* never conflicts with upstream
git push
```

## 4) If Fizgig overwrites lora_trainer_gui.py heavily
Chaotic patch is monkey-patched — no merge needed. Only if new Fizgig renames GUI class, edit `extensions/chaotic/patch.py` target list.

## 5) Share
Push `extensions/chaotic` + `launch_chaotic.*` + this guide to your fork. Users clone chaotic fork directly.

## AI-Toolkit parity notes for docs

- Control = `toolkit/control_adapter` → Fizgig `control_directory` → `latent_control_{i}` → `pack_control_latent` → DiT `ref_tokens` (same as Klein edit).
- Steps = `toolkit/config.py: steps` → Chaotic `CHAOTIC_STEPS` → epochs = ceil(steps / num_images). SaveEverySteps simulated as saveEveryEpochs = epochs // 5.
- LoRA = `toolkit/lora` → Fizgig `networks/lora.py` LoraModule/LoKRModule (identical Kohya .safetensors, ComfyUI-ready).
- Video = toolkit `i2v` adapter → Fizgig H3 video latent (22/56 frames) + audio VAE rows.

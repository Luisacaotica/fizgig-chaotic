"""Chaotic patch — applied at startup by launch_chaotic.pyw.

Adds:
- Steps vs Epochs toggle (AI-Toolkit parity)
- Control + Target dataset UI (exposes control_directory)
- Advanced Training card: optimizer, scheduler, saves, noise, dropout, AI-Toolkit parity
- 8GB VRAM helpers + LoRA/KoKR visibility

All patches are monkey-patches on lora_trainer_gui.LoRATrainerGUI — no file edits, safe across `git pull`.
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog

# Ostris edit port (krea2)
try:
    from extensions.chaotic.krea2_ostris_edit import apply_krea2_ostris_patch
except Exception:
    try:
        from fizgig.extensions.chaotic.krea2_ostris_edit import apply_krea2_ostris_patch  # alt
    except: apply_krea2_ostris_patch = None


COLORS = None  # will import from gui module

def _steps_per_epoch(dataset_folder: str) -> int:
    if not dataset_folder or not os.path.isdir(dataset_folder):
        return 0
    # Count images + videos + audio (H3 clips are .mp4, not images)
    exts = {'.jpg','.jpeg','.png','.webp','.bmp','.tiff','.tif',
            '.mp4','.mov','.avi','.webm','.mkv',
            '.wav','.mp3','.flac','.m4a','.ogg','.opus'}
    n = 0
    try:
        for f in os.listdir(dataset_folder):
            if f.startswith('.'): continue
            if os.path.splitext(f)[1].lower() in exts:
                n += 1
        # Also count datasets/_tf2 style where clips are in subfolder? Already counted
        # Fallback: batch count from cache total_batches if available (more accurate)
        if n == 0:
            # Try to read total batches from dataset config if dataset folder empty (e.g. clips counted as 0)
            n = 1
    except: pass
    return max(1, n)

def _epochs_from_steps(steps: int, steps_per_epoch: int) -> int:
    if steps_per_epoch <= 0:
        return max(1, steps)
    return max(1, (steps + steps_per_epoch - 1) // steps_per_epoch)

CHAOTIC_COLORS = {
    "bg_deep": "#0A0A0A",
    "bg_surface": "#1A1A1A",
    "bg_hover": "#2A1D0F",
    "bg_header": "#111111",
    "text_primary": "#FFF2E6",
    "text_secondary": "#FFB266",
    "text_explain": "#FFD9B3",
    "text_muted": "#8A6B4A",
    "accent": "#FF6B00",
    "accent_hover": "#FF8533",
    "accent_subtle": "#331A00",
    "queue_blue": "#FF8C1A",
    "queue_blue_hover": "#FFB366",
    "border": "#3A2410",
    "border_focus": "#FF6B00",
    "scrollbar_thumb": "#FF6B00",
    "scrollbar_thumb_hover": "#FF8533",
    "success": "#FF6B00",
    "warning": "#FF8C1A",
    "error": "#FF3B00",
}

def _patch_automagic_catalog():
    try:
        from fizgig.training import optimizers as opt_mod
        # Inject automagic variants into catalog so available_optimizers() sees them
        # Use None module check (always available) since we ship them
        opt_mod._CATALOG["automagic"]  = (None, "Automagic v1 — polarity adaptive (experimental, ai-toolkit)")
        opt_mod._CATALOG["automagic2"] = (None, "Automagic v2 — per-tensor adaptive (experimental)")
        opt_mod._CATALOG["automagic3"] = (None, "Automagic v3 — polarity-history v3, fused, v3 pooled (experimental, recommended)")
        # Also wire create_optimizer to handle these names directly (maps to our port)
        _orig_create = opt_mod.create_optimizer
        def _chaotic_create(name, params, lr, args_str="", eps_floor_8bit=False):
            key = (name or "").strip().lower()
            if key in ("automagic", "automagic2", "automagic3"):
                mod_map = {
                    "automagic": "extensions.chaotic.optimizers.automagic",
                    "automagic2": "extensions.chaotic.optimizers.automagic2",
                    "automagic3": "extensions.chaotic.optimizers.automagic3",
                }
                import importlib
                cls_name = "Automagic" if key=="automagic" else "Automagic2" if key=="automagic2" else "Automagic3"
                mod = importlib.import_module(mod_map[key])
                cls = getattr(mod, cls_name)
                kwargs = opt_mod.parse_optimizer_args(args_str)
                # automagic3 defaults: lr is start lr, controller adapts
                opt = cls(params, lr=lr, **kwargs)
                label = f"{key}({args_str.strip()})" if args_str.strip() else key
                print(f"[chaotic][optim] {label} — {opt_mod.describe(key)}")
                return opt, label
            return _orig_create(name, params, lr, args_str, eps_floor_8bit)
        opt_mod.create_optimizer = _chaotic_create
        print("[chaotic] automagic v1/v2/v3 injected into optimizer catalog")
    except Exception as e:
        print(f"[chaotic] automagic catalog patch failed: {e}")
        import traceback; traceback.print_exc()

def apply_chaotic_patches(GUIClass):
    # Apply Krea2 Ostris patch early (no UI)
    if 'apply_krea2_ostris_patch' in globals() and apply_krea2_ostris_patch:
        try: apply_krea2_ostris_patch()
        except Exception as e: print(f"[chaotic] krea2 ostris patch failed: {e}")
    # Inject automagic optimizers
    try: _patch_automagic_catalog()
    except Exception as e: print(f"[chaotic] automagic inject failed: {e}")
    # Apply orange/black theme BEFORE any widget is created - patch the module COLORS dict
    try:
        import lora_trainer_gui as gui_mod
        # Update dict in place so every reference sees orange
        for k, v in CHAOTIC_COLORS.items():
            if k in gui_mod.COLORS:
                gui_mod.COLORS[k] = v
        # Also keep module globals in sync (BG_COLOR etc alias COLORS values at import time)
        try:
            gui_mod.BG_COLOR = gui_mod.COLORS["bg_deep"]
            gui_mod.FG_COLOR = gui_mod.COLORS["text_primary"]
            gui_mod.ACCENT_COLOR = gui_mod.COLORS["accent"]
            gui_mod.ENTRY_BG = gui_mod.COLORS["bg_surface"]
            gui_mod.BUTTON_ACTIVE = gui_mod.COLORS["bg_hover"]
            gui_mod.BORDER_COLOR = gui_mod.COLORS["border"]
        except: pass
        print("[chaotic] orange/black theme applied")
    except Exception as e:
        print(f"[chaotic] theme patch failed: {e}")

    orig_init = GUIClass.__init__
    def chaotic_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # ttk style orange after widgets exist
        try:
            _apply_orange_ttk_style(self)
        except Exception as e:
            print(f"[chaotic] ttk style failed: {e}")
        try:
            _inject_chaotic_ui(self)
        except Exception as e:
            print(f"[chaotic] UI injection failed: {e}")
            import traceback; traceback.print_exc()
        # Chaotic defaults & extras
        try:
            _inject_chaotic_start_ui(self)
        except Exception as e:
            print(f"[chaotic] start control/reg patch failed: {e}")
        try:
            _patch_sample_default(self)
        except Exception as e:
            print(f"[chaotic] sample default patch failed: {e}")
        try:
            _inject_chaotic_gizmo_auto(self)
        except Exception as e:
            print(f"[chaotic] gizmo auto patch failed: {e}")
        try:
            _patch_optimizer_tooltips(self)
        except Exception as e:
            print(f"[chaotic] tooltip patch failed: {e}")
        try:
            _patch_image_prep(self)
        except Exception as e:
            print(f"[chaotic] image prep patch failed: {e}")
        try:
            _patch_slider_validation(self)
        except Exception as e:
            print(f"[chaotic] slider validation patch failed: {e}")
    GUIClass.__init__ = chaotic_init

    # Patch start_training for steps->epochs and control_directory
    if hasattr(GUIClass, 'start_training'):
        orig_start = GUIClass.start_training
        def patched_start(self):
            try:
                ctrl_var = getattr(self, '_chaotic_control_dir_var', None)
                if ctrl_var is not None:
                    ctrl = ctrl_var.get().strip()
                    self._chaotic_control_dir = ctrl if ctrl and os.path.isdir(ctrl) else None
                    if self._chaotic_control_dir:
                        print(f"[chaotic] Control+Target active: target={self.image_folder_var.get()} control={self._chaotic_control_dir}")
                # sync advanced chaotic fields into real entries before start
                _sync_chaotic_to_real(self)
            except Exception as e:
                print(f"[chaotic] control sync failed: {e}")
            swapped = False
            old_val = None
            try:
                if getattr(self, '_chaotic_steps_mode', None) and self._chaotic_steps_mode.get():
                    e_widget = self.entries.get('MAX_TRAIN_EPOCHS')
                    steps_widget = self.entries.get('CHAOTIC_STEPS')
                    if e_widget is not None and steps_widget is not None:
                        try:
                            steps = int(steps_widget.get().strip())
                        except: steps = 0
                        if steps > 0:
                            folder = self.image_folder_var.get() if hasattr(self, 'image_folder_var') else ''
                            spe = _steps_per_epoch(folder)
                            epochs = _epochs_from_steps(steps, spe)
                            old_val = e_widget.get()
                            e_widget.delete(0, tk.END)
                            e_widget.insert(0, str(epochs))
                            swapped = True
                            print(f"[chaotic] Steps {steps} -> epochs {epochs} (spe={spe})")
            except Exception as e:
                print(f"[chaotic] pre-start steps swap failed: {e}")
            try:
                return orig_start(self)
            finally:
                if swapped:
                    try:
                        e_widget.delete(0, tk.END)
                        e_widget.insert(0, old_val)
                    except: pass
        GUIClass.start_training = patched_start

    # Patch TOML generation if exists (fallback)
    for meth_name in ('_generate_dataset_toml', '_build_dataset_config', '_write_dataset_config', '_create_dataset_config'):
        if hasattr(GUIClass, meth_name):
            orig = getattr(GUIClass, meth_name)
            def make_patched(orig_fn):
                def patched(self, *a, **kw):
                    res = orig_fn(self, *a, **kw)
                    try:
                        ctrl = getattr(self, '_chaotic_control_dir', None)
                        if ctrl and isinstance(res, dict) and 'datasets' in res:
                            for ds in res['datasets']:
                                ds['control_directory'] = ctrl
                            print(f"[chaotic] Injected control_directory into dataset TOML: {ctrl}")
                        if isinstance(res, str) and os.path.isfile(res):
                            ctrl = getattr(self, '_chaotic_control_dir', None)
                            if ctrl:
                                _patch_toml_file(res, ctrl)
                    except Exception as e:
                        print(f"[chaotic] TOML patch failed: {e}")
                    return res
                return patched
            setattr(GUIClass, meth_name, make_patched(orig))
    print("[chaotic] patches applied")

def _patch_toml_file(path: str, control_dir: str):
    try:
        import toml
        data = toml.load(path)
        if 'datasets' in data:
            for ds in data['datasets']:
                ds['control_directory'] = control_dir
            with open(path, 'w', encoding='utf-8') as f:
                toml.dump(data, f)
    except:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                txt = f.read()
            if 'control_directory' not in txt:
                txt = txt.replace('image_directory', f'control_directory = "{control_dir}"\nimage_directory')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(txt)
        except: pass

def _sync_chaotic_to_real(self):
    # Chaotic advanced fields -> real entries (so trainer sees them)
    mapping = {
        'CHAOTIC_OPTIMIZER': 'OPTIMIZER_TYPE',
        'CHAOTIC_SCHEDULER': 'LR_SCHEDULER',
        'CHAOTIC_WARMUP': 'LR_WARMUP_STEPS',
        'CHAOTIC_GRAD_ACCUM': 'GRADIENT_ACCUMULATION',
        'CHAOTIC_MAX_NORM': 'MAX_GRAD_NORM',
        'CHAOTIC_SAVE_EVERY': 'SAVE_EVERY_N_EPOCHS',
        'CHAOTIC_DROPOUT': 'NETWORK_DROPOUT',
    }
    for src, dst in mapping.items():
        if src in self.entries and dst in self.entries:
            try:
                val = self.entries[src].get()
                dst_w = self.entries[dst]
                # ttk.Combobox vs Entry handling
                try:
                    dst_w.delete(0, tk.END)
                    dst_w.insert(0, str(val))
                except:
                    try:
                        dst_w.set(str(val))
                    except: pass
            except: pass

def _apply_orange_ttk_style(self):
    try:
        style = ttk.Style()
        # use clam as base for color control
        try: style.theme_use("clam")
        except: pass
        # TButton orange
        style.configure("TButton", background="#1A1A1A", foreground="#FFF2E6", bordercolor="#FF6B00")
        style.map("TButton", background=[("active","#FF6B00"), ("pressed","#CC5500")], foreground=[("active","#FFFFFF")])
        style.configure("TCheckbutton", background="#1A1A1A", foreground="#FFF2E6")
        style.map("TCheckbutton", background=[("active","#1A1A1A")])
        style.configure("TCombobox", fieldbackground="#1A1A1A", background="#1A1A1A", foreground="#FFF2E6", arrowcolor="#FF6B00")
        style.configure("TEntry", fieldbackground="#1A1A1A", foreground="#FFF2E6")
        style.configure("TFrame", background="#0A0A0A")
        style.configure("TLabel", background="#1A1A1A", foreground="#FFF2E6")
        # Notebook tabs orange when selected
        style.configure("TNotebook", background="#0A0A0A", bordercolor="#FF6B00")
        style.map("TNotebook.Tab", background=[("selected","#FF6B00"), ("!selected","#1A1A1A")], foreground=[("selected","#FFFFFF"), ("!selected","#FFB266")])
        # Progressbar
        style.configure("Horizontal.TProgressbar", background="#FF6B00", troughcolor="#1A1A1A", bordercolor="#FF6B00")
    except Exception as e:
        print(f"[chaotic] ttk detail failed: {e}")
    # force root bg
    try:
        self.master.configure(bg="#0A0A0A")
        # ttk style for scrollbar thumb already via COLORS
    except: pass

def _inject_chaotic_ui(self):
    try:
        # title already has [CHAOTIC] from launcher, ensure orange indicator
        t = self.master.title()
        if "[CHAOTIC" not in t:
            self.master.title(t + " [CHAOTIC - ORANGE]")
        # also set icon background via root option
        try: self.master.configure(bg="#0A0A0A")
        except: pass
    except: pass
    print("[chaotic] injecting UI...")
    # import COLORS and CollapsibleFrame from gui module
    try:
        import lora_trainer_gui as gui_mod
        global COLORS
        COLORS = gui_mod.COLORS
        CollapsibleFrame = gui_mod.CollapsibleFrame
    except Exception as e:
        print(f"[chaotic] import failed: {e}")
        return
    # Find outer container (the scrollable_frame's outer)
    outer = None
    sec = getattr(self, 'collapsible_sections', {}).get('training')
    if sec:
        try:
            outer = sec.master  # outer is parent of training section
        except: pass
    if outer is None:
        # fallback: find any Frame with bg deep
        try:
            # training parent fallback
            training_parent = self.entries['MAX_TRAIN_EPOCHS'].master
            # climb 3 levels to outer
            outer = training_parent.master.master if training_parent else None
        except: pass
    if outer is None:
        print("[chaotic] could not find outer frame")
        return
    print(f"[chaotic] outer found: {outer}")

    # --- Create Chaotic Advanced Section as CollapsibleFrame ---
    # User wants it BELOW Presets and ABOVE Output (not after Training)
    try:
        chaotic_section = CollapsibleFrame(outer, "Chaotic — Advanced Training (AI-Toolkit parity)  ✦  Steps · Control · Optimizer · Scheduler", default_expanded=True)
        output_sec = getattr(self, 'collapsible_sections', {}).get('output')
        if output_sec is not None and str(output_sec.winfo_manager()) == "pack":
            try:
                chaotic_section.pack(fill=tk.X, padx=36, pady=(0, 16), before=output_sec)
                print(f"[chaotic] packed before output_section (below Presets, above Output)")
            except Exception as e:
                print(f"[chaotic] before-pack failed ({e}), fallback to simple pack")
                chaotic_section.pack(fill=tk.X, padx=36, pady=(0, 16))
        else:
            chaotic_section.pack(fill=tk.X, padx=36, pady=(0, 16))
            print(f"[chaotic] no output_sec to anchor, packed at end")
        content = chaotic_section.get_content_frame()
        content.columnconfigure(1, weight=1)
        self._chaotic_section = chaotic_section
        self._chaotic_content = content
    except Exception as e:
        print(f"[chaotic] failed to create section: {e}")
        return

    row = 0
    # Banner
    tk.Label(content, text="Steps mode + Control+Target + Optimizer/Scheduler — tudo aqui. Valores sincronizam com as secoes originais.",
             font=("Segoe UI", 9, "italic"), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
             wraplength=700, justify=tk.LEFT).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4,8))
    row+=1

    # --- Steps vs Epochs ---
    tk.Label(content, text="Training length:", font=("Segoe UI", 10, "bold"), fg="#FF6B00", bg=COLORS["bg_surface"]).grid(row=row, column=0, sticky=tk.W, padx=5, pady=(6,2))
    row+=1
    self._chaotic_steps_mode = tk.BooleanVar(value=False)
    try:
        import json as _json
        pref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "presets", "chaotic.json")
        pref_path = os.path.abspath(pref_path)
        if os.path.isfile(pref_path):
            with open(pref_path, 'r', encoding='utf-8') as f:
                pj = _json.load(f)
                if pj.get('steps_mode'): self._chaotic_steps_mode.set(True)
    except: pass
    cb = ttk.Checkbutton(content, text="Use Steps instead of Epochs (AI-Toolkit: train.steps)", variable=self._chaotic_steps_mode, command=lambda: _on_steps_toggle(self))
    cb.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
    row+=1
    steps_frame = ttk.Frame(content)
    steps_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=(20,5), pady=2)
    ttk.Label(steps_frame, text="Max Steps:").pack(side=tk.LEFT, padx=(0,4))
    default_steps = "1000"
    try:
        epochs_val = int(self.entries.get('MAX_TRAIN_EPOCHS').get() or 0)
        spe = _steps_per_epoch(self.image_folder_var.get() if hasattr(self,'image_folder_var') else '')
        if epochs_val and spe: default_steps = str(epochs_val * spe)
    except: pass
    ent = ttk.Entry(steps_frame, width=10)
    ent.insert(0, default_steps)
    ent.pack(side=tk.LEFT)
    self.entries['CHAOTIC_STEPS'] = ent
    self._chaotic_steps_frame = steps_frame
    conv_label = tk.Label(steps_frame, text="", font=("Segoe UI", 9, "italic"), fg="#8A9BAE", bg=COLORS["bg_surface"])
    conv_label.pack(side=tk.LEFT, padx=(10,0))
    self._chaotic_conv_label = conv_label
    def _update_conv(*_):
        try:
            steps = int(ent.get().strip() or 0)
            spe = _steps_per_epoch(self.image_folder_var.get() if hasattr(self,'image_folder_var') else '')
            if steps and spe:
                epochs = _epochs_from_steps(steps, spe)
                conv_label.config(text=f"≈ {epochs} epochs @ {spe} imgs/epoch  •  save every {max(1, epochs//5)}")
            else:
                conv_label.config(text="(set Dataset folder to see conversion)")
        except: conv_label.config(text="")
    ent.bind("<KeyRelease>", _update_conv)
    ent.bind("<FocusOut>", _update_conv)
    try: self.image_folder_var.trace_add("write", lambda *_: _update_conv())
    except: pass
    _update_conv()
    row+=1
    tk.Label(content, text="Fizgig usa epochs. AI-Toolkit usa steps. Chaotic converte: epochs = ceil(steps / num_images).", font=("Segoe UI", 9, "italic"), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=20, pady=(0,6))
    row+=1
    ttk.Separator(content, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=8)
    row+=1

    # Control moved to Start tab (together with Regularization, both optional)
    tk.Label(content, text="Control + Regularization moved → Start tab (below Dataset folder, optional)", font=("Segoe UI", 9, "italic"), fg="#FF6B00", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=4)
    # Keep vars for backwards compat but init them if not yet created (Start tab will own them)
    if not hasattr(self, '_chaotic_control_dir_var'):
        self._chaotic_control_dir_var = tk.StringVar(value=getattr(self, '_chaotic_control_dir', "") or "")
        self.entries['CHAOTIC_CONTROL_DIR'] = tk.Entry(content, textvariable=self._chaotic_control_dir_var)  # hidden, for sync
        self.entries['CHAOTIC_CONTROL_DIR'].grid_remove()
    if not hasattr(self, '_chaotic_enable_control'):
        self._chaotic_enable_control = tk.BooleanVar(value=bool(self._chaotic_control_dir_var.get()))
    # Pairing label kept for status (now in Start tab, but keep hidden label for logic)
    if not hasattr(self, '_chaotic_pairing_label'):
        self._chaotic_pairing_label = tk.Label(content, text="", bg=COLORS["bg_surface"])
        self._chaotic_pairing_label.grid_remove()
    row+=1
    ttk.Separator(content, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=8)
    row+=1

    # Krea2 Ostris moved to Samples tab for live edit testing
    tk.Label(content, text="Krea2 Ostris Edit moved → Samples tab (for live edit testing)", font=("Segoe UI", 9, "italic"), fg="#FF6B00", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=4)
    row+=1
    ttk.Separator(content, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=8)
    row+=1

    # --- Optimizer / Scheduler / Saves / VRAM ---
    tk.Label(content, text="Optimizer & Scheduler (AI-Toolkit: optimizer, lr_scheduler)", font=("Segoe UI", 10, "bold"), fg="#FF6B00", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(2,4))
    row+=1
    # Optimizer
    ttk.Label(content, text="Optimizer:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    opt_combo = ttk.Combobox(content, values=["adamw","adamw8bit","lion","pagedadamw8bit","ademamix8bit","automagic","automagic2","automagic3"], width=22)
    # sync with real entry if exists
    real_opt = ""
    try: real_opt = self.entries.get('OPTIMIZER_TYPE').get() or "adamw8bit"
    except: real_opt="adamw8bit"
    opt_combo.set(real_opt)
    opt_combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_OPTIMIZER'] = opt_combo
    tk.Label(content, text="automagic3 = adaptive LR v3 (fused, polarity-history)", font=("Segoe UI", 8), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=row, column=2, sticky=tk.W, padx=5)
    # Live sync: chaotic -> real and real -> chaotic (avoid duplicate confusion)
    try:
        real_opt_combo = self.entries.get('OPTIMIZER_TYPE')
        if real_opt_combo is not None:
            def _sync_opt_to_real(*_a):
                try: real_opt_combo.set(opt_combo.get())
                except: pass
            def _sync_opt_to_chaotic(*_a):
                try: opt_combo.set(real_opt_combo.get())
                except: pass
            opt_combo.bind("<<ComboboxSelected>>", lambda e: _sync_opt_to_real())
            real_opt_combo.bind("<<ComboboxSelected>>", lambda e: _sync_opt_to_chaotic())
    except: pass
    row+=1
    # Scheduler
    ttk.Label(content, text="LR Scheduler:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    sched_combo = ttk.Combobox(content, values=["constant","constant_with_warmup","cosine","cosine_with_restarts","linear","polynomial","rex"], width=22)
    try: real_sched = self.entries.get('LR_SCHEDULER').get() or "constant"
    except: real_sched="constant"
    sched_combo.set(real_sched)
    sched_combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_SCHEDULER'] = sched_combo
    try:
        real_sched_combo = self.entries.get('LR_SCHEDULER')
        if real_sched_combo is not None:
            def _sync_sched_to_real(*_a):
                try: real_sched_combo.set(sched_combo.get())
                except: pass
            def _sync_sched_to_chaotic(*_a):
                try: sched_combo.set(real_sched_combo.get())
                except: pass
            sched_combo.bind("<<ComboboxSelected>>", lambda e: _sync_sched_to_real())
            real_sched_combo.bind("<<ComboboxSelected>>", lambda e: _sync_sched_to_chaotic())
    except: pass
    # Warmup
    warm_frame = ttk.Frame(content)
    warm_frame.grid(row=row, column=2, sticky=tk.W, padx=5, pady=3)
    tk.Label(warm_frame, text="Warmup:", font=("Segoe UI", 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
    warm_ent = ttk.Entry(warm_frame, width=6)
    try: warm_ent.insert(0, str(self.entries.get('LR_WARMUP_STEPS').get() or "0"))
    except: warm_ent.insert(0, "0")
    warm_ent.pack(side=tk.LEFT, padx=4)
    self.entries['CHAOTIC_WARMUP'] = warm_ent
    tk.Label(warm_frame, text="steps", font=("Segoe UI", 8), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT)
    row+=1

    # Grad accumulation / Max norm
    ttk.Label(content, text="Grad Accum:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    ga_ent = ttk.Entry(content, width=8)
    try: ga_ent.insert(0, str(self.entries.get('GRADIENT_ACCUMULATION').get() or "1"))
    except: ga_ent.insert(0, "1")
    ga_ent.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_GRAD_ACCUM'] = ga_ent
    tmp = ttk.Frame(content); tmp.grid(row=row, column=2, sticky=tk.W, padx=5, pady=3)
    ttk.Label(tmp, text="Max Grad Norm:").pack(side=tk.LEFT, padx=(0,4))
    mgn_ent = ttk.Entry(tmp, width=6)
    try: mgn_ent.insert(0, str(self.entries.get('MAX_GRAD_NORM').get() or "1.0"))
    except: mgn_ent.insert(0, "1.0")
    mgn_ent.pack(side=tk.LEFT)
    self.entries['CHAOTIC_MAX_NORM'] = mgn_ent
    row+=1

    # Save every / dropout
    ttk.Label(content, text="Save Every N epochs:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    se_ent = ttk.Entry(content, width=8)
    try: se_ent.insert(0, str(self.entries.get('SAVE_EVERY_N_EPOCHS').get() or "1"))
    except: se_ent.insert(0, "1")
    se_ent.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_SAVE_EVERY'] = se_ent
    tmp2 = ttk.Frame(content); tmp2.grid(row=row, column=2, sticky=tk.W, padx=5, pady=3)
    ttk.Label(tmp2, text="Network Dropout:").pack(side=tk.LEFT, padx=(0,4))
    do_ent = ttk.Entry(tmp2, width=6)
    try: do_ent.insert(0, str(self.entries.get('NETWORK_DROPOUT').get() or "0.0"))
    except: do_ent.insert(0, "0.0")
    do_ent.pack(side=tk.LEFT)
    self.entries['CHAOTIC_DROPOUT'] = do_ent
    row+=1

    # Caption dropout / noise (AI-Toolkit)
    ttk.Label(content, text="Caption Dropout:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    cd_ent = ttk.Entry(content, width=8)
    cd_ent.insert(0, "0.05")
    cd_ent.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_CAPTION_DROPOUT'] = cd_ent
    tk.Label(content, text="0.05 = 5% captions dropped (AI-Toolkit)", font=("Segoe UI", 8), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=row, column=2, sticky=tk.W, padx=5)
    row+=1

    # VRAM tip
    tk.Label(content, text="LoRA adapter: Fizgig ja treina LoRA (rank/alpha) e LoKR (Kronecker) — veja Training Parameters > Network Type. Para 4060 Ti 8GB: rank 8-16, 0.25 MP, Auto VRAM. Context LoRA = AI-Toolkit control_lora.",
             font=("Segoe UI", 8, "italic"), fg="#5A6B7E", bg=COLORS["bg_surface"], wraplength=680, justify=tk.LEFT).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(8,0))
    row+=1

    # Do NOT auto-expand collapsed Optimizer/Scheduler - that caused duplication confusion
    # Keep them collapsed; Chaotic values below OVERRIDE them at launch (see _sync_chaotic_to_real)
    tk.Label(content, text="Note: values above override the collapsed Optimizer / Other Options sections below when set.", font=("Segoe UI", 8, "italic"), fg="#FF8C1A", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4,0))
    row+=1

    _on_steps_toggle(self)
    print("[chaotic] advanced UI injected")
    # also inject Samples tab edit UI
    try:
        _inject_chaotic_samples_ui(self)
    except Exception as e:
        print(f"[chaotic] samples UI failed: {e}")
        import traceback; traceback.print_exc()

def _inject_chaotic_samples_ui(self):
    # Samples tab - Krea2 Ostris Edit injected directly into Prompt & Dimensions card (below its fields)
    try:
        import lora_trainer_gui as gui_mod
        global COLORS
        COLORS = gui_mod.COLORS
    except: return
    # locate Prompt & Dimensions card via sample_prompt_text's parent
    prompt_card = None
    try:
        if hasattr(self, 'sample_prompt_text'):
            prompt_card = self.sample_prompt_text.master  # prompt_card is parent of the Text widget
    except: pass
    if prompt_card is None:
        print("[chaotic] could not find prompt_card")
        return
    print(f"[chaotic] prompt_card found: {prompt_card}")
    try:
        from lora_trainer_gui import COLORS as _C
        # Determine next free row in prompt_card (currently rows 0-10 used, next is 11)
        # Find max grid row used
        max_row = 0
        try:
            for child in prompt_card.winfo_children():
                info = child.grid_info()
                if info:
                    r = int(info.get('row', 0))
                    if r > max_row: max_row = r
        except: max_row = 10
        r = max_row + 1
        # Container for Krea2 edit so we can hide/show on arch change (H3 glitch fix)
        container = tk.Frame(prompt_card, bg=_C["bg_surface"])
        container.grid(row=r, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        self._chaotic_krea2_container = container
        # Inside container use pack for simplicity (hidden via container grid_remove)
        ttk.Separator(container, orient="horizontal").pack(fill=tk.X, pady=6)
        tk.Label(container, text="Chaotic — Krea2 Ostris Edit (test edit in sampling)", font=("Segoe UI", 10, "bold"), fg="#FF6B00", bg=_C["bg_surface"]).pack(anchor=tk.W, padx=5, pady=(2,2))
        tk.Label(container, text="Qwen3-VL vision (384px) + VAE latents (1MP t=0) + kv_cache. Use to test if edit LoRA works live - refs become tokens at t=0 (index_timestep_zero).", font=("Segoe UI", 8), fg=_C["text_muted"], bg=_C["bg_surface"], wraplength=640, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(0,4))
        tk.Label(container, text="Reference images (up to 3) — Picture N + VAE:", font=("Segoe UI", 9, "bold"), fg="#FF6B00", bg=_C["bg_surface"]).pack(anchor=tk.W, padx=5, pady=(4,2))
        self._chaotic_krea2_images = []
        self._chaotic_kv_cache = tk.BooleanVar(value=False)
        for idx in range(3):
            fr = ttk.Frame(container)
            fr.pack(fill=tk.X, padx=5, pady=2, anchor=tk.W)
            ttk.Label(fr, text=f"Image {idx+1}:").pack(side=tk.LEFT, padx=(0,4))
            var = tk.StringVar(value="")
            ent = ttk.Entry(fr, textvariable=var, width=42)
            ent.pack(side=tk.LEFT)
            self._chaotic_krea2_images.append(var)
            self.entries[f'CHAOTIC_KREA2_IMG{idx+1}'] = ent
            def _mk(v=var, i=idx):
                def _b():
                    p = filedialog.askopenfilename(title=f"Select reference {i+1}", filetypes=[("Images","*.png *.jpg *.jpeg *.webp"),("All","*.*")])
                    if p: v.set(p)
                return _b
            ttk.Button(fr, text="Browse", command=_mk()).pack(side=tk.LEFT, padx=(4,0))
            ttk.Button(fr, text="Clear", command=lambda v=var: v.set("")).pack(side=tk.LEFT, padx=(2,0))
        kv_fr = ttk.Frame(container)
        kv_fr.pack(fill=tk.X, padx=5, pady=4, anchor=tk.W)
        ttk.Checkbutton(kv_fr, text="kv_cache (LoRA trained with kv_cache=true)", variable=self._chaotic_kv_cache).pack(side=tk.LEFT)
        self.entries['CHAOTIC_KV_CACHE'] = self._chaotic_kv_cache
        tk.Label(kv_fr, text="off=normal edit, on=cached KV (faster, only if LoRA trained with it)", font=("Segoe UI", 8), fg=_C["text_muted"], bg=_C["bg_surface"]).pack(side=tk.LEFT, padx=(8,0))
        tk.Label(container, text="How to test: select 1-3 refs, keep VAE connected (for VAE latent), prompt with trigger, Generate Sample. Refs are encoded via Qwen (semantic) + VAE (t=0).", font=("Segoe UI", 8, "italic"), fg=_C["text_explain"], bg=_C["bg_surface"], wraplength=640, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(4,0))
        # Arch-aware visibility: show only for Krea2, hide for Klein/H3
        def _update_krea2_visibility(*_a):
            try:
                is_krea2 = False
                if hasattr(self, '_is_krea2_arch'):
                    is_krea2 = self._is_krea2_arch()
                elif hasattr(self, 'architecture_var'):
                    is_krea2 = "Krea" in str(self.architecture_var.get())
                if is_krea2:
                    container.grid()
                else:
                    container.grid_remove()
            except: pass
        # trace arch change + call once
        try:
            if hasattr(self, 'architecture_var'):
                self.architecture_var.trace_add("write", lambda *_: _update_krea2_visibility())
            # also hook the sample arch combo if exists
            if hasattr(self, '_samples_arch_combo'):
                self._samples_arch_combo.bind("<<ComboboxSelected>>", lambda e: _update_krea2_visibility())
        except: pass
        _update_krea2_visibility()
        print("[chaotic] Krea2 edit injected into Prompt & Dimensions card (Krea2-only, hidden for H3)")
    except Exception as e:
        print(f"[chaotic] samples card failed: {e}")
        import traceback; traceback.print_exc()

def _inject_chaotic_start_ui(self):
    # Start tab: Optional Control + Regularization (both optional, together)
    try:
        import lora_trainer_gui as gui_mod
        COLORS = gui_mod.COLORS
        FONT_FAMILY = gui_mod.FONT_FAMILY
    except:
        COLORS = {"bg_surface":"#1A1A1A", "text_explain":"#FFD9B3", "text_muted":"#8A6B4A", "bg_deep":"#0A0A0A"}
        FONT_FAMILY = "Segoe UI"
    # Locate Start tab outer via image_folder_var entry
    outer = None
    training_card = None
    try:
        if hasattr(self, 'image_folder_var'):
            for w in self.master.winfo_children():
                def _find_entry(widget):
                    for child in widget.winfo_children():
                        try:
                            if isinstance(child, (tk.Entry, ttk.Entry)) and child.cget("textvariable") == str(self.image_folder_var):
                                return child
                        except: pass
                        res = _find_entry(child)
                        if res: return res
                    return None
                ent = _find_entry(w)
                if ent is not None:
                    try:
                        training_card = ent.master.master  # card
                        outer = training_card.master  # outer
                        break
                    except: pass
        if outer is None and hasattr(self, 'collapsible_sections'):
            sec = self.collapsible_sections.get("training")
            if sec: outer = sec.master
    except: pass
    if outer is None:
        outer = self.master
        print("[chaotic] start outer fallback to master")
    print(f"[chaotic] start outer found: {outer}, training_card: {training_card}")
    try:
        # Use _start_section_card if available
        card = None
        if hasattr(self, '_start_section_card'):
            card = self._start_section_card(outer, "Optional — Control & Regularization", "Control (paired edit) and Regularization images are optional. When enabled, they are included in Image Prep and training. Control needs same basenames as target; Regularization is extra images to prevent overfitting (video+image).")
        else:
            card = tk.Frame(outer, bg=COLORS["bg_surface"], highlightbackground=COLORS.get("border","#3A2410"), highlightthickness=1)
            card.pack(fill=tk.X, padx=36, pady=(0,16))
            tk.Label(card, text="Optional — Control & Regularization", font=(FONT_FAMILY, 12, "bold"), fg=COLORS.get("text_primary","#FFF2E6"), bg=COLORS["bg_surface"]).pack(anchor=tk.W, padx=12, pady=8)
        card.grid_columnconfigure(1, weight=1)
        # Move card to be directly after Training image folder and hide Post-Training Tools shortcut (user request to give space)
        try:
            # training_card already from outer finding via entry; if not, search
            if 'training_card' not in locals() or training_card is None:
                def _contains_text(widget, text):
                    for c in widget.winfo_children():
                        try:
                            if isinstance(c, tk.Label) and text in str(c.cget("text")):
                                return True
                        except: pass
                        if _contains_text(c, text):
                            return True
                    return False
                for ch in outer.winfo_children():
                    if _contains_text(ch, "Training image folder"):
                        training_card = ch
                        break
            # Find Post-Training Tools card to hide it
            def _contains_text2(widget, text):
                for c in widget.winfo_children():
                    try:
                        if isinstance(c, tk.Label) and text in str(c.cget("text")):
                            return True
                    except: pass
                    if _contains_text2(c, text):
                        return True
                return False
            post_card = None
            for ch in outer.winfo_children():
                if _contains_text2(ch, "Post-Training Tools"):
                    post_card = ch
                    break
            if training_card is not None and training_card != card:
                card.pack_forget()
                card.pack(fill=tk.X, padx=36, pady=(0,16), after=training_card)
                print(f"[chaotic] Start card reordered after Training image folder")
                if post_card is not None:
                    try:
                        post_card.pack_forget()
                        print(f"[chaotic] Post-Training Tools hidden to give space for Optional")
                    except: pass
            elif post_card is not None and post_card != card:
                card.pack_forget()
                card.pack(fill=tk.X, padx=36, pady=(0,16), before=post_card)
                print(f"[chaotic] Start card reordered before Post-Training Tools (fallback)")
                try:
                    post_card.pack_forget()
                    print(f"[chaotic] Post-Training Tools hidden (fallback)")
                except: pass
        except Exception as e:
            print(f"[chaotic] start reorder failed: {e}")
            import traceback; traceback.print_exc()
        # Enable Control
        if not hasattr(self, '_chaotic_enable_control'):
            self._chaotic_enable_control = tk.BooleanVar(value=bool(getattr(self, '_chaotic_control_dir_var', tk.StringVar()).get() if hasattr(self, '_chaotic_control_dir_var') else False))
        if not hasattr(self, '_chaotic_control_dir_var'):
            self._chaotic_control_dir_var = tk.StringVar(value="")
        # If hidden entry exists, reuse its var
        # Control row
        r0 = tk.Frame(card, bg=COLORS["bg_surface"])
        r0.pack(fill=tk.X, padx=5, pady=4)
        ttk.Checkbutton(r0, text="Enable Control (paired edit) — depth/pose/canny/edit", variable=self._chaotic_enable_control).pack(side=tk.LEFT)
        tk.Label(r0, text="Control folder:", font=(FONT_FAMILY, 9), fg=COLORS.get("text_muted","#8A6B4A"), bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(12,4))
        # Ensure entry exists - make it expand to fill row (fix misalignment)
        if 'CHAOTIC_CONTROL_DIR' not in self.entries or not isinstance(self.entries['CHAOTIC_CONTROL_DIR'], tk.Widget):
            ent = ttk.Entry(r0, textvariable=self._chaotic_control_dir_var, width=28)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4,0))
            self.entries['CHAOTIC_CONTROL_DIR'] = ent
        else:
            # Reparent: create visible entry in this row
            ent = ttk.Entry(r0, textvariable=self._chaotic_control_dir_var, width=28)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4,0))
            self.entries['CHAOTIC_CONTROL_DIR_VISIBLE'] = ent
        def _browse_ctrl2():
            d = filedialog.askdirectory(title="Select CONTROL folder (paired to Dataset)")
            if d:
                self._chaotic_control_dir_var.set(d)
                self._chaotic_enable_control.set(True)
                self._chaotic_control_dir = d
        ttk.Button(r0, text="Browse", command=_browse_ctrl2).pack(side=tk.LEFT, padx=(4,0))
        ttk.Button(r0, text="Clear", command=lambda: (self._chaotic_control_dir_var.set(""), self._chaotic_enable_control.set(False))).pack(side=tk.LEFT, padx=(2,0))
        # Pairing status
        pairing2 = tk.Label(card, text="", font=(FONT_FAMILY, 8, "italic"), fg="#8A9BAE", bg=COLORS["bg_surface"], wraplength=680, justify=tk.LEFT)
        pairing2.pack(anchor=tk.W, padx=5, pady=2)
        self._chaotic_pairing_label2 = pairing2
        # Regularization
        if not hasattr(self, '_chaotic_enable_reg'):
            self._chaotic_enable_reg = tk.BooleanVar(value=False)
        if not hasattr(self, '_chaotic_reg_dir_var'):
            self._chaotic_reg_dir_var = tk.StringVar(value="")
        r1 = tk.Frame(card, bg=COLORS["bg_surface"])
        r1.pack(fill=tk.X, padx=5, pady=4)
        ttk.Checkbutton(r1, text="Enable Regularization — extra images to preserve prior (video+image)", variable=self._chaotic_enable_reg).pack(side=tk.LEFT)
        tk.Label(r1, text="Reg folder:", font=(FONT_FAMILY, 9), fg=COLORS.get("text_muted","#8A6B4A"), bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=(12,4))
        reg_ent = ttk.Entry(r1, textvariable=self._chaotic_reg_dir_var, width=28)
        reg_ent.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4,0))
        self.entries['CHAOTIC_REG_DIR'] = reg_ent
        def _browse_reg():
            d = filedialog.askdirectory(title="Select REGULARIZATION folder")
            if d:
                self._chaotic_reg_dir_var.set(d)
                self._chaotic_enable_reg.set(True)
        ttk.Button(r1, text="Browse", command=_browse_reg).pack(side=tk.LEFT, padx=(4,0))
        ttk.Button(r1, text="Clear", command=lambda: (self._chaotic_reg_dir_var.set(""), self._chaotic_enable_reg.set(False))).pack(side=tk.LEFT, padx=(2,0))
        # Slider toggle (no data)
        if not hasattr(self, '_chaotic_slider_mode'):
            self._chaotic_slider_mode = tk.BooleanVar(value=False)
        r2 = tk.Frame(card, bg=COLORS["bg_surface"])
        r2.pack(fill=tk.X, padx=5, pady=4)
        ttk.Checkbutton(r2, text="Slider LoRA (no data needed) — train concept slider via prompts (pos/neg) without images", variable=self._chaotic_slider_mode).pack(side=tk.LEFT)
        self.entries['CHAOTIC_SLIDER_MODE'] = self._chaotic_slider_mode
        tk.Label(card, text="When Slider is ON, image folder can be empty; training uses prompt anchors/targets. Regularization still recommended.", font=(FONT_FAMILY, 8, "italic"), fg=COLORS.get("text_explain","#FFD9B3"), bg=COLORS["bg_surface"], wraplength=680, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=2)
        # Update pairing for new location
        def _update_start_pairing(*_a):
            try:
                target = self.image_folder_var.get() if hasattr(self, 'image_folder_var') else ''
                ctrl = self._chaotic_control_dir_var.get().strip() if hasattr(self, '_chaotic_control_dir_var') else ''
                enabled = self._chaotic_enable_control.get() if hasattr(self, '_chaotic_enable_control') else bool(ctrl)
                if not enabled or not ctrl:
                    pairing2.config(text="Control disabled → unpaired (default).", fg="#8A9BAE"); return
                if not os.path.isdir(ctrl):
                    pairing2.config(text="Control folder doesn't exist.", fg="#EF4444"); return
                if not target or not os.path.isdir(target):
                    pairing2.config(text=f"Control: {ctrl} — set Dataset folder to check pairing.", fg="#8A9BAE"); return
                t_files = set(os.path.splitext(f)[0] for f in os.listdir(target) if os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.webp'])
                c_files = set(os.path.splitext(f)[0] for f in os.listdir(ctrl) if os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.webp'])
                matched = len(t_files & c_files); total=len(t_files)
                if matched==0: pairing2.config(text=f"⚠ 0/{total} paired — need same basename!", fg="#F59E0B")
                elif matched<total: pairing2.config(text=f"⚠ {matched}/{total} paired — {total-matched} missing will ERROR", fg="#F59E0B")
                else: pairing2.config(text=f"✓ {matched}/{total} paired — control active (Klein/Edit)", fg="#10B981")
            except Exception as e: pairing2.config(text=f"Pairing check failed: {e}", fg="#F59E0B")
        try:
            self.image_folder_var.trace_add("write", lambda *_: _update_start_pairing())
            self._chaotic_control_dir_var.trace_add("write", lambda *_: _update_start_pairing())
            self._chaotic_enable_control.trace_add("write", lambda *_: _update_start_pairing())
        except: pass
        _update_start_pairing()
        print("[chaotic] Start tab Control & Regularization card injected")
    except Exception as e:
        print(f"[chaotic] start card failed: {e}")
        import traceback; traceback.print_exc()

def _on_steps_toggle(self):
    on = bool(self._chaotic_steps_mode.get())
    try:
        if on: self._chaotic_steps_frame.grid()
        else: self._chaotic_steps_frame.grid_remove()
        # When steps mode is on, disable original Max Epochs to avoid confusion (chaotic overrides)
        try:
            e = self.entries.get('MAX_TRAIN_EPOCHS')
            if e is not None:
                e.configure(state='disabled' if on else 'normal')
                # Find its label if stored
                lbl = self.labels.get('MAX_TRAIN_EPOCHS') if hasattr(self, 'labels') else None
                if lbl is not None:
                    lbl.configure(fg="#6B7280" if on else "#8A9BAE")
        except: pass
        # Also toggle note in chaotic card if exists
        try:
            if hasattr(self, '_chaotic_epochs_note'):
                self._chaotic_epochs_note.configure(text="Steps mode ON — Training Parameters Max Epochs is disabled, value from Max Steps will be used (overrides)." if on else "")
        except: pass
    except: pass
    try:
        import json as _json
        pref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "presets", "chaotic.json")
        pref_path = os.path.abspath(pref_path)
        os.makedirs(os.path.dirname(pref_path), exist_ok=True)
        with open(pref_path, 'w', encoding='utf-8') as f:
            _json.dump({"steps_mode": on}, f)
    except: pass

def _patch_image_prep(self):
    # Patch Image Prep to also process Control and Regularization folders when enabled
    try:
        orig_convert = getattr(self, 'convert_images', None)
        orig_auto = getattr(self, '_auto_prep_images', None)
        orig_resize = getattr(self, '_resize_only_images', None)
        orig_crop = getattr(self, '_face_crop_only_images', None)
        if not orig_convert:
            return
        # Wrap convert_images to also handle control/reg after main
        def _patched_convert():
            # Call original for main dataset
            result = orig_convert()
            # Then process control/reg if enabled
            try:
                import os as _os
                # Control
                if hasattr(self, '_chaotic_enable_control') and self._chaotic_enable_control.get():
                    ctrl = self._chaotic_control_dir_var.get().strip() if hasattr(self, '_chaotic_control_dir_var') else ""
                    if ctrl and _os.path.isdir(ctrl):
                        # Use same prep settings but output to same output folder? For paired training, control images should be processed similarly
                        # We reuse the same logic but with control as source
                        print(f"[chaotic][prep] also processing Control folder: {ctrl}")
                        # We can call the internal prep with same params but swapped source
                        # For now, just resize them similarly via simple resize (reuse target_area)
                        try:
                            # Access current prep settings
                            src = ctrl
                            out = self.convert_output_var.get() or src
                            # Use same target_area as main: from prep_megapixels_var
                            try:
                                ta = self._prep_target_area(float(self.prep_megapixels_var.get()))
                            except: ta = self._prep_target_area(1.0)
                            replace = self.delete_originals_var.get()
                            mode = self.prep_mode_var.get()
                            # Call appropriate internal method with control source
                            if mode == "Auto Prep (Face Crops)":
                                self._auto_prep_images(src, out, ta, self._get_face_selection_mode(), float(self.face_padding_var.get() or 20), replace)
                            elif mode == "Resize Only":
                                self._resize_only_images(src, out, ta, replace)
                            elif mode == "Face Crop Only":
                                self._face_crop_only_images(src, out, ta, self._get_face_selection_mode(), float(self.face_padding_var.get() or 20), replace)
                        except Exception as e:
                            print(f"[chaotic][prep] control prep failed: {e}")
                if hasattr(self, '_chaotic_enable_reg') and self._chaotic_enable_reg.get():
                    reg = self._chaotic_reg_dir_var.get().strip() if hasattr(self, '_chaotic_reg_dir_var') else ""
                    if reg and _os.path.isdir(reg):
                        print(f"[chaotic][prep] also processing Regularization folder: {reg}")
                        try:
                            src = reg
                            out = self.convert_output_var.get() or src
                            try: ta = self._prep_target_area(float(self.prep_megapixels_var.get()))
                            except: ta = self._prep_target_area(1.0)
                            replace = self.delete_originals_var.get()
                            mode = self.prep_mode_var.get()
                            if mode == "Auto Prep (Face Crops)":
                                self._auto_prep_images(src, out, ta, self._get_face_selection_mode(), float(self.face_padding_var.get() or 20), replace)
                            elif mode == "Resize Only":
                                self._resize_only_images(src, out, ta, replace)
                            elif mode == "Face Crop Only":
                                self._face_crop_only_images(src, out, ta, self._get_face_selection_mode(), float(self.face_padding_var.get() or 20), replace)
                        except Exception as e:
                            print(f"[chaotic][prep] reg prep failed: {e}")
            except Exception as e:
                print(f"[chaotic][prep] wrapper failed: {e}")
            return result
        self.convert_images = _patched_convert
        print("[chaotic] Image Prep patched to include Control & Regularization")
    except Exception as e:
        print(f"[chaotic] image prep patch failed: {e}")

def _patch_slider_validation(self):
    # Allow Slider LoRA training without images (no data needed)
    try:
        # Patch start_training validation that checks image folder exists
        orig_start = getattr(self, 'start_training', None)
        # Our patched_start already wraps, but we need to allow empty folder when slider on
        # Instead patch the validation method that checks folder before training
        # Find method that validates: often _validate_inputs or similar
        for meth_name in ['_validate_inputs', 'validate_inputs', '_check_dataset', '_validate_dataset']:
            if hasattr(self, meth_name):
                orig = getattr(self, meth_name)
                def _make(m):
                    def _patched(*a, **kw):
                        # If slider mode on, skip empty folder check
                        try:
                            if hasattr(self, '_chaotic_slider_mode') and self._chaotic_slider_mode.get():
                                print("[chaotic][slider] validation bypass - no data needed for slider")
                                return True
                        except: pass
                        return m(*a, **kw)
                    return _patched
                setattr(self, meth_name, _make(orig))
        # Also patch the direct folder check in start_training wrapper (already patched)
        # Ensure slider var is accessible
        print("[chaotic] slider validation patched (no data needed when Slider ON)")
    except Exception as e:
        print(f"[chaotic] slider patch failed: {e}")

def _patch_sample_default(self):
    # Chaotic default: Enable Sample Generation unchecked by default (user request)
    try:
        if hasattr(self, 'sample_enabled_var'):
            # Only change if default is checked and user hasn't manually set pref
            # Check chaotic pref for sample default
            import json as _json, os as _os
            pref_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "presets", "chaotic.json")
            pref_path = _os.path.abspath(pref_path)
            # If chaotic.json exists and has sample_enabled key, respect it; else set False
            should_uncheck = True
            if _os.path.isfile(pref_path):
                try:
                    with open(pref_path, encoding='utf-8') as f:
                        pj = _json.load(f)
                        if "sample_enabled" in pj:
                            should_uncheck = False  # user chose before
                except: pass
            if should_uncheck:
                self.sample_enabled_var.set(False)
                # also persist the choice
                try:
                    # update chaotic.json with sample_enabled
                    pj = {}
                    if _os.path.isfile(pref_path):
                        try:
                            with open(pref_path, encoding='utf-8') as f:
                                pj = _json.load(f)
                        except: pj = {}
                    pj["sample_enabled"] = False
                    with open(pref_path, 'w', encoding='utf-8') as f:
                        _json.dump(pj, f)
                except: pass
                print("[chaotic] sample generation default unchecked")
                # Force grey visual on start (was only grey after enable/disable toggle)
                try:
                    if hasattr(self, 'toggle_sample_settings'):
                        self.toggle_sample_settings()
                    elif hasattr(self, '_on_sample_enabled_toggle'):
                        self._on_sample_enabled_toggle()
                    # Also try generic update
                    if hasattr(self, 'sample_settings_frame'):
                        # If toggle didn't grey, manually set state
                        try:
                            # The sample_settings_frame contains the cards; when disabled it should look grey
                            # Force update by calling the toggle again after a short delay
                            self.master.after(100, lambda: getattr(self, 'toggle_sample_settings', lambda: None)())
                        except: pass
                except: pass
            # Also hook to persist future toggles
            try:
                self.sample_enabled_var.trace_add("write", lambda *_: _persist_sample_enabled(self))
            except: pass
    except Exception as e:
        print(f"[chaotic] sample default patch inner failed: {e}")

def _persist_sample_enabled(self):
    try:
        import json as _json, os as _os
        pref_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "presets", "chaotic.json")
        pref_path = _os.path.abspath(pref_path)
        pj = {}
        if _os.path.isfile(pref_path):
            try:
                with open(pref_path, encoding='utf-8') as f:
                    pj = _json.load(f)
            except: pj = {}
        pj["sample_enabled"] = bool(self.sample_enabled_var.get())
        _os.makedirs(_os.path.dirname(pref_path), exist_ok=True)
        with open(pref_path, 'w', encoding='utf-8') as f:
            _json.dump(pj, f)
    except: pass

def _inject_chaotic_gizmo_auto(self):
    # Find the "Open Gizmo" button and add "Auto" next to it
    try:
        import tkinter as _tk
        # Find gizmo row by searching for button with text "Open Gizmo"
        def _find_gizmo_row(widget):
            for child in widget.winfo_children():
                # check if this is button with Open Gizmo
                try:
                    if isinstance(child, _tk.Button) and "Open Gizmo" in str(child.cget("text")):
                        return child.master
                except: pass
                # recurse
                try:
                    res = _find_gizmo_row(child)
                    if res is not None:
                        return res
                except: pass
            return None
        gizmo_row = _find_gizmo_row(self.master)
        if gizmo_row is None:
            # fallback: try self
            gizmo_row = _find_gizmo_row(self)
        if gizmo_row is None:
            print("[chaotic] gizmo row not found for Auto button")
            return
        # Check if already added
        for c in gizmo_row.winfo_children():
            try:
                if "Auto" in str(c.cget("text")) and "Gizmo" in str(c.cget("text")):
                    print("[chaotic] Auto Gizmo button already exists")
                    return
            except: pass
        # Add Auto button
        try:
            import lora_trainer_gui as gui_mod
            COLORS = gui_mod.COLORS
            FONT_FAMILY = gui_mod.FONT_FAMILY
        except:
            COLORS = {"accent":"#FF6B00", "bg_surface":"#1A1A1A", "text_primary":"#FFF2E6"}
            FONT_FAMILY = "Segoe UI"
        # Patch method onto instance if not exists
        if not hasattr(self, '_chaotic_auto_video'):
            def _auto_video():
                _chaotic_auto_video_impl(self)
            self._chaotic_auto_video = _auto_video
        _tk.Button(gizmo_row, text="⚡ Auto", command=self._chaotic_auto_video,
                  bg="#FF6B00", fg="#FFFFFF",
                  activebackground="#FF8533", activeforeground="#FFFFFF",
                  font=(FONT_FAMILY, 9, "bold"), relief=_tk.FLAT, bd=0, padx=12, pady=6, cursor="hand2").pack(side=_tk.LEFT, padx=(8,0))
        _tk.Label(gizmo_row, text="auto-trim whole video to H3 spec (24fps, 17n+5 frames, 32k stereo)", font=(FONT_FAMILY, 8), fg="#8A9BAE" if "8A9BAE" in str(COLORS) else "#6B7280", bg=COLORS.get("bg_surface","#1A1A1A")).pack(side=_tk.LEFT, padx=8)
        print("[chaotic] Auto Gizmo button injected below Open Gizmo")
    except Exception as e:
        print(f"[chaotic] gizmo auto inject failed: {e}")
        import traceback; traceback.print_exc()

def _chaotic_auto_video_impl(gui):
    import os as _os, threading as _th, subprocess as _sp
    from tkinter import filedialog as _fd, messagebox as _mb
    # Ask source video
    src = _fd.askopenfilename(title="Select source video for Auto prep (whole video will be auto-trimmed)", filetypes=[("Video","*.mp4 *.mov *.avi *.mkv *.webm"),("All","*.*")])
    if not src or not _os.path.isfile(src):
        return
    # Ask output folder - default to image folder if set, else same dir as src
    default_out = ""
    try:
        default_out = gui.image_folder_var.get() if hasattr(gui, 'image_folder_var') else ""
        if not default_out or not _os.path.isdir(default_out):
            default_out = _os.path.dirname(src)
    except: default_out = _os.path.dirname(src)
    out_dir = _fd.askdirectory(title="Select output folder for auto clips (whole video will be split)", initialdir=default_out)
    if not out_dir:
        return
    try: _os.makedirs(out_dir, exist_ok=True)
    except: pass
    # Run in thread to not block GUI
    def _worker():
        try:
            import gizmo as _gz
            ffmpeg = _gz.find_ffmpeg()
            if not ffmpeg or not _os.path.isfile(ffmpeg):
                _mb.showerror("Auto - no ffmpeg", "ffmpeg not found (needs imageio-ffmpeg). Run install_fizgig.bat again.")
                return
            info = _gz.probe_source(ffmpeg, src)
            fps = info.get("fps") or 24.0
            duration = info.get("duration") or 0
            if duration <= 0:
                _mb.showerror("Auto - cannot read", f"Could not probe duration for {src}")
                return
            total_frames = int(round(duration * fps))
            # Auto choose: split whole video into sequential valid GRID_FRAMES chunks
            # Use largest possible to minimize clips: prefer 124, then 107 etc, covering remainder
            GRID = list(_gz.GRID_FRAMES)  # 5,22,39,56,73,90,107,124
            # Simple greedy: while remaining, pick largest GRID <= remaining, else smallest
            remaining = total_frames
            start_f = 0
            clips = []
            while remaining > 0:
                # pick largest grid <= remaining
                pick = None
                for g in reversed(GRID):
                    if g <= remaining:
                        pick = g
                        break
                if pick is None:
                    # remaining smaller than smallest (5) -> merge with last clip or pad
                    # Extend last clip by padding with duplicated frames via -frames (ffmpeg will handle)
                    # For simplicity, break and pad last clip to smallest
                    remaining = 0
                    break
                clips.append((start_f, pick))
                start_f += pick
                remaining -= pick
            if not clips:
                _mb.showerror("Auto - too short", f"Video is {total_frames} frames at {fps}fps, smaller than smallest valid {GRID[0]}.")
                return
            # For each clip, export via gizmo.build_export_command
            from gizmo import target_size as _target_size, build_export_command as _build
            src_w, src_h = info["display_width"], info["display_height"]
            sar = info.get("sar",1.0)
            # Use 0.25 MP as safe default for 8GB (user can change Target Megapixels later, clips are cut at native then resized at train time)
            # We cut at native resolution then let training resize, so width/height here are native snapped
            # For auto we just use native snapped size
            success = 0
            failed = []
            for idx, (sf, frames) in enumerate(clips):
                start_s = sf / fps
                # target size at native (snap to 32, keep aspect)
                w, h = _target_size(src_w, src_h, megapixels=10.0)  # 10MP ~ native (clamped to max_w*h)
                # ensure multiples of 32
                w, h = max(32, w // 32 * 32), max(32, h // 32 * 32)
                dst = _gz.output_name(src, out_dir, muted=False, claimed=[])
                # ensure unique name for sequential clips
                base, ext = _os.path.splitext(dst)
                if idx > 0:
                    dst = f"{base}_{idx+1:02d}{ext}"
                cmd = _build(ffmpeg, src, dst, start_s, frames, w, h, keep_every=None, with_audio=True, crop=None, sar=sar)
                # Run ffmpeg
                try:
                    p = _gz._run(cmd)
                    if p.returncode == 0 and _os.path.isfile(dst):
                        success += 1
                    else:
                        failed.append(f"clip {idx+1} {frames}f failed: {(p.stderr or b'')[:200].decode(errors='ignore')}")
                except Exception as e:
                    failed.append(str(e))
            # Also auto-handle audio: ensure 32k stereo via build_export_command already does atrim 32k
            msg = f"Auto finished: {success}/{len(clips)} clips written to {out_dir}\nEach clip is 17n+5 frames @24fps, 32k stereo, multiples of 32."
            if failed:
                msg += "\n\nFailed:\n" + "\n".join(failed[:5])
            _mb.showinfo("Auto - done", msg)
            # Optionally set image folder to out_dir
            try:
                if hasattr(gui, 'image_folder_var'):
                    gui.image_folder_var.set(out_dir)
            except: pass
        except Exception as e:
            import traceback as _tb
            _mb.showerror("Auto - error", f"{type(e).__name__}: {e}\n{_tb.format_exc()[:800]}")
    _th.Thread(target=_worker, daemon=True).start()

def _patch_optimizer_tooltips(self):
    # Add tooltips for optimizer args / scheduler specifics
    try:
        import lora_trainer_gui as gui_mod
        ToolTip = gui_mod.ToolTip
        # Optimizer args tooltip - explains per-optimizer specific args
        if "OPTIMIZER_ARGS" in self.entries:
            ent = self.entries["OPTIMIZER_ARGS"]
            # Master tooltip covering many optimizers
            tip = (
                "Extra optimizer kwargs as key=value pairs, space separated.\n\n"
                "adamw/adamw8bit: weight_decay=0.01 betas=0.9,0.999 eps=1e-8\n"
                "lion8bit: needs ~1/10 AdamW LR (sign update)\n"
                "automagic/v3: lr=1e-6 min_lr=1e-8 max_lr=1e3 beta2=0.999 eps=1e-30 clip_threshold=1.0 weight_decay=0.0 polarity_history=8 fused=True\n"
                "  v3 pools polarity_history (2-64, default 8) - no gain knob, lr*=exp(vote)\n"
                "ademamix8bit: weight_decay=0.01 betas=0.9,0.99 (slow EMA)\n"
                "paged*: same as base but pages to CPU under pressure"
            )
            ToolTip(ent, tip)
            # Also add visible selectable text below to be easy to copy-paste (user request)
            try:
                parent = ent.master
                COLORS = gui_mod.COLORS
                # Find max row to avoid overlapping existing widgets
                max_r = 0
                for ch in parent.winfo_children():
                    try:
                        inf = ch.grid_info()
                        if inf: max_r = max(max_r, int(inf.get('row', 0)))
                    except: pass
                r = max_r + 1
                frame = tk.Frame(parent, bg=COLORS.get("bg_surface","#1A1A1A"))
                frame.grid(row=r, column=0, columnspan=4, sticky=tk.EW, padx=5, pady=(4,8))
                frame.columnconfigure(0, weight=1)
                txt = tk.Text(frame, height=6, wrap=tk.WORD, bg=COLORS.get("bg_surface","#1A1A1A"), fg=COLORS.get("text_muted","#8A6B4A"), font=("Consolas", 8), relief=tk.FLAT, highlightthickness=1, highlightbackground=COLORS.get("border","#3A2410"), bd=0, padx=6, pady=4)
                txt.insert("1.0", tip)
                # Make read-only but selectable
                txt.bind("<Key>", lambda e: "break")
                txt.configure(state="normal")
                # Allow selection, prevent edit via bind
                txt.grid(row=0, column=0, sticky=tk.EW, padx=(0,4))
                frame.grid_columnconfigure(0, weight=1)
                def _copy_tip():
                    try:
                        parent.clipboard_clear()
                        parent.clipboard_append(tip)
                    except: pass
                btn = ttk.Button(frame, text="Copy", width=6, command=_copy_tip)
                btn.grid(row=0, column=1, sticky=tk.N)
                self._chaotic_opt_tip_text = txt
                self._chaotic_opt_tip_frame = frame
            except Exception as e:
                print(f"[chaotic] opt tip label failed: {e}")
        # Scheduler tooltips
        if "LR_SCHEDULER" in self.entries:
            ent = self.entries["LR_SCHEDULER"]
            ToolTip(ent, "constant (default) | constant_with_warmup | cosine | cosine_with_restarts | linear | polynomial | rex\nWarmup steps only for schedulers with warmup")
        if "LR_WARMUP_STEPS" in self.entries:
            ent = self.entries["LR_WARMUP_STEPS"]
            ToolTip(ent, "Warmup steps: linear warmup from 0 to lr. 0 = no warmup. Useful for large LR or automagic.")
        # Sample generation enable tooltip already exists, but ensure
        if hasattr(self, 'sample_enabled_check'):
            ToolTip(self.sample_enabled_check, "Uncheck to skip sample generation during training (faster, less VRAM). Chaotic default: unchecked.")
    except Exception as e:
        print(f"[chaotic] tooltip patch failed: {e}")


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
    exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
    n = 0
    try:
        for f in os.listdir(dataset_folder):
            if os.path.splitext(f)[1].lower() in exts:
                n += 1
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

    # --- Control + Target ---
    tk.Label(content, text="Control + Target (paired training — depth/pose/canny/edit)", font=("Segoe UI", 10, "bold"), fg="#FF6B00", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(2,2))
    row+=1
    tk.Label(content, text="Target = Dataset folder (Start tab). Control = condicionamento. Mesmo nome de arquivo nas duas pastas = par.", font=("Segoe UI", 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"], wraplength=680, justify=tk.LEFT).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0,4))
    row+=1
    ctrl_frame = ttk.Frame(content)
    ctrl_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
    ttk.Label(ctrl_frame, text="Control folder:").pack(side=tk.LEFT, padx=(0,4))
    self._chaotic_control_dir_var = tk.StringVar(value=getattr(self, '_chaotic_control_dir', "") or "")
    ctrl_ent = ttk.Entry(ctrl_frame, textvariable=self._chaotic_control_dir_var, width=46)
    ctrl_ent.pack(side=tk.LEFT)
    self.entries['CHAOTIC_CONTROL_DIR'] = ctrl_ent
    def _browse_ctrl():
        d = filedialog.askdirectory(title="Select CONTROL folder (paired to Dataset)")
        if d:
            self._chaotic_control_dir_var.set(d)
            self._chaotic_control_dir = d
            _update_pairing_label()
    ttk.Button(ctrl_frame, text="Browse", command=_browse_ctrl).pack(side=tk.LEFT, padx=(4,0))
    ttk.Button(ctrl_frame, text="Clear", command=lambda: (self._chaotic_control_dir_var.set(""), setattr(self,'_chaotic_control_dir',None), _update_pairing_label())).pack(side=tk.LEFT, padx=(4,0))
    row+=1
    pairing_label = tk.Label(content, text="", font=("Segoe UI", 9, "italic"), fg="#8A9BAE", bg=COLORS["bg_surface"], wraplength=680, justify=tk.LEFT)
    pairing_label.grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
    self._chaotic_pairing_label = pairing_label
    def _update_pairing_label(*_):
        try:
            target = self.image_folder_var.get() if hasattr(self,'image_folder_var') else ''
            ctrl = self._chaotic_control_dir_var.get().strip()
            self._chaotic_control_dir = ctrl if ctrl else None
            if not ctrl:
                pairing_label.config(text="No control folder → unpaired (default).", fg="#8A9BAE"); return
            if not os.path.isdir(ctrl):
                pairing_label.config(text="Control folder doesn't exist.", fg="#EF4444"); return
            if not target or not os.path.isdir(target):
                pairing_label.config(text=f"Control: {ctrl}  — set Dataset folder to check pairing.", fg="#8A9BAE"); return
            t_files = set(os.path.splitext(f)[0] for f in os.listdir(target) if os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.webp'])
            c_files = set(os.path.splitext(f)[0] for f in os.listdir(ctrl) if os.path.splitext(f)[1].lower() in ['.jpg','.jpeg','.png','.webp'])
            matched = len(t_files & c_files); total=len(t_files)
            if matched==0: pairing_label.config(text=f"⚠ 0/{total} paired — need same basename!", fg="#F59E0B")
            elif matched<total: pairing_label.config(text=f"⚠ {matched}/{total} paired — {total-matched} missing control will ERROR", fg="#F59E0B")
            else: pairing_label.config(text=f"✓ {matched}/{total} paired — control conditioning active (Klein Edit / Krea2 / H3)", fg="#10B981")
        except Exception as e: pairing_label.config(text=f"Pairing check failed: {e}", fg="#F59E0B")
    try:
        self.image_folder_var.trace_add("write", lambda *_: _update_pairing_label())
        self._chaotic_control_dir_var.trace_add("write", lambda *_: _update_pairing_label())
    except: pass
    _update_pairing_label()
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
    row+=1
    # Scheduler
    ttk.Label(content, text="LR Scheduler:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    sched_combo = ttk.Combobox(content, values=["constant","constant_with_warmup","cosine","cosine_with_restarts","linear","polynomial","rex"], width=22)
    try: real_sched = self.entries.get('LR_SCHEDULER').get() or "constant"
    except: real_sched="constant"
    sched_combo.set(real_sched)
    sched_combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_SCHEDULER'] = sched_combo
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

    # Expand original collapsed sections automatically under chaotic
    try:
        for key in ("optimizer","scheduler","memory"):
            sec2 = getattr(self, 'collapsible_sections', {}).get(key)
            if sec2 and not sec2.expanded:
                sec2.toggle()
    except: pass

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
        ttk.Separator(prompt_card, orient="horizontal").grid(row=r, column=0, columnspan=3, sticky="ew", padx=5, pady=10)
        r+=1
        tk.Label(prompt_card, text="Chaotic — Krea2 Ostris Edit (test edit in sampling)", font=("Segoe UI", 10, "bold"), fg="#FF6B00", bg=_C["bg_surface"]).grid(row=r, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(2,2))
        r+=1
        tk.Label(prompt_card, text="Qwen3-VL vision (384px) + VAE latents (1MP t=0) + kv_cache. Use to test if edit LoRA works live - refs become tokens at t=0 (index_timestep_zero).", font=("Segoe UI", 8), fg=_C["text_muted"], bg=_C["bg_surface"], wraplength=640, justify=tk.LEFT).grid(row=r, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(0,4))
        r+=1
        tk.Label(prompt_card, text="Reference images (up to 3) — Picture N + VAE:", font=("Segoe UI", 9, "bold"), fg="#FF6B00", bg=_C["bg_surface"]).grid(row=r, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4,2))
        r+=1
        self._chaotic_krea2_images = []
        self._chaotic_kv_cache = tk.BooleanVar(value=False)
        for idx in range(3):
            fr = ttk.Frame(prompt_card)
            fr.grid(row=r, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
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
            r+=1
        kv_fr = ttk.Frame(prompt_card)
        kv_fr.grid(row=r, column=0, columnspan=3, sticky=tk.W, padx=5, pady=4)
        ttk.Checkbutton(kv_fr, text="kv_cache (LoRA trained with kv_cache=true)", variable=self._chaotic_kv_cache).pack(side=tk.LEFT)
        self.entries['CHAOTIC_KV_CACHE'] = self._chaotic_kv_cache
        tk.Label(kv_fr, text="off=normal edit, on=cached KV (faster, only if LoRA trained with it)", font=("Segoe UI", 8), fg=_C["text_muted"], bg=_C["bg_surface"]).pack(side=tk.LEFT, padx=(8,0))
        r+=1
        tk.Label(prompt_card, text="How to test: select 1-3 refs, keep VAE connected (for VAE latent), prompt with trigger, Generate Sample. Refs are encoded via Qwen (semantic) + VAE (t=0).", font=("Segoe UI", 8, "italic"), fg=_C["text_explain"], bg=_C["bg_surface"], wraplength=640, justify=tk.LEFT).grid(row=r, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(4,0))
        print("[chaotic] Krea2 edit injected into Prompt & Dimensions card")
    except Exception as e:
        print(f"[chaotic] samples card failed: {e}")
        import traceback; traceback.print_exc()

def _on_steps_toggle(self):
    on = bool(self._chaotic_steps_mode.get())
    try:
        if on: self._chaotic_steps_frame.grid()
        else: self._chaotic_steps_frame.grid_remove()
    except: pass
    try:
        import json as _json
        pref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "presets", "chaotic.json")
        pref_path = os.path.abspath(pref_path)
        os.makedirs(os.path.dirname(pref_path), exist_ok=True)
        with open(pref_path, 'w', encoding='utf-8') as f:
            _json.dump({"steps_mode": on}, f)
    except: pass


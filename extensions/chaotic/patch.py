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

def apply_chaotic_patches(GUIClass):
    orig_init = GUIClass.__init__
    def chaotic_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
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

def _inject_chaotic_ui(self):
    try:
        self.master.title(self.master.title() + " [CHAOTIC]")
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
    tk.Label(content, text="Training length:", font=("Segoe UI", 10, "bold"), fg="#FF7EDB", bg=COLORS["bg_surface"]).grid(row=row, column=0, sticky=tk.W, padx=5, pady=(6,2))
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
    tk.Label(content, text="Control + Target (paired training — depth/pose/canny/edit)", font=("Segoe UI", 10, "bold"), fg="#FF7EDB", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(2,2))
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

    # --- Optimizer / Scheduler / Saves / VRAM ---
    tk.Label(content, text="Optimizer & Scheduler (AI-Toolkit: optimizer, lr_scheduler)", font=("Segoe UI", 10, "bold"), fg="#FF7EDB", bg=COLORS["bg_surface"]).grid(row=row, column=0, columnspan=3, sticky=tk.W, padx=5, pady=(2,4))
    row+=1
    # Optimizer
    ttk.Label(content, text="Optimizer:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
    opt_combo = ttk.Combobox(content, values=["adamw","adamw8bit","lion","prodigy","adafactor","adamw8bit + prodigy"], width=22)
    # sync with real entry if exists
    real_opt = ""
    try: real_opt = self.entries.get('OPTIMIZER_TYPE').get() or "adamw8bit"
    except: real_opt="adamw8bit"
    opt_combo.set(real_opt)
    opt_combo.grid(row=row, column=1, sticky=tk.W, padx=5, pady=3)
    self.entries['CHAOTIC_OPTIMIZER'] = opt_combo
    tk.Label(content, text="adamw8bit = 8GB safe", font=("Segoe UI", 8), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(row=row, column=2, sticky=tk.W, padx=5)
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

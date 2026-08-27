"""Fizgig Chaotic - Luisa Caotica Edition launcher."""
import os, sys, subprocess, traceback, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHONW = os.path.join(HERE, "venv", "Scripts", "pythonw.exe")
VENV_PYTHON = os.path.join(HERE, "venv", "Scripts", "python.exe")
LOG = os.path.join(HERE, "launch_chaotic_error.log")

def _report(title, message, detail=""):
    logged=False
    try:
        with open(LOG,"w",encoding="utf-8") as f:
            f.write(f"{title}\n\n{message}\n\n{detail}\n\npython: {sys.executable}\nversion: {sys.version}\n")
        logged=True
    except: pass
    body = message + (f"\n\nDetails saved to:\n{LOG}" if logged else "")
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, body, title, 0x10010)
    except:
        if getattr(sys,"stderr",None):
            sys.stderr.write(f"{title}\n{body}\n{detail}\n")
    sys.exit(1)

# re-launch in venv if not already inside it (keep console for python.exe)
try:
    venv_dir = os.path.join(HERE, "venv")
    is_in_venv = False
    try:
        is_in_venv = os.path.commonpath([os.path.abspath(sys.executable), os.path.abspath(venv_dir)]) == os.path.abspath(venv_dir)
    except: 
        is_in_venv = "venv" in sys.executable.lower() and "fizgig" in sys.executable.lower()
    if not is_in_venv and os.path.exists(VENV_PYTHONW):
        is_console = sys.executable.lower().endswith("python.exe")
        target = VENV_PYTHON if is_console else VENV_PYTHONW
        try:
            subprocess.Popen([target, os.path.abspath(__file__)])
        except Exception as exc:
            _report("Fizgig Chaotic could not start", f"The bundled Python failed to launch:\n{target}\n\nRe-run install_fizgig.bat.", f"{type(exc).__name__}: {exc}")
        sys.exit(0)
except: pass

try:
    import tkinter as tk
except Exception as exc:
    _report("Fizgig Chaotic - Python is missing Tkinter","This Python was installed without Tkinter.\nReinstall Python from python.org and tick 'tcl/tk and IDLE' then run install_fizgig.bat again.", f"{type(exc).__name__}: {exc}")

try:
    if os.path.isfile(LOG):
        os.remove(LOG)
except: pass

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))
# MUST patch before Tk mainloop - import gui, patch class, then instantiate ourselves (NOT runpy)
try:
    spec = importlib.util.spec_from_file_location("chaotic_patch", os.path.join(HERE, "extensions", "chaotic", "patch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import lora_trainer_gui as gui_mod
    target = getattr(gui_mod, "LoRATrainerGUI", None)
    if target is None:
        for v in vars(gui_mod).values():
            if isinstance(v, type) and hasattr(v, "start_training"):
                target = v; break
    if target:
        mod.apply_chaotic_patches(target)
        print(f"[chaotic] patched {target.__name__}")
    else:
        print("[chaotic] WARNING: no GUI class found to patch")
        target = gui_mod.LoRATrainerGUI
except Exception as exc:
    _report("Fizgig Chaotic - patch failed", f"Chaotic extension failed to load:\n\n{type(exc).__name__}: {exc}\n\nVanilla Fizgig will still work via run_fizgig.bat / launch.pyw.", traceback.format_exc())
    import lora_trainer_gui as gui_mod
    target = gui_mod.LoRATrainerGUI

# Now run the GUI ourselves - this uses the PATCHED class
try:
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('fizgig.lora.studio.chaotic')
    except: pass
    root = tk.Tk()
    # proving banner in window title
    try:
        root.title(root.title() + " - CHAOTIC (Luisa Caotica)")
    except: pass
    gui = target(root)
    # also force title after GUI sets it
    try:
        current = root.title()
        if "CHAOTIC" not in current:
            root.title(current + " - CHAOTIC")
    except: pass
    try:
        gui._check_for_paused_state_on_startup()
    except: pass
    try:
        root.after(2500, gui._start_update_check)
    except: pass
    print("[chaotic] GUI started - look for CHAOTIC in title and Training tab")
    root.mainloop()
except (SystemExit, KeyboardInterrupt):
    raise
except BaseException as exc:
    _report("Fizgig Chaotic could not start", f"Fizgig hit an error while starting:\n\n{type(exc).__name__}: {exc}\n\nAttach the log.", traceback.format_exc())

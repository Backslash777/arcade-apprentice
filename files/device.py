"""Portable AI Arcade - pick the fastest thing that ACTUALLY works for the brain:
  NVIDIA (CUDA)  |  AMD Radeon (ROCm build - also appears as 'cuda' in PyTorch)  |  AMD/Intel via DirectML  |  CPU.
Every candidate must pass a tiny real training step before it's trusted. Signed: Claude, 9 Oct 2026."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

FILES = Path(__file__).resolve().parent
STATE = FILES / "gpu-setup.json"      # written by setup: which backend was installed for which graphics card


def graphics_cards():
    """Names of the graphics cards Windows can see (any brand)."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_VideoController).Name"], capture_output=True, text=True,
                             timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return [l.strip() for l in out.splitlines() if l.strip() and "Basic Display" not in l and "Virtual" not in l]
    except Exception:
        return []


def vendor(names):
    s = " ".join(names).lower()
    if "nvidia" in s:
        return "nvidia"
    if "amd" in s or "radeon" in s:
        return "amd"
    if "intel" in s and ("arc" in s or "iris" in s):
        return "intel"
    return "none"


def smoke(dev):
    """A miniature PPO step (conv, pool, linear, softmax, gather, backward, Adam) on this device."""
    import torch
    import torch.nn as nn
    net = nn.Sequential(nn.Conv2d(4, 8, 3, padding=1), nn.MaxPool2d(3, 2, 1), nn.ReLU(), nn.Flatten(),
                        nn.Linear(8 * 42 * 42, 6)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    x = torch.randint(0, 255, (8, 4, 84, 84), dtype=torch.uint8).to(dev)
    logits = net(x.float() / 255.0)
    a = torch.randint(0, 6, (8,)).to(dev)
    lp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
    loss = -(lp.exp() * lp).mean() + logits.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()
    v = float(loss.detach().cpu())
    if v != v:
        raise RuntimeError("NaN from the graphics card")


def pick(prefer_gpu=True):
    """-> (torch device, human name, backend in {'cuda','rocm','directml','cpu'}, problem text or None)"""
    import torch
    problem = None
    if prefer_gpu and torch.cuda.is_available():
        try:
            dev = torch.device("cuda")
            smoke(dev)
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            return dev, torch.cuda.get_device_name(0), backend, None
        except Exception as e:
            problem = f"{torch.cuda.get_device_name(0)} found but PyTorch can't use it ({str(e).splitlines()[0]})"
    if prefer_gpu:
        try:
            import torch_directml
            dev = torch_directml.device()
            smoke(dev)
            return dev, torch_directml.device_name(0), "directml", None
        except ImportError:
            pass
        except Exception as e:
            problem = f"DirectML graphics route failed ({str(e).splitlines()[0]})"
    import platform
    return torch.device("cpu"), platform.processor() or "CPU", "cpu", problem


def needs_repair():
    """True if this PC has a graphics card we haven't set up for (new PC, or the folder was copied from another one),
    or the backend that was set up has stopped working."""
    cards = graphics_cards()
    v = vendor(cards)
    try:
        st = json.loads(STATE.read_text())
    except Exception:
        st = {}
    try:
        dev, name, backend, problem = pick()
    except Exception as e:      # PyTorch won't even load
        print("  PyTorch won't load:", e)
        return True
    if backend != "cpu":
        return False                                   # a graphics card works - nothing to fix
    if v == "none":
        return False                                   # no usable card: CPU is the right answer
    if st.get("cards") == cards and st.get("tried"):
        return False                                   # already tried everything for this exact card; CPU it is
    if problem:
        print("  " + problem)
    print(f"  Graphics card not in use yet: {', '.join(cards)}")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--record":       # setup calls this at the end
        dev, name, backend, problem = pick()
        cards = graphics_cards()
        STATE.write_text(json.dumps({"cards": cards, "vendor": vendor(cards), "backend": backend, "device": name,
                                     "problem": problem, "tried": True}, indent=1))
        print(f"  Brain will run on: {name} ({backend})" + (f" - note: {problem}" if problem else ""))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--vendor":
        print(vendor(graphics_cards()))
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "--test":         # exit 0 if the GPU route works
        sys.exit(0 if pick()[2] != "cpu" else 1)
    sys.exit(3 if needs_repair() else 0)

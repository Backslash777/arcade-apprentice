"""Kept for older launchers: same as `device.py` (exit 3 = graphics card needs setting up). Signed: Claude."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import device
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        sys.exit(0 if device.pick()[2] != "cpu" else 1)
    sys.exit(3 if device.needs_repair() else 0)

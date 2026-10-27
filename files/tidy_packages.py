"""Arcade Apprentice - clear out graphics-card libraries that a failed attempt left behind.

Setup tries the fastest way to use the graphics card first (NVIDIA's CUDA, AMD's ROCm, then DirectML for AMD and
Intel cards) and falls back to the processor. An attempt that doesn't work can leave gigabytes behind - AMD's ROCm
libraries alone are several GB - so once setup has settled on what works, this removes every graphics-card library
the working PyTorch doesn't use. It never touches anything else.

    python tidy_packages.py          remove them
    python tidy_packages.py --dry    just say what would go
"""
import re
import subprocess
import sys
from importlib import metadata

GPU_LIBS = re.compile(r"^(nvidia-|rocm|torch-directml|pytorch-triton|triton)")   # the only packages it may remove


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    have = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            have[norm(name)] = dist
    return have


def needed(roots, have):
    """The given packages and everything they depend on, all the way down (only what applies to this PC)."""
    from pip._vendor.packaging.requirements import Requirement
    keep, seen = set(), set()
    todo = [(norm(r), frozenset()) for r in roots]
    while todo:
        name, extras = todo.pop()
        if (name, extras) in seen or name not in have:
            continue
        seen.add((name, extras))
        keep.add(name)
        for line in have[name].requires or []:
            try:
                req = Requirement(line)
            except Exception:
                continue
            if req.marker is not None and not any(req.marker.evaluate({"extra": e}) for e in ("", *extras)):
                continue
            todo.append((norm(req.name), frozenset(norm(x) for x in req.extras)))
    return keep


def size_of(dist):
    total = 0
    for f in dist.files or []:
        try:
            total += f.locate().stat().st_size
        except OSError:
            pass
    return total


def main():
    have = installed()
    if "torch" not in have:
        print("  PyTorch isn't installed, so the graphics libraries were left as they are.")
        return 0
    # Which way the working PyTorch drives the card: NVIDIA and AMD ROCm builds say so in their version number;
    # otherwise it's DirectML (if setup kept torch-directml, its test passed) or just the processor.
    version = have["torch"].version
    family = "rocm" if "+rocm" in version else "nvidia-" if "+cu" in version else None
    roots = ["torch"] + (["torch-directml"] if family is None and "torch-directml" in have else [])
    keep = needed(roots, have)
    if family:      # belt and braces: that build's own libraries stay, even if it doesn't list them
        keep |= {n for n in have if n.startswith(family)}
    spare = sorted(n for n in have if GPU_LIBS.match(n) and n not in keep)
    if not spare:
        print("  Nothing left over from other graphics cards to tidy up.")
        return 0
    gb = sum(size_of(have[n]) for n in spare) / 2 ** 30
    print(f"  Removing graphics libraries this PC doesn't use ({gb:.1f} GB): {', '.join(spare)}")
    if "--dry" in sys.argv:
        return 0
    return subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", *spare]).returncode


if __name__ == "__main__":
    sys.exit(main())

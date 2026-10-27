"""Portable AI Arcade - keep the learning off the cores the arcade cabinets are using (Alfred, 21 Oct 2026).

On processors with two kinds of core (Intel 12th-14th gen and later: fast 'performance' cores + slower 'efficiency'
cores), hundreds of cabinet processes crowd the learning thread off the fast cores. The lab (_dev/gpu_lab) measured
learning at ~5.9 s a batch on its own but ~8.1 s while 256 cabinets played. Giving the trainer a few performance cores
of its own, and the cabinets everything else, lets both work at once.
On any other processor (one kind of core, or not Windows) this does nothing. PAA_SPLIT=0 switches it off."""
from __future__ import annotations

import ctypes
import os
import sys


def core_classes():
    """Efficiency class of every logical CPU, from Windows' CPU sets (higher = faster core). None if not available."""
    if sys.platform != "win32":
        return None
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        n = ctypes.c_ulong(0)
        k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        if n.value == 0:
            return None
        buf = (ctypes.c_ubyte * n.value)()
        if not k32.GetSystemCpuSetInformation(buf, n.value, ctypes.byref(n), None, 0):
            return None
        raw, out, off = bytes(buf), {}, 0
        while off + 19 <= n.value:
            size = int.from_bytes(raw[off:off + 4], "little")
            if size <= 0:
                break
            out[raw[off + 14]] = raw[off + 18]            # LogicalProcessorIndex -> EfficiencyClass
            off += size
        return [out[i] for i in sorted(out)] if out else None
    except Exception:
        return None


def plan(trainer_cpus=None):
    """(cpus for the trainer, cpus for the cabinets) or None. The trainer gets the first `trainer_cpus` logical CPUs of
    the fastest class; the cabinets get every other CPU. No number = half of the fast CPUs (up to 8). A number from
    TUNE FOR THIS PC (benchmark.json trainer_cpus / PAA_SPLIT_CPUS) is honoured up to all but two of the fast CPUs
    (ALFRED 21 Oct 2026 review: the old half-cap silently turned the tune's '12 fast cores' trial into 8)."""
    if os.environ.get("PAA_SPLIT", "1") == "0":
        return None
    if os.environ.get("PAA_SPLIT_CPUS"):                 # TUNE FOR THIS PC tries sizes this way
        trainer_cpus = int(os.environ["PAA_SPLIT_CPUS"])
    cls = core_classes()
    if not cls or len(set(cls)) < 2:
        return None                                       # one kind of core: nothing to gain
    fast = [i for i, c in enumerate(cls) if c == max(cls)]
    if trainer_cpus:
        k = max(2, min(int(trainer_cpus), len(fast) - 2))   # asked for: up to all but two of the fast CPUs
    else:
        k = max(2, min(8, len(fast) // 2))                  # default: half of them
    mine = fast[:k]
    rest = [i for i in range(len(cls)) if i not in mine]
    if len(rest) < 4:
        return None
    return mine, rest


def apply(trainer_pid, cabinet_pids, split):
    """Pin the trainer and its cabinets. Returns a one-line note for the console."""
    import psutil
    mine, rest = split
    psutil.Process(trainer_pid).cpu_affinity(mine)
    done = 0
    for pid in cabinet_pids:
        try:
            psutil.Process(pid).cpu_affinity(rest)
            done += 1
        except Exception:
            pass
    return (f"Cores: learning on {len(mine)} of the fast cores, the {done} arcade machines on the other "
            f"{len(rest)} (PAA_SPLIT=0 turns this off).")


if __name__ == "__main__":
    print("classes:", core_classes())
    print("plan(8):", plan(8))

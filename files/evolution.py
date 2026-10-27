"""Portable AI Arcade - EVOLUTION: a family of brains that compete, copy the winners and mutate.

Jonny's evolution system (Machine Learning\\2600\\Evolution\\_tools\\pbt_master.py - Population-Based Training,
Jaderberg et al., DeepMind 2017), rebuilt for the portable kit so it works with every game the kit can load.

A population of brains takes turns on the graphics card (one at a time). Each turn ("slice") a brain practises for a
while with its OWN settings - its genes: learning speed, randomness, how far ahead it plans, and how big a step each
lesson may take. When every brain has had a turn:
  SELECTION  - brains are ranked on their real game scores from that turn
  EXPLOIT    - the bottom third copy the brain of a random top-third brain
  EXPLORE    - the copied genes then MUTATE (each x0.8 or x1.25, with a 20% chance of a fresh random value)
The champion is copied to the AI's own run folder, so Watch, the graphs and the report card show the best brain.
Each turn runs the kit's normal trainer.py on a member folder (files\\pop\\m0, m1, ...), so everything the trainer
can do (arcade, Go-Explore, explorer bonus, learn from me) works inside evolution too.

Run: python evolution.py <AI folder>      STOP: the page's Stop button (writes files/run/STOP).
Signed: Claude, 10 Oct 2026 - from Jonny's pbt.py.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embeddable Python ignores the script folder

import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

FILES = Path(__file__).resolve().parent
GENES = {"learning_rate": [1e-4, 8e-4], "curiosity": [0.003, 0.03], "gamma": [0.97, 0.997], "clip": [0.05, 0.25]}
LOG_FIELDS = ["step", "score_mean100", "best_game", "entropy", "sps", "generation", "member", "fitness"]


def rand_gene(name):
    lo, hi = GENES[name]
    if name == "gamma":                       # sample the planning horizon 1/(1-gamma) log-uniformly
        h = 10 ** random.uniform(math.log10(1 / (1 - lo)), math.log10(1 / (1 - hi)))
        return round(1 - 1 / h, 5)
    return float(f"{10 ** random.uniform(math.log10(lo), math.log10(hi)):.3g}")


def mutate(genes):
    out = {}
    for k, v in genes.items():
        lo, hi = GENES[k]
        if random.random() < 0.2:
            out[k] = rand_gene(k)
            continue
        f = random.choice([0.8, 1.25])
        if k == "gamma":
            out[k] = round(min(hi, max(lo, 1 - 1 / (1 / (1 - v) * f))), 5)
        else:
            out[k] = float(f"{min(hi, max(lo, v * f)):.3g}")
    return out


def read_jsonl(p):
    try:
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    except (FileNotFoundError, ValueError):
        return []


class Evolution:
    def __init__(self, proj):
        self.proj = Path(proj).resolve()
        self.run = self.proj / "files" / "run"
        self.pop = self.proj / "files" / "pop"
        self.state_file = self.proj / "files" / "evolution_state.json"
        self.run.mkdir(parents=True, exist_ok=True)
        self.pop.mkdir(parents=True, exist_ok=True)
        self.cfg = self.read_cfg()

    def read_cfg(self):
        return json.loads((self.proj / "project.json").read_text())

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(self.run / "console.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------------ state
    def load_state(self):
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        n = int(self.cfg.get("evo_population", 4))
        random.seed(int(time.time()))
        members = [{"id": f"m{i}", "genes": {k: rand_gene(k) for k in GENES}, "step": 0, "fitness": None,
                    "lineage": [f"m{i}"], "history": []} for i in range(n)]
        # brain m0 starts with the AI's own settings (the dials on the page)
        members[0]["genes"] = {"learning_rate": float(self.cfg.get("learning_rate", 2.5e-4)),
                               "curiosity": float(self.cfg.get("curiosity", 0.01)),
                               "gamma": float(self.cfg.get("gamma", 0.99)), "clip": 0.1}
        if self.cfg.get("evo_start_near_dials"):     # ALFRED 19 Oct 2026: start the family near the dials, not at random
            for m in members[1:]:
                m["genes"] = mutate(members[0]["genes"])
        # if the AI has already been practising, every brain in the family starts from what it learned
        seed = self.run / "latest.pt"
        start = 0
        if seed.exists():
            try:
                import torch
                start = int(torch.load(seed, map_location="cpu", weights_only=False).get("global_step", 0))
            except Exception:
                start = 0
            for m in members:
                d = self.pop / m["id"] / "files" / "run"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(seed, d / "latest.pt")
                m["step"] = start
        return {"generation": 0, "next_member": 0, "members": members, "total_steps": 0, "best": None,
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "seeded_from_step": start}

    def save_state(self, st):
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, self.state_file)

    def status(self, st, state="training", extra=None, live=None):
        best = (st["best"] or {}).get("fitness")
        d = {"state": state, "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "step": st["total_steps"],
             "best_mean100": best if best is not None else -1e9, "pid": os.getpid(),
             "evo": {"generation": st["generation"], "population": len(st["members"]),
                     "member": (extra or {}).get("member"), "genes": (extra or {}).get("genes")}}
        if live:
            d.update({k: live.get(k) for k in ("sps", "score_mean100", "device") if k in live})
            d["step"] = st["total_steps"] + max(0, int(live.get("step", 0)) - (extra or {}).get("from_step", 0))
        try:
            (self.run / "status.json").write_text(json.dumps(d))
        except OSError:
            pass

    # ------------------------------------------------------------------------------------------------ one turn
    def member_cfg(self, m, target):
        cfg = dict(self.read_cfg())
        cfg.update(m["genes"])
        cfg.update({"total_steps": target, "length": "slice", "anneal": False, "game_name":
                    f"{cfg.get('game_name', '?')} - brain {m['id']}",
                    "demos_from": str(self.proj)})
        if cfg.get("platform") == "arcade" and cfg.get("rom_file"):
            cfg["rom_path"] = str(self.proj.parent.parent / "ROMs - put your games here" / cfg["rom_file"])
        return cfg

    def run_slice(self, st, m, slice_n):
        mdir = self.pop / m["id"]
        mrun = mdir / "files" / "run"
        mrun.mkdir(parents=True, exist_ok=True)
        (mrun / "STOP").unlink(missing_ok=True)
        target = m["step"] + slice_n
        (mdir / "project.json").write_text(json.dumps(self.member_cfg(m, target), indent=1))
        n_before = len(read_jsonl(mrun / "games.jsonl"))
        g = m["genes"]
        self.say(f"GENERATION {st['generation']} | brain {m['id']} practising {m['step']/1e6:.1f}M -> {target/1e6:.1f}M moves | "
                 f"learning speed {g['learning_rate']:.2g}, randomness {g['curiosity']:.2g}, plans ahead {g['gamma']}, "
                 f"step size {g['clip']:.2g}")
        extra = {"member": m["id"], "genes": g, "from_step": m["step"]}
        p = subprocess.Popen([sys.executable, "-u", str(FILES / "trainer.py"), str(mdir)], cwd=str(FILES))
        stop = False
        while p.poll() is None:
            if (self.run / "STOP").exists() and not stop:
                stop = True
                (mrun / "STOP").write_text("stop")
                self.say("STOP pressed - this brain saves, then the family stops.")
            try:
                live = json.loads((mrun / "status.json").read_text())
            except Exception:
                live = None
            self.status(st, "stopping" if stop else "training", extra, live)
            time.sleep(3)
        try:
            new_step = int(json.loads((mrun / "status.json").read_text()).get("step", m["step"]))
        except Exception:
            new_step = m["step"]
        games = [x for x in read_jsonl(mrun / "games.jsonl")[n_before:] if not x.get("restart")]
        scores = [x["score"] for x in games]
        fit = sum(scores[-100:]) / len(scores[-100:]) if scores else None
        ent = None
        try:
            with open(mrun / "train_log.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            ent = float(rows[-1].get("entropy") or "nan") if rows else None
            sps = float(rows[-1].get("sps") or 0) if rows else 0
        except Exception:
            sps = 0
        return new_step, fit, games, stop, p.returncode, ent, sps

    # ------------------------------------------------------------------------------------------------ graphs
    def log_row(self, row):
        f = self.run / "train_log.csv"
        if f.exists():
            with open(f, newline="") as fh:
                head = next(csv.reader(fh), [])
            if head != LOG_FIELDS:                 # practice-mode log from before: keep it, start a fresh one
                os.replace(f, self.run / f"train_log_practice_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        new = not f.exists()
        with open(f, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k) for k in LOG_FIELDS})

    # ------------------------------------------------------------------------------------------------ main loop
    def main(self):
        (self.run / "STOP").unlink(missing_ok=True)
        st = self.load_state()
        self.save_state(st)
        slice_n = int(self.cfg.get("evo_slice") or (500_000 if self.cfg.get("platform") == "arcade" else 3_000_000))   # ALFRED 20 Oct: Atari turns 1M -> 3M moves (start-up ~1 min a turn)
        self.say(f"EVOLUTION | {self.cfg['game_name']} | a family of {len(st['members'])} brains | each practises "
                 f"{slice_n/1e6:g}M moves per turn | generation {st['generation']}"
                 + (f" | started from the practised brain at {st['seeded_from_step']/1e6:.1f}M moves"
                    if st.get("seeded_from_step") else ""))
        while True:
            gen = st["generation"]
            for i in range(st["next_member"], len(st["members"])):
                m = st["members"][i]
                t0 = time.time()
                for _try in range(6):
                    new_step, fit, games, stopped, rc, ent, sps = self.run_slice(st, m, slice_n)
                    froze = self.pop / m["id"] / "files" / "run" / "FROZE"
                    if not froze.exists():
                        break
                    froze.unlink(missing_ok=True)
                    self.say(f"Brain {m['id']} froze - starting its turn again from its last save.")
                    time.sleep(10)
                tries = 0
                while rc not in (0, None) and not stopped and tries < 3 and not (self.run / "STOP").exists():
                    # ALFRED 19 Oct 2026: a brain's start-up sometimes dies inside NVIDIA's runtime compiler
                    # (nvrtc64_120_0.dll, access violation) - a fresh start nearly always gets past it
                    tries += 1
                    self.say(f"Brain {m['id']}'s practice crashed - trying its turn again in 20 seconds ({tries} of 3).")
                    time.sleep(20)
                    new_step, fit, games, stopped, rc, ent, sps = self.run_slice(st, m, slice_n)
                if rc not in (0, None) and not stopped:
                    self.say(f"Brain {m['id']}'s practice crashed (see its window/console above). Stopping the family.")
                    self.status(st, "crashed", extra={"member": m["id"]})
                    return
                st["total_steps"] += max(0, new_step - m["step"])
                done_slice = new_step >= m["step"] + slice_n
                m["step"] = new_step
                if stopped and not done_slice:
                    self.save_state(st)
                    self.status(st, "stopped")
                    self.say("Stopped and saved. Press Train to carry on - the same brain picks up where it left off.")
                    return
                m["fitness"] = fit
                m["history"].append({"gen": gen, "fitness": fit, "step": new_step, "genes": dict(m["genes"])})
                with open(self.run / "games.jsonl", "a") as f:
                    for gm in games:
                        f.write(json.dumps({**gm, "step": st["total_steps"], "member": m["id"]}) + "\n")
                self.log_row({"step": st["total_steps"], "score_mean100": fit, "fitness": fit,
                              "best_game": max([x["score"] for x in games], default=None), "entropy": ent,
                              "sps": sps, "generation": gen, "member": m["id"]})
                self.say(f"   brain {m['id']} averaged {fit if fit is None else round(fit, 1)} over its last "
                         f"{min(100, len(games))} games ({(time.time() - t0)/60:.1f} min)")
                st["next_member"] = i + 1
                self.save_state(st)
                if stopped:
                    self.status(st, "stopped")
                    self.say("Stopped and saved.")
                    return
            self.select(st, gen)
            st["generation"] += 1
            st["next_member"] = 0
            self.save_state(st)

    def select(self, st, gen):
        ranked = sorted([m for m in st["members"] if m["fitness"] is not None], key=lambda m: m["fitness"], reverse=True)
        q = max(1, int(round(len(ranked) / 3)))
        top, bottom = ranked[:q], (ranked[-q:] if len(ranked) > q else [])
        events = []
        if ranked:
            b = ranked[0]
            if not st["best"] or b["fitness"] > st["best"]["fitness"]:
                st["best"] = {"fitness": b["fitness"], "member": b["id"], "gen": gen, "genes": b["genes"], "step": b["step"]}
            src = self.pop / b["id"] / "files" / "run" / "latest.pt"
            if src.exists():                  # the champion is what Watch and the report card show
                shutil.copyfile(src, self.run / "latest.pt.tmp")
                os.replace(self.run / "latest.pt.tmp", self.run / "latest.pt")
                shutil.copyfile(src, self.run / "best.pt")
        for loser in bottom:
            if loser in top:
                continue
            parent = random.choice(top)
            s = self.pop / parent["id"] / "files" / "run" / "latest.pt"
            d = self.pop / loser["id"] / "files" / "run"
            d.mkdir(parents=True, exist_ok=True)
            if s.exists():
                shutil.copyfile(s, d / "latest.pt")
            old = dict(loser["genes"])
            loser["genes"] = mutate(parent["genes"])
            loser["step"] = parent["step"]
            loser["lineage"] = parent["lineage"][-5:] + [loser["id"]]
            events.append({"replaced": loser["id"], "by": parent["id"], "fitness_was": loser["fitness"],
                           "parent_fitness": parent["fitness"], "old_genes": old, "new_genes": loser["genes"]})
            self.say(f"   EVOLVE: brain {loser['id']} ({loser['fitness'] if loser['fitness'] is None else round(loser['fitness'])}) "
                     f"takes brain {parent['id']}'s brain ({round(parent['fitness'])}) and mutates its settings")
        with open(self.proj / "files" / "evolution_log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "generation": gen,
                                "ranking": [{"id": m["id"], "fitness": m["fitness"], "genes": m["genes"]} for m in ranked],
                                "events": events, "total_steps": st["total_steps"]}) + "\n")
        self.say(f"GENERATION {gen} done | best this round {round(ranked[0]['fitness']) if ranked else '-'} "
                 f"(brain {ranked[0]['id'] if ranked else '-'}) | all-time best "
                 f"{round(st['best']['fitness']) if st['best'] else '-'}")


if __name__ == "__main__":
    try:
        Evolution(sys.argv[1]).main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        try:
            import diagnose
            diagnose.log_error(f"evolution {Path(sys.argv[1]).name}", tb)
        except Exception:
            pass
        try:
            run = Path(sys.argv[1]) / "files" / "run"
            (run / "status.json").write_text(json.dumps({"state": "crashed", "error": tb.strip().splitlines()[-1]}))
        except Exception:
            pass
        input("Evolution crashed - press Enter to close this window.")

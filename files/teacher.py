"""Portable AI Arcade - TEACHER BRAINS: one AI plays, another learns by copying it.

python teacher.py <teacher AI folder> <pupil AI folder> [moves]

The teacher's newest brain plays its game (one cabinet / one Atari, on the CPU - no graphics card needed, so it can
run while something else trains), and every move it makes is saved in the pupil's files\\demos folder exactly like
"Learn from me" saves yours. Next time the pupil trains, it copies the teacher first, then learns to beat it.
Both AIs must play the same game. The teacher can be a practice brain or an evolution champion - or the other way.
Signed: Claude, 10 Oct 2026.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.distributions.categorical import Categorical

import demos
from trainer import Agent


def load_brain(proj):
    for name in ("latest.pt", "best.pt"):
        p = Path(proj) / "files" / "run" / name
        if p.exists():
            st = torch.load(p, map_location="cpu", weights_only=False)
            a = Agent(st["n_actions"], st.get("width", 1))
            a.load_state_dict(st["agent"], strict=False)
            a.eval()
            return a, st["n_actions"], st.get("global_step", 0)
    raise SystemExit("The teacher hasn't learned anything yet - train it first.")


def pick(agent, stack, greedy):
    with torch.no_grad():
        logits, _ = agent(torch.from_numpy(stack[None]))
    return int(logits.argmax(-1)) if greedy else int(Categorical(logits=logits).sample())


def teach(src, dst, moves=20000, greedy_share=0.5):
    src, dst = Path(src), Path(dst)
    cs, cd = (json.loads((p / "project.json").read_text()) for p in (src, dst))
    if cs["game_id"] != cd["game_id"]:
        raise SystemExit(f"Different games: {cs['game_name']} can't teach {cd['game_name']}.")
    if cs.get("platform") == "arcade":   # the game's buttons changed since these brains last played? carry them over
        try:
            import controls
            for who in (src, dst):
                controls.ensure_fits(who, print)
        except Exception as e:
            print("Couldn't carry the brains over to the game's new buttons:", e)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 4))
    agent, n_act, at = load_brain(src)
    frames, actions, starts = [], [], []
    played, scores, t0 = 0, [], time.time()
    print(f"{cs['game_name']}: '{src.name}' (brain after {at/1e6:.1f}M moves) is teaching '{dst.name}' - {moves:,} moves.",
          flush=True)
    if cs.get("platform") == "arcade":
        from arcade_vec import Cabinet, local_core, scratch_dir
        rt = scratch_dir("teach")
        local_core(rt)
        rom = src.parent.parent / "ROMs - put your games here" / cs["rom_file"]
        cab = Cabinet(cs["game_id"], rom, rt, random.randrange(1 << 20), 0.0)

        def new_game():
            return cab.new_game()

        def step(a):
            o, _p, te, tr = cab.step(a)
            return o, te or tr, cab.score
    else:
        from ale_py import ALEInterface, roms
        ale = ALEInterface()
        ale.setInt("random_seed", random.randrange(1 << 30))
        ale.setFloat("repeat_action_probability", 0.0)
        ale.setInt("frame_skip", 1)
        ale.loadROM(roms.get_rom_path(cs["game_id"]))
        acts = ale.getMinimalActionSet()
        state = {"frames": None, "score": 0}

        def shot():
            img = np.maximum(state["frames"][-1], state["frames"][-2])
            return np.asarray(Image.fromarray(img).resize((84, 84), Image.BOX))

        def new_game():
            ale.reset_game()
            for _ in range(random.randint(0, 30)):
                ale.act(acts[0])
            if len(acts) > 1 and "FIRE" in str(acts[1]):
                ale.act(acts[1])
            state["frames"] = [ale.getScreenGrayscale().copy()] * 2
            state["score"] = 0
            return shot()

        def step(a):
            fr = []
            for _ in range(4):
                state["score"] += ale.act(acts[a])
                fr.append(ale.getScreenGrayscale().copy())
                if ale.game_over():
                    break
            state["frames"] = fr if len(fr) > 1 else fr * 2
            return shot(), ale.game_over(), state["score"]
    while len(actions) < moves:
        obs = new_game()
        stack = np.zeros((4, 84, 84), np.uint8)
        stack[-1] = obs
        greedy = random.random() < greedy_share      # some games its best moves, some its natural style
        first, over, n = True, False, 0
        while not over and n < 27000 and len(actions) < moves:
            a = pick(agent, stack, greedy)
            frames.append(stack[-1].copy())
            actions.append(a)
            starts.append(first)
            first = False
            obs, over, score = step(a)
            stack = np.roll(stack, -1, axis=0)
            stack[-1] = obs
            n += 1
        played += 1
        scores.append(score)
        print(f"  game {played}: {score:.0f} points ({len(actions):,} moves so far)", flush=True)
    r = demos.save_demo(dst, frames, actions, starts, n_act, cd["game_id"])
    msg = (f"Done: {len(actions):,} of the teacher's moves saved for '{dst.name}' (teacher averaged "
           f"{sum(scores)/len(scores):.0f} over {played} games, {(time.time()-t0)/60:.1f} min). Press Train on "
           f"'{dst.name}' and it copies the teacher first.")
    print(msg, flush=True)
    try:
        with open(dst / "files" / "run" / "console.log", "a", encoding="utf-8") as f:
            f.write(f"TEACHER: {msg}\n")
    except OSError:
        pass
    return r


if __name__ == "__main__":
    try:
        teach(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20000)
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e.code if isinstance(e.code, str) else f"Stopped ({e.code}).", flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
    try:                                            # in a window: let them read it before it closes
        if sys.stdin and sys.stdin.isatty():
            input("\nPress Enter to close this window.")
    except Exception:
        pass

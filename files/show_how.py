"""SHOW IT HOW - a short game where you show the kit which way the level goes and when you lose a life, so it can find
where the game keeps how far along it is and its lives (progress_find.py). Lights up 'Pay it for getting further',
'Penalty for losing a life' and 'A game over costs every life it had left' for that game. ALFRED 1 Nov 2026.

python show_how.py "<My AIs\\the AI>"      (opened by the page: Settings > a greyed-out dial > Show it how)"""
import sys, json, time, shutil
from pathlib import Path
import numpy as np

FILES = Path(__file__).resolve().parent
sys.path.insert(0, str(FILES))
sys.path.insert(0, str(FILES / "arcade" / "generic"))
import pygame
import arcade_vec
import scrollscan
import progress_find as pf
from PIL import Image

UP, DOWN, LEFT, RIGHT, COIN, START = 4, 5, 6, 7, 2, 3
KEYS = {pygame.K_UP: UP, pygame.K_DOWN: DOWN, pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
        pygame.K_z: 0, pygame.K_x: 8, pygame.K_c: 1, pygame.K_a: 9, pygame.K_s: 10,     # D = 'lost a life' here
        pygame.K_5: COIN, pygame.K_1: START}
BG, WHITE, GREY, GOLD, GREEN, RED = (12, 21, 35), (235, 235, 240), (150, 155, 170), (255, 196, 64), (90, 220, 120), (255, 110, 110)
HELP = ["SHOW IT HOW - play a little, the way the level goes.",
        "Arrows move.  Z X C (A S) are the game's buttons.",
        "Press R to start recording, then walk the way the level goes.",
        "Lost a life?  Press D once, straight after.  Lose 2 or 3, go a good way along, then press F.",
        "R = start recording (again)     F = finished, look now     Esc = close without saving"]


def main():
    proj = Path(sys.argv[1])
    cfg = json.loads((proj / "project.json").read_text())
    gid = cfg["game_id"]
    if not gid.startswith("found-"):
        raise SystemExit("Show it how works on arcade games the kit found itself.")
    import controls
    pdir = controls.pack_dir_for(gid)
    rom = controls.find_rom(gid) or (proj.parent.parent / "ROMs - put your games here" / cfg["rom_file"])
    root = arcade_vec.scratch_dir("showhow")
    arcade_vec.local_core(root, json.loads((pdir / "pack.json").read_text()).get("core", "fbneo"))
    env = arcade_vec.make_env(gid, rom, root)
    env.reset(seed=1)
    pygame.init()
    f = env.host.frame
    sc = 3 if f.shape[0] <= 260 else 2
    W, H = max(f.shape[1] * sc, 760), f.shape[0] * sc + 190
    win = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Show it how - {cfg.get('game_name', gid)}")
    font, big = pygame.font.SysFont("segoeui", 18), pygame.font.SysFont("segoeui", 22, bold=True)
    clock = pygame.time.Clock()
    rec = None
    g_prev = np.asarray(Image.fromarray(env.host.frame).convert("L"), np.uint8)
    mode, frame_n, result, msg = "ready", 0, None, "Press R when you're ready to start recording."
    check = None                      # live readings once something is found
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); return
            if e.type != pygame.KEYDOWN:
                continue
            if mode == "record" and e.key in (pygame.K_d, pygame.K_RETURN, pygame.K_KP_ENTER):
                rec.mark(); msg = f"Life lost - noted ({len(rec.marks)})."
            elif mode == "record" and e.key == pygame.K_f:
                mode, msg = "think", "Looking..."
            elif e.key == pygame.K_r and mode != "think":
                rec = pf.Recording(env.snap()); mode, result, check = "record", None, None    # from right now
                msg = "Recording - walk the way the level goes. D when you lose a life, F when you're done."
                g_prev = np.asarray(Image.fromarray(env.host.frame).convert("L"), np.uint8)
            elif mode == "result" and e.key == pygame.K_y and result and (result[0] or result[1]):
                msg = save(pdir, result); mode = "saved"
        held = pygame.key.get_pressed()
        buttons = tuple(sorted({b for k, b in KEYS.items() if held[k]}))
        env.host.run(buttons, capture=True)
        frame_n += 1
        if frame_n % env.decision_repeat == 0:
            mem = env.snap()
            g = np.asarray(Image.fromarray(env.host.frame).convert("L"), np.uint8)
            dx, dy = scrollscan.shift(g_prev, g)
            g_prev = g
            if mode == "record":
                rec.add(mem, dx, dy)
            if check is not None:
                check["cam"] += dx if result[0] and result[0]["axis"] == "x" else dy
        if mode == "think":
            draw(win, env, sc, font, big, mode, rec, msg, None, None); pygame.display.flip()
            sp, why_s = pf.find_scroll(rec)
            lv, why_l = pf.find_lives(rec)
            result = (sp, lv, why_s, why_l)
            mode, check = "result", {"cam": 0}
            msg = "Keep playing to check the readings below, then Y to save (R = record again, Esc = don't save)." \
                if (sp or lv) else "Nothing found yet - press R and try again (go further along, lose a couple of lives)."
        draw(win, env, sc, font, big, mode, rec, msg, result, check)
        pygame.display.flip()
        clock.tick(60)


def reading(env, spec):
    m = env.snap()
    o = spec["offset"]
    if spec["fmt"] == "u8":
        v, top = int(m[o]), 255
    else:
        a, b = int(m[o]), int(m[o + 1])
        v, top = ((a << 8 | b) if spec["fmt"] == "u16be" else (b << 8 | a)), 65535
    return top - v if spec.get("invert") else v


def draw(win, env, sc, font, big, mode, rec, msg, result, check):
    win.fill(BG)
    f = env.host.frame
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(f.swapaxes(0, 1)))
    surf = pygame.transform.scale(surf, (f.shape[1] * sc, f.shape[0] * sc))
    win.blit(surf, ((win.get_width() - surf.get_width()) // 2, 0))
    y = f.shape[0] * sc + 8
    if mode == "ready":
        win.blit(big.render("Not recording yet - press R to start.", True, GOLD), (12, y)); y += 30
        for line in HELP[1:]:
            win.blit(font.render(line, True, GREY), (12, y)); y += 22
    elif mode == "record":
        mx, my = rec.moved()
        far, lost = max(mx, my), len(rec.marks)
        ready = "  - enough, press F" if far >= 300 and lost >= 2 else ""
        win.blit(big.render(f"Recording: {rec.steps * env.decision_repeat // 60}s   screen moved {far:,} px   "
                            f"lives lost: {lost}{ready}", True, GREEN if ready else GOLD), (12, y)); y += 30
        for line in HELP[1:]:
            win.blit(font.render(line, True, GREY), (12, y)); y += 22
    elif mode in ("result", "saved", "think") and result:
        sp, lv, ws, wl = result
        if sp:
            v = reading(env, sp) / float(sp["units_per_px"] or 1)
            txt = f"How far along: FOUND - reading {v:,.0f}  (screen says it moved {check['cam']:+,} px since)"
        else:
            txt = f"How far along: not found - {ws}"
        win.blit(big.render(txt, True, GREEN if sp else RED), (12, y)); y += 30
        if lv:
            txt = f"Lives: FOUND - reading {int(env.snap()[lv['offset']]) - lv['adjust']}"
        else:
            txt = f"Lives: not found - {wl}"
        win.blit(big.render(txt, True, GREEN if lv else RED), (12, y)); y += 30
    if msg:
        win.blit(font.render(msg, True, WHITE), (12, y)); y += 24
    win.blit(font.render("R record   D lost a life   F finished   Y save   Esc close", True, GREY),
             (12, win.get_height() - 26))


def save(pdir, result):
    sp, lv, _, _ = result
    pj = pdir / "pack.json"
    shutil.copy2(pj, pdir / f"pack.json.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    pack = json.loads(pj.read_text(encoding="utf-8"))
    said = []
    if sp:
        keep = {k: sp[k] for k in ("offset", "fmt", "units_per_px", "max_jump", "new_stage_drop") if k in sp}
        if sp.get("invert"):
            keep["invert"] = True
        keep["per_life"] = bool(lv or pack.get("lives"))
        keep["found"] = f"shown by the player {time.strftime('%Y-%m-%d %H:%M')} (axis {sp['axis']}, match {sp['match']})"
        pack["progress"] = keep; said.append("how far along")
    if lv:
        pack["lives"] = {"offset": lv["offset"], "adjust": 0}; said.append("lives")
        if pack.get("progress"):
            pack["progress"]["per_life"] = True
    tmp = pj.with_name("pack.json.tmp")
    tmp.write_text(json.dumps(pack, indent=1), encoding="utf-8")
    tmp.replace(pj)
    return f"Saved {' and '.join(said)}. Close this window; the dials work from the next time you press Train."


if __name__ == "__main__":
    main()

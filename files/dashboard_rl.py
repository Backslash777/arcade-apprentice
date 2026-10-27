"""Reinforcement Learning live report card - every game, 2600 + Arcade, one page. Stdlib only.
Read-only: never starts, stops or touches training. Signed: Claude (9 Oct 2026).

  python dashboard_rl.py --open [--game "Arcade/Space Invaders"] [--port 8790]
If a dashboard is already running on the port, it just opens the browser on the requested game."""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # embeddable Python ignores the script folder

import argparse
import csv
import io
import json
import os
import shutil
import socket
import sys
import subprocess
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

HERE = Path(__file__).resolve().parent


def find_ml_root():
    for p in [HERE, *HERE.parents]:
        if p.name.lower() == "machine learning":
            return p
    return Path(r"C:\Users\terri\My Drive\Personal\0AI\Machine Learning")


ML = find_ml_root()
SIDES = ("Arcade", "2600")


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


class Tail:
    """Incremental reader for an append-only text file (jsonl/csv)."""

    def __init__(self, path):
        self.path, self.offset, self.ino, self.buf = Path(path), 0, None, b""

    def new_lines(self):
        try:
            st = self.path.stat()
        except OSError:
            return None
        ident = (st.st_size < self.offset)
        if ident or self.ino is None:
            if ident:
                self.offset, self.buf = 0, b""
            self.ino = True
        if st.st_size == self.offset:
            return []
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                data = f.read(64 * 1024 * 1024)
        except OSError:
            return []
        self.offset += len(data)
        data = self.buf + data
        lines = data.split(b"\n")
        self.buf = lines.pop()
        return [l.decode("utf-8", "replace").strip() for l in lines if l.strip()]


class Game:
    def __init__(self, side, folder: Path, evo=False):
        self.side, self.folder, self.files, self.evo = side, folder, folder / "files", evo
        self.id = f"{side}/Evolution/{folder.name}" if evo else f"{side}/{folder.name}"
        self.run = self.files / "run"
        self.lock = threading.Lock()
        self.games = []          # arcade: (step, score, corner or None)
        self.evals = []          # atari eval dicts
        self.train = []          # rows from train_log.csv
        self.train_header = None
        self.t_games = Tail(self.run / "games.jsonl")
        self.t_eval = Tail(self.run / "eval_log.jsonl")
        self.t_train = Tail(self.run / "train_log.csv")

    def cfg(self):
        g = read_json(self.files / "game.json") or {}
        if not g and (self.folder / "project.json").exists():
            pj = read_json(self.folder / "project.json") or {}
            g = {"name": pj.get("game_name", self.folder.name)}
            ts = pj.get("total_steps")
            if isinstance(ts, (int, float)) and 0 < ts < 1e11:      # "until I stop it" is stored as 1e12
                g["target"] = ts
        g.setdefault("name", self.folder.name)
        return g

    def refresh(self):
        with self.lock:
            for line in self.t_games.new_lines() or []:
                try:
                    d = json.loads(line)
                    if d.get("restart"):
                        continue          # Go-Explore restart from the map: not a from-the-start game
                    self.games.append((d.get("step", 0), d.get("score", 0), d.get("corner_pct")))
                except ValueError:
                    pass
            for line in self.t_eval.new_lines() or []:
                try:
                    self.evals.append(json.loads(line))
                except ValueError:
                    pass
            for line in self.t_train.new_lines() or []:
                if line.startswith("time,"):
                    self.train_header = line.split(",")
                    continue
                if self.train_header:
                    vals = next(csv.reader(io.StringIO(line)))
                    if len(vals) == len(self.train_header):
                        self.train.append(dict(zip(self.train_header, vals)))

    def status(self):
        st = read_json(self.run / "status.json") or {}
        upd = st.get("updated")
        age = None
        try:
            mt = (self.run / "status.json").stat().st_mtime
            age = time.time() - mt
        except OSError:
            pass
        live = st.get("state") in ("training", "stopping") and age is not None and age < 180
        return st, live, age

    def summary(self):
        self.refresh()
        st, live, age = self.status()
        cfg = self.cfg()
        kind = "atari" if self.side == "2600" else "arcade"
        platform = kind
        if self.side == "Portable":           # the kit: games.jsonl for both, but say which machine it is
            pj = read_json(self.folder / "project.json") or {}
            platform = pj.get("platform") or (pj.get("hardware") or {}).get("platform") or "atari"
        best = last = None
        if kind == "arcade" and self.games:
            sc = [g[1] for g in self.games]
            best, last = max(sc), sum(sc[-100:]) / len(sc[-100:])
        elif self.evals:
            best = max(e["max"] for e in self.evals)
            last = self.evals[-1]["mean"]
        return {"id": self.id, "side": self.side, "name": cfg["name"], "kind": kind, "platform": platform, "live": live, "evo": self.evo,
                "state": st.get("state", "not started"), "step": st.get("step", 0), "best": best, "last": last,
                "bar": cfg.get("bar"), "age": age}

    def detail(self):
        s = self.summary()
        cfg = self.cfg()
        st, live, age = self.status()
        with self.lock:
            games = list(self.games)
            evals = list(self.evals)
            train = list(self.train)
        out = {**s, "cfg": cfg, "status": st}
        if self.evo:
            out["pbt"] = read_json(self.files / "pbt_state.json")
            try:
                out["pbt_log"] = [json.loads(l) for l in open(self.files / "pbt_log.jsonl", encoding="utf-8") if l.strip()][-40:]
            except OSError:
                out["pbt_log"] = []
        # per-game scatter (downsampled to <= 4000 pts) + rolling mean of 100
        if games:
            n = len(games)
            stride = max(1, n // 4000)
            out["scatter"] = [[games[i][0], games[i][1]] for i in range(0, n, stride)]
            roll, acc, cacc, rc = [], 0.0, 0.0, []
            window = []
            cw = []
            for i, (step, score, corner) in enumerate(games):
                window.append(score)
                acc += score
                if len(window) > 100:
                    acc -= window.pop(0)
                if corner is not None:
                    cw.append(corner)
                    cacc += corner
                    if len(cw) > 100:
                        cacc -= cw.pop(0)
                if i % stride == 0 or i == n - 1:
                    roll.append([step, acc / len(window)])
                    if cw:
                        rc.append([step, cacc / len(cw)])
            out["rolling"] = roll
            out["corner_series"] = rc
            last500 = [g[1] for g in games[-500:]]
            out["recent_scores"] = last500
            top = sorted(range(n), key=lambda i: games[i][1], reverse=True)[:10]
            out["top10"] = [{"score": games[i][1], "step": games[i][0], "game_no": i + 1} for i in top]
            out["games_played"] = n
        if evals:
            out["evals"] = evals
        if train:
            stride = max(1, len(train) // 1500)
            keys = [k for k in ("sps", "entropy", "kl", "expl_var", "train_score_mean200", "score_mean100",
                                "corner_pct100") if k in train[0]]
            ser = {k: [] for k in keys}
            for i in range(0, len(train), stride):
                r = train[i]
                for k in keys:
                    try:
                        v = float(r[k])
                        if v == v and abs(v) != float("inf"):
                            ser[k].append([int(r["step"]), v])
                    except (ValueError, KeyError):
                        pass
            out["train_series"] = ser
            if len(train) >= 2:
                try:
                    t0 = time.mktime(time.strptime(train[0]["time"], "%Y-%m-%d %H:%M:%S"))
                    t1 = time.mktime(time.strptime(train[-1]["time"], "%Y-%m-%d %H:%M:%S"))
                    out["wall_hours"] = (t1 - t0) / 3600
                except (ValueError, KeyError):
                    pass
            # ALFRED 23 Oct 2026: hours of training for the graph's Hours view - the time between log rows, leaving
            # out the gaps when it was stopped (rows more than 10 minutes apart). [moves, hours] pairs, <= 2,000.
            try:
                hm, h, prev_t, top = [], 0.0, None, -1
                for r in train:
                    t = time.mktime(time.strptime(r["time"], "%Y-%m-%d %H:%M:%S"))
                    s_ = int(float(r["step"]))
                    if prev_t is not None and 0 < t - prev_t <= 600:
                        h += (t - prev_t) / 3600
                    prev_t = t
                    if s_ > top:
                        hm.append([s_, round(h, 4)])
                        top = s_
                k = max(1, len(hm) // 2000)
                out["hours_map"] = hm[::k] + ([hm[-1]] if hm and (len(hm) - 1) % k else [])
                out["train_hours"] = h
            except (ValueError, KeyError):
                pass
        try:
            with (self.run / "console.log").open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 6000))
                out["console"] = f.read().decode("utf-8", "replace").splitlines()[-24:]
        except OSError:
            out["console"] = []
        return out


GAMES: dict[str, Game] = {}


def clean(o):
    """JSON for browsers: NaN/inf become null."""
    if isinstance(o, float):
        return o if o == o and abs(o) != float("inf") else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


PORTABLE = os.environ.get("PORTABLE_PROJECTS")


def discover():
    if PORTABLE:
        base = Path(PORTABLE)
        for d in sorted(base.iterdir()) if base.is_dir() else []:
            if d.is_dir() and (d / "project.json").exists() and f"Portable/{d.name}" not in GAMES:
                g = Game("Portable", d)
                g.id = f"Portable/{d.name}"
                GAMES[g.id] = g
        return
    for side in SIDES:
        for section in ("Reinforcement Learning", "Evolution"):
            base = ML / side / section
            if not base.is_dir():
                continue
            for d in sorted(base.iterdir()):
                if d.is_dir() and (d / "files").is_dir():
                    gid = f"{side}/{d.name}" if section == "Reinforcement Learning" else f"{side}/Evolution/{d.name}"
                    if gid not in GAMES:
                        GAMES[gid] = Game(side, d, evo=(section == "Evolution"))


SYS = {"t": 0, "data": {}}
try:                                    # psutil's first CPU reading is always 0 - take it now, not on the report card
    import psutil as _psutil
    _psutil.cpu_percent(interval=None)
except Exception:
    pass


def system_stats():
    if time.time() - SYS["t"] < 2:
        return SYS["data"]
    d = {}
    try:
        import psutil
        d["cpu"] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        d["ram"] = vm.percent
        d["ram_text"] = f"{vm.used/2**30:.1f} / {vm.total/2**30:.0f} GB"
    except Exception:
        pass
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            r = subprocess.run([smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            name, util, mu, mt, temp, pw = [x.strip() for x in r.stdout.strip().split(",")[:6]]
            d.update({"gpu_name": name, "gpu": float(util), "vram": 100 * float(mu) / float(mt),
                      "vram_text": f"{float(mu)/1024:.1f} / {float(mt)/1024:.0f} GB", "gpu_temp": temp,
                      "gpu_power": pw})
        except Exception:
            pass
    SYS.update(t=time.time(), data=d)
    return d


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                return self.send(PAGE, "text/html")
            if u.path == "/api/games":
                discover()
                return self.send(json.dumps(clean([g.summary() for g in GAMES.values()]), default=str))
            if u.path == "/api/game":
                discover()
                g = GAMES.get(q.get("id", [""])[0])
                return self.send(json.dumps(clean(g.detail() if g else {"error": "unknown game"}), default=str))
            if u.path == "/api/system":
                return self.send(json.dumps(system_stats()))
            if u.path == "/api/watch":
                discover()
                g = GAMES.get(q.get("id", [""])[0])
                if g and PORTABLE:
                    subprocess.Popen([sys.executable, str(HERE / "watch.py"), str(g.folder)], cwd=str(HERE),
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    return self.send(json.dumps({"ok": True, "msg": "Opening the game window"}))
                bat = g.folder / "WATCH.bat" if g else None
                if not bat or not bat.exists():
                    return self.send(json.dumps({"ok": False, "msg": "no WATCH.bat for this game"}))
                env = dict(os.environ)
                env.pop("PYTHONHOME", None)
                env.pop("PYTHONPATH", None)
                subprocess.Popen(["cmd", "/c", "start", "", str(bat)], cwd=str(g.folder), env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                return self.send(json.dumps({"ok": True, "msg": "WATCH window opening (latest brain)"}))
            if u.path.startswith("/help/"):          # the kit's fonts and pictures of Chip (files\help)
                kinds = {".woff2": "font/woff2", ".png": "image/png", ".webp": "image/webp"}
                base = (HERE / "help").resolve()
                f = (base / unquote(u.path[len("/help/"):])).resolve()
                if f.suffix.lower() in kinds and f.is_file() and base in f.parents:
                    data = f.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", kinds[f.suffix.lower()])
                    self.send_header("Cache-Control", "max-age=86400")
                    self.end_headers()
                    self.wfile.write(data)
                    return
            if u.path == "/api/quit":               # the kit's "Close the kit" closes the report card too
                self.send(json.dumps({"ok": True}))
                threading.Timer(0.3, lambda: os._exit(0)).start()
                return
            if u.path == "/api/ping":
                return self.send('{"app":"rl-report-card"}')
        except Exception as e:  # never die on a bad read
            return self.send(json.dumps({"error": repr(e)}))
        self.send_response(404)
        self.end_headers()


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--game", default="")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/" + (f"#{quote(a.game)}" if a.game else "")
    if port_in_use(a.port):
        print("Report card already running - opening it.")
        webbrowser.open(url)
        return
    discover()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"RL report card on {url}  (read-only; close this window to stop the report, training is unaffected)")
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Report card - Arcade Apprentice</title>
<script>(function(){var t=null;try{var q=new URLSearchParams(location.search).get('theme');if(q=='light'||q=='dark'){t=q;localStorage.setItem('theme',q)}else t=localStorage.getItem('theme')}catch(e){}
if(t!='light'&&t!='dark')t=window.matchMedia&&matchMedia('(prefers-color-scheme: light)').matches?'light':'dark';document.documentElement.dataset.theme=t})()</script>
<link rel="icon" type="image/png" href="/help/chip-icon.png">
<style>
@font-face{font-family:'Press Start 2P';src:url('/help/fonts/press-start-2p.woff2') format('woff2');font-display:swap}
@font-face{font-family:'Atkinson Hyperlegible';src:url('/help/fonts/atkinson-400.woff2') format('woff2');font-weight:400;font-display:swap}
@font-face{font-family:'Atkinson Hyperlegible';src:url('/help/fonts/atkinson-700.woff2') format('woff2');font-weight:700;font-display:swap}
@font-face{font-family:'Zero Fix';src:url('/help/fonts/zero-fix-400.woff2') format('woff2');font-weight:400;font-display:swap;unicode-range:U+0030}
@font-face{font-family:'Zero Fix';src:url('/help/fonts/zero-fix-700.woff2') format('woff2');font-weight:700;font-display:swap;unicode-range:U+0030}
@font-face{font-family:'VT323';src:url('/help/fonts/vt323.woff2') format('woff2');font-display:swap}
/* same colours as the kit's page and arcadeapprentice.co.uk: the Golden Mile at night / by day */
:root{color-scheme:dark;--bg:#12052a;--glow1:rgba(255,60,172,.11);--glow2:rgba(34,230,255,.07);--panel:#170838;--card:#1d0b45;--card-hi:#261157;
--line:#341a5e;--line-strong:#55318c;--grid:#2a1650;--text:#fbf3e1;--muted:#cbbbe0;--faint:#9d8cc0;
--accent:#ff6a1a;--accent-deep:#b8420a;--accent-ink:#0c1626;--accent-text:#ffe14d;--accent-soft:rgba(255,106,26,.15);
--teal:#22e6ff;--lime:#7cff6b;--pink:#ff3cac;--violet:#b86bff;--s-avg:#ff7a2e;
--c-dots:rgba(34,230,255,.30);--c-band:rgba(255,106,26,.15);--c-bar:#ffe14d;--focus:#ffe14d;
--shadow:0 1px 0 rgba(255,255,255,.04) inset,0 18px 40px -26px rgba(0,0,0,.9);
--ui:"Zero Fix","Atkinson Hyperlegible","Segoe UI",system-ui,Arial,sans-serif;--pixel:"Press Start 2P","Courier New",monospace;--crt:"VT323",Consolas,monospace}
:root[data-theme="light"]{color-scheme:light;--bg:#f7eedb;--glow1:rgba(255,60,172,.07);--glow2:rgba(34,230,255,.12);--panel:#fffbf2;--card:#fbf3e1;--card-hi:#f4e7cb;
--line:#e8d9bb;--line-strong:#d3bd93;--grid:#eee2c8;--text:#12052a;--muted:#584a72;--faint:#7a6c91;
--accent-text:#b8420a;--accent-soft:rgba(255,106,26,.12);--teal:#007c91;--lime:#1d7f33;--pink:#c8157a;--violet:#7a35d0;--s-avg:#d9520b;
--c-dots:rgba(0,124,145,.28);--c-band:rgba(217,82,11,.12);--c-bar:#9a7400;--focus:#7a35d0;
--shadow:0 1px 2px rgba(18,5,42,.06),0 14px 34px -24px rgba(18,5,42,.35)}
*{box-sizing:border-box}
body{margin:0;color:var(--text);font:14.5px/1.5 var(--ui);-webkit-font-smoothing:antialiased;min-height:100vh;
 background:radial-gradient(1100px 520px at 88% -12%,var(--glow1),transparent 62%),radial-gradient(900px 480px at -8% -18%,var(--glow2),transparent 60%),var(--bg);background-attachment:fixed}
:focus-visible{outline:3px solid var(--focus);outline-offset:2px}
main{max-width:1560px;margin:auto;padding:22px 30px 34px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px 20px;margin-bottom:16px;flex-wrap:wrap}
.brand{display:flex;gap:14px;align-items:center}
.chiphead{width:64px;height:auto;flex:none;filter:drop-shadow(0 4px 10px rgba(255,106,26,.28))}
.wordmark{font:400 12px/1.4 var(--pixel);color:var(--text);text-shadow:2px 2px 0 var(--pink);margin-bottom:7px;letter-spacing:.5px}
h1{font-size:25px;line-height:1.2;margin:0;font-weight:700}
h2{font-size:15px;font-weight:700;margin:0}
.sub{font-size:13px;margin-top:3px;color:var(--muted)}
.badges{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{padding:6px 12px;border-radius:999px;background:var(--card);border:1px solid var(--line);font:700 12.5px/1.3 var(--ui);white-space:nowrap;color:var(--text)}
button{font:inherit;cursor:pointer;color:var(--text)}
.watchbtn{background:var(--accent);color:var(--accent-ink);border-color:var(--accent-deep);box-shadow:0 3px 0 var(--accent-deep);font-size:13px;padding:7px 14px}
.watchbtn:hover{filter:brightness(1.07)}
.themebtn{position:relative;width:66px;height:34px;padding:0 9px;border-radius:999px;border:1px solid var(--line);background:var(--card);display:flex;align-items:center;justify-content:space-between;flex:none}
.themebtn svg{width:15px;height:15px;position:relative;z-index:1;color:var(--muted)}
.themebtn::before{content:'';position:absolute;top:3px;left:3px;width:26px;height:26px;border-radius:50%;background:var(--accent);box-shadow:inset 0 -2px 0 var(--accent-deep);transition:transform .22s ease}
.themebtn .moon{color:var(--accent-ink)}
:root[data-theme="light"] .themebtn::before{transform:translateX(32px)}
:root[data-theme="light"] .themebtn .moon{color:var(--muted)}:root[data-theme="light"] .themebtn .sun{color:var(--accent-ink)}
.dot{width:8px;height:8px;display:inline-block;margin-right:7px;background:var(--faint);border-radius:50%;vertical-align:middle;opacity:.7}
.dot.live{background:var(--lime);opacity:1;animation:pulse 1.8s ease-out infinite}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(124,255,107,.5)}70%{box-shadow:0 0 0 7px transparent}100%{box-shadow:0 0 0 0 transparent}}
.tabgroup{font-size:13px;font-weight:700;color:var(--muted);width:100%;margin:2px 0 -2px}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.tab{cursor:pointer;padding:10px 14px;border-radius:14px;background:var(--panel);border:1px solid var(--line);min-width:170px;transition:border-color .15s}
.tab:hover{border-color:var(--line-strong)}.tab.sel{border-color:var(--accent);background:var(--accent-soft);box-shadow:0 0 0 3px var(--accent-soft)}
.tab .side{font-size:12px;color:var(--muted)}
.tab .nm{font-weight:700;font-size:14.5px}.tab .mini{font-size:12px;color:var(--muted)}
.verdict{font-size:14px;padding:11px 14px;border-radius:14px;border:1px solid var(--line);background:var(--panel);margin-bottom:14px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:14px}
.metric{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px 15px 13px;box-shadow:var(--shadow)}
.metric .label{font:20px/1 var(--crt);color:var(--teal)}
.metric .value{font:400 21px/1.2 var(--pixel);color:var(--accent-text);margin-top:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:14px 16px;min-width:0;box-shadow:var(--shadow)}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}
.grid.nocorner .flex6{grid-column:span 6}
.panelhead{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;gap:10px;flex-wrap:wrap}
.legend{display:flex;gap:12px;font-size:12.5px;color:var(--muted);flex-wrap:wrap}
.xsw{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;flex:none}
.xsw button{border:0;background:transparent;padding:4px 12px;font-size:12.5px;color:var(--muted)}
.xsw button.on{background:var(--accent-soft);color:var(--text);font-weight:700}.xsw button:disabled{opacity:.45;cursor:default}
.key{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
canvas{width:100%;display:block}
.meterlabel{display:flex;justify-content:space-between;font-size:13px;margin-top:8px}
.meter{height:9px;background:var(--card-hi);border-radius:999px;overflow:hidden;margin-top:3px}
.meter i{display:block;height:100%;background:var(--teal);border-radius:999px;transition:width .6s}
.progress{height:8px;background:var(--card-hi);border-radius:3px;overflow:hidden;margin:8px 0 2px}
.progress div{height:100%;background:repeating-linear-gradient(90deg,var(--c-bar) 0 8px,transparent 8px 11px);transition:width .5s}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-size:12.5px;font-weight:700;padding:4px 8px}
td{padding:5px 8px;border-top:1px solid var(--line);font-variant-numeric:tabular-nums}
.console{font:12px/1.5 "Cascadia Mono",Consolas,monospace;background:#07020f;border:1px solid #2a1650;border-radius:12px;padding:10px 12px;
max-height:300px;overflow:auto;white-space:pre;color:#e9e1f7}
.good{color:var(--lime)}.warn{color:var(--accent-text)}.bad{color:var(--pink)}.quiet{color:var(--muted)}
footer{display:flex;justify-content:space-between;gap:8px 16px;flex-wrap:wrap;color:var(--faint);font-size:13px;margin-top:16px}
@media(max-width:1100px){.span3,.span4,.span5,.span6,.span7,.span8,.grid.nocorner .flex6{grid-column:span 12}}
@media(max-width:700px){main{padding:16px 14px 28px}}
@media (prefers-reduced-motion: reduce){*{animation:none!important;transition:none!important}}
</style></head><body><main>
<header><div class="brand"><img class="chiphead" src="/help/chip-head.webp" width="64" height="37" alt=""><div><div class="wordmark">Report card</div>
<h1 id="title">Loading&hellip;</h1><div class="sub" id="subtitle"></div></div></div>
<div class="badges"><button class="themebtn" id="themebtn" onclick="flipTheme()" aria-label="Light page" aria-pressed="false" title="Switch to the light page"><svg class="moon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M10.6 1.6A6.6 6.6 0 1 0 14.4 11 5.4 5.4 0 0 1 10.6 1.6z"/></svg><svg class="sun" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3.2" fill="currentColor"/><path stroke="currentColor" stroke-width="1.6" stroke-linecap="round" d="M8 1v1.6M8 13.4V15M1 8h1.6M13.4 8H15M3.05 3.05l1.13 1.13M11.82 11.82l1.13 1.13M3.05 12.95l1.13-1.13M11.82 4.18l1.13-1.13"/></svg></button>
<button class="badge watchbtn" id="watchbtn" title="Opens the game window with the newest trained brain">&#9654; Watch it play</button><span class="badge" id="conn"><i class="dot"></i>Connecting</span><span class="badge" id="clock"></span></div></header>
<nav class="tabs" id="tabs"></nav>
<div class="verdict" id="verdict"></div>
<section class="metrics">
<div class="metric"><div class="label" id="m1l">Average, last 100 games</div><div class="value" id="m1">&mdash;</div><div class="sub" id="m1s"></div></div>
<div class="metric"><div class="label">Best game</div><div class="value" id="m2">&mdash;</div><div class="sub" id="m2s"></div></div>
<div class="metric" id="m3box"><div class="label">Versus the bar</div><div class="value" id="m3">&mdash;</div><div class="sub" id="m3s"></div></div>
<div class="metric" id="m4box"><div class="label">Corner camping</div><div class="value" id="m4">&mdash;</div><div class="sub" id="m4s"></div></div>
<div class="metric"><div class="label">Moves a second</div><div class="value" id="m5">&mdash;</div><div class="sub" id="m5s"></div></div>
<div class="metric"><div class="label">Moves practised</div><div class="value" id="m6">&mdash;</div><div class="progress"><div id="prog" style="width:0"></div></div><div class="sub" id="m6s"></div></div>
</section>
<section class="grid" id="grid">
<div class="panel span8"><div class="panelhead"><h2>Score as it learns</h2><div class="legend" id="lg1"></div><div class="xsw" role="group" aria-label="Along the bottom"><button id="xsw_m" onclick="setX('moves')" title="Moves of practice - fair for the same game on different PCs">Moves</button><button id="xsw_h" onclick="setX('hours')" title="Hours of training - fair for different games on the same PC">Hours</button></div></div><canvas id="c1" height="300"></canvas><div class="sub" id="c1note"></div></div>
<div class="panel span4"><div class="panelhead"><h2 id="c2t">Last 500 games</h2><span class="sub" id="c2s">how the scores spread out</span></div><canvas id="c2" height="300"></canvas></div>
<div class="panel span4" id="cornerpanel"><div class="panelhead"><h2>Corner camping</h2><span class="sub">time spent parked in a corner</span></div><canvas id="c3" height="190"></canvas></div>
<div class="panel span4 flex6"><div class="panelhead"><h2>Brain health</h2><div class="legend"><span><i class="key" style="background:var(--violet)"></i>randomness</span><span><i class="key" style="background:var(--teal)"></i>how well it predicts its score</span></div></div><canvas id="c4" height="190"></canvas></div>
<div class="panel span4 flex6"><div class="panelhead"><h2>Your computer</h2><span class="sub" id="gpuname"></span></div>
<div class="meterlabel"><span>Graphics card</span><span id="gput">&mdash;</span></div><div class="meter"><i id="gpu" style="background:var(--violet);width:0"></i></div>
<div class="meterlabel"><span>Graphics memory</span><span id="vramt">&mdash;</span></div><div class="meter"><i id="vram" style="background:var(--violet);width:0"></i></div>
<div class="meterlabel"><span>Processor</span><span id="cput">&mdash;</span></div><div class="meter"><i id="cpu" style="width:0"></i></div>
<div class="meterlabel"><span>Memory</span><span id="ramt">&mdash;</span></div><div class="meter"><i id="ram" style="background:var(--lime);width:0"></i></div>
<canvas id="c5" height="70" style="margin-top:10px"></canvas><div class="sub">moves a second, from the start to now</div></div>
<div class="panel span5"><div class="panelhead"><h2 id="tabletitle">Hall of fame</h2><span class="sub" id="tablesub"></span></div><div style="max-height:300px;overflow:auto"><table id="tbl"></table></div></div>
<div class="panel span12" id="evopanel" style="display:none"><div class="panelhead"><h2>Evolution: the family</h2><span class="sub" id="evosub"></span></div>
<div class="grid"><div style="grid-column:span 7"><table id="poptbl"></table></div><div style="grid-column:span 5"><canvas id="c6" height="210"></canvas><div class="sub">best and average score in each generation</div></div></div>
<div class="console" id="evolog" style="margin-top:10px;max-height:170px"></div></div>
<div class="panel span7"><div class="panelhead"><h2>Training console</h2><span class="sub">the training's own words</span></div><div class="console" id="console"></div></div>
</section>
<footer><span>Read-only: closing this page never touches training. &copy; 2026 Jonathan Terrington</span><span id="updated"></span></footer>
</main>
<script>
const $=id=>document.getElementById(id);
let current=decodeURIComponent(location.hash.slice(1))||null, list=[], lastOk=0;
// ALFRED 23 Oct 2026: the score graph's Moves / Hours switch (remembered in this browser)
let XMODE='moves';try{XMODE=localStorage.getItem('xmode')||'moves'}catch(e){}
function setX(m){XMODE=m;try{localStorage.setItem('xmode',m)}catch(e){}load()}
function hoursOf(map){if(!map||map.length<2)return null;const S=map.map(p=>p[0]),H=map.map(p=>p[1]),n=S.length-1;
 return s=>{if(s<=S[0])return H[0];if(s>=S[n])return H[n];let lo=0,hi=n;while(hi-lo>1){const m=(lo+hi)>>1;if(S[m]<=s)lo=m;else hi=m}
  return H[lo]+(s-S[lo])/Math.max(1,S[hi]-S[lo])*(H[hi]-H[lo])}}
function hrs(h){return h<=0?'0h':h<1?Math.round(h*60)+'m':(h<10&&Math.abs(h-Math.round(h))>1e-6?h.toFixed(1):Math.round(h))+'h'}
function hourTicks(x0,x1){const st=[0.25,0.5,1,2,3,6,12,24,48,96,168].find(s=>(x1-x0)/s<=6)||336,t=[];for(let v=Math.ceil(x0/st-1e-9)*st;v<=x1+1e-9;v+=st)t.push(v);return t}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cap=s=>s?String(s)[0].toUpperCase()+String(s).slice(1):'';
const fmt=(v,d=0)=>v==null||isNaN(v)?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d});
const big=v=>v==null?'—':Math.abs(v)<10&&v!=Math.round(v)?(+v).toFixed(2):Math.abs(v)>=1e6?(v/1e6).toFixed(1)+'M':Math.abs(v)>=1e3?(v/1e3).toFixed(1)+'k':String(Math.round(v));
function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim()}
const FONT='"Zero Fix","Atkinson Hyperlegible","Segoe UI",sans-serif';
function curTheme(){return document.documentElement.dataset.theme=='light'?'light':'dark'}
function setTheme(t){if(t){try{localStorage.setItem('theme',t)}catch(e){}document.documentElement.dataset.theme=t}
 const b=$('themebtn');if(b){const l=curTheme()=='light';b.setAttribute('aria-pressed',l?'true':'false');b.title=l?'Switch to the dark page':'Switch to the light page'}}
function flipTheme(){setTheme(curTheme()=='dark'?'light':'dark');load()}

function setup(c){const r=window.devicePixelRatio||1;if(!c._h){c._h=+c.getAttribute('height');c.style.height=c._h+'px'}const w=c.clientWidth;c.width=w*r;c.height=c._h*r;
 const g=c.getContext('2d');g.setTransform(r,0,0,r,0,0);return [g,w,c._h]}
function chart(c,series,o={}){const [g,W,H]=setup(c);const P={l:52,r:12,t:10,b:24};g.clearRect(0,0,W,H);
 let xs=[],ys=[];series.forEach(s=>(s.pts||[]).forEach(p=>{xs.push(p[0]);ys.push(p[1]);if(s.hi)s.hi.forEach(q=>ys.push(q[1]))}));
 if(o.bar!=null)ys.push(o.bar);
 if(!xs.length){g.fillStyle=css('--faint');g.font='13px '+FONT;g.textAlign='center';g.fillText(o.empty||'No data yet',W/2,H/2);return}
 let x0=Math.min(...xs),x1=Math.max(...xs);if(x1==x0)x1=x0+1;let y0=o.ymin!=null?o.ymin:Math.min(0,...ys),y1=o.ymax!=null?o.ymax:Math.max(...ys)*1.06;if(y1==y0)y1=y0+1;
 const X=x=>P.l+(x-x0)/(x1-x0)*(W-P.l-P.r),Y=y=>H-P.b-(y-y0)/(y1-y0)*(H-P.t-P.b);
 g.strokeStyle=css('--grid');g.fillStyle=css('--faint');g.font='11px '+FONT;g.lineWidth=1;
 for(let i=0;i<=4;i++){const yv=y0+(y1-y0)*i/4,yy=Y(yv);g.beginPath();g.moveTo(P.l,yy);g.lineTo(W-P.r,yy);g.stroke();g.textAlign='right';g.fillText(o.pct?Math.round(yv)+'%':big(yv),P.l-6,yy+3)}
 if(o.xticks)o.xticks(x0,x1).forEach(xv=>{const xx=X(xv);g.textAlign=xx<P.l+14?'left':xx>W-P.r-14?'right':'center';g.fillText(o.xfmt?o.xfmt(xv):big(xv),xx,H-7);g.beginPath();g.moveTo(xx,H-P.b);g.lineTo(xx,H-P.b+4);g.stroke()});
 else for(let i=0;i<=5;i++){const xv=x0+(x1-x0)*i/5;g.textAlign=i==0?'left':i==5?'right':'center';g.fillText(o.xfmt?o.xfmt(xv):big(xv),i==0?P.l:i==5?W-P.r:X(xv),H-7)}
 if(o.bar!=null){g.setLineDash([5,4]);g.strokeStyle=css('--c-bar');g.beginPath();g.moveTo(P.l,Y(o.bar));g.lineTo(W-P.r,Y(o.bar));g.stroke();g.setLineDash([]);
  g.fillStyle=css('--c-bar');g.textAlign='left';g.fillText('bar '+big(o.bar),P.l+4,Y(o.bar)-4)}
 series.forEach(s=>{const pts=s.pts||[];if(!pts.length)return;
  if(s.hi&&s.lo){g.fillStyle=s.band;g.beginPath();s.hi.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));
   [...s.lo].reverse().forEach(p=>g.lineTo(X(p[0]),Y(p[1])));g.closePath();g.fill()}
  if(s.dots){g.fillStyle=s.color;pts.forEach(p=>{g.fillRect(X(p[0])-1,Y(p[1])-1,2,2)});return}
  g.strokeStyle=s.color;g.lineWidth=s.w||2;g.lineJoin='round';g.beginPath();pts.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();
  if(s.mark){g.fillStyle=s.color;pts.forEach(p=>{g.beginPath();g.arc(X(p[0]),Y(p[1]),3,0,7);g.fill()})}})}
function hist(c,vals,bar){const [g,W,H]=setup(c);g.clearRect(0,0,W,H);if(!vals||!vals.length){g.fillStyle=css('--faint');g.font='13px '+FONT;g.textAlign='center';g.fillText('No games yet',W/2,H/2);return}
 // from the lowest score to the highest (some games, like Pitfall, go below zero)
 const mn=Math.min(...vals,0),mx=Math.max(...vals,mn+1),nb=20,b=new Array(nb).fill(0);vals.forEach(v=>b[Math.min(nb-1,Math.max(0,Math.floor((v-mn)/(mx-mn)*nb)))]++);const bm=Math.max(...b);
 const P={l:10,r:10,t:10,b:24},bw=(W-P.l-P.r)/nb;b.forEach((n,i)=>{const h=(H-P.t-P.b)*n/bm;const lo=mn+i*(mx-mn)/nb;
  g.fillStyle=bar!=null&&lo>=bar?css('--lime'):css('--s-avg');g.globalAlpha=.85;g.beginPath();g.roundRect?g.roundRect(P.l+i*bw+1,H-P.b-h,bw-2,h,[3,3,0,0]):g.rect(P.l+i*bw+1,H-P.b-h,bw-2,h);g.fill();g.globalAlpha=1});
 g.fillStyle=css('--faint');g.font='11px '+FONT;g.textAlign='left';g.fillText(big(mn),P.l,H-7);g.textAlign='right';g.fillText(big(mx),W-P.r,H-7);
 if(bar!=null&&bar>mn&&bar<mx){const x=P.l+((bar-mn)/(mx-mn))*(W-P.l-P.r);g.strokeStyle=css('--c-bar');g.setLineDash([4,3]);g.beginPath();g.moveTo(x,P.t);g.lineTo(x,H-P.b);g.stroke();g.setLineDash([]);
  g.fillStyle=css('--c-bar');g.textAlign='center';g.fillText('bar',x,H-7)}}

async function getJSON(u){const r=await fetch(u,{cache:'no-store'});return r.json()}
const PLAT=g=>g.platform=='atari'?'Atari 2600':'Arcade';
function tabs(){const one=g=>`<div class="tab ${g.id==current?'sel':''}" data-id="${esc(g.id)}" tabindex="0"><div class="side">${PLAT(g)}${g.evo?', evolution':''}</div>
 <div class="nm"><i class="dot ${g.live?'live':''}"></i>${esc(g.name)}</div><div class="mini">${g.state=='not started'?'Not trained yet':(g.live?'Training now':esc(cap(g.state)))}${g.step?', '+big(g.step)+' moves':''}${g.last!=null?', average '+fmt(g.last):''}</div></div>`;
 const both=list.some(g=>g.evo)&&list.some(g=>!g.evo);
 $('tabs').innerHTML=(both?'<div class="tabgroup">Practice</div>':'')+list.filter(g=>!g.evo).map(one).join('')+(list.some(g=>g.evo)?(both?'<div class="tabgroup">Evolution</div>':'')+list.filter(g=>g.evo).map(one).join(''):'');
 document.querySelectorAll('.tab').forEach(t=>{t.onclick=()=>{current=t.dataset.id;location.hash=encodeURIComponent(current);tabs();load()};t.onkeydown=e=>{if(e.key=='Enter'||e.key==' '){e.preventDefault();t.click()}}})}
async function loadList(){list=await getJSON('/api/games');if(!current||!list.find(g=>g.id==current)){const l=list.find(g=>g.live);current=(l||list[0]||{}).id}tabs()}
async function load(){if(!current)return;const d=await getJSON('/api/game?id='+encodeURIComponent(current));if(d.error)return;
 lastOk=Date.now();const cfg=d.cfg||{},st=d.status||{},bar=d.bar,kit=d.side=='Portable';
 $('title').textContent=d.name+(d.evo?' (evolution)':'');document.title=d.name+' - report card';
 const P=d.pbt;$('evopanel').style.display=d.evo?'':'none';
 if(d.evo){const mem=(P&&P.members)||[];const cur=(d.status||{}).current_member;
  $('evosub').textContent=P?`generation ${P.generation}, ${mem.length} brains, ${big(P.slice)} moves per turn, ${big(P.total_steps)} in total`:'not started yet';
  $('poptbl').innerHTML='<tr><th>Brain</th><th>Score</th><th>Learning speed</th><th>Randomness</th><th>Plans ahead</th><th>Step size</th><th>Moves</th><th>Family line</th></tr>'+[...mem].sort((a,b)=>(b.fitness??-1)-(a.fitness??-1)).map(m=>`<tr${m.id==cur?' style="background:var(--accent-soft)"':''}><td><b>${esc(m.id)}</b>${m.id==cur?' <span class="warn">learning now</span>':''}</td><td>${fmt(m.fitness)}</td><td>${m.genes.lr}</td><td>${m.genes.ent}</td><td>${Math.round(1/(1-m.genes.gamma))} moves</td><td>${m.genes.clip}</td><td>${big(m.step)}</td><td class="quiet">${(m.lineage||[]).map(esc).join(' → ')}</td></tr>`).join('');
  const L=d.pbt_log||[];chart($('c6'),[{pts:L.map(r=>[r.generation,(r.ranking[0]||{}).fitness||0]),color:css('--s-avg'),mark:1},{pts:L.map(r=>[r.generation,r.ranking.reduce((a,b)=>a+(b.fitness||0),0)/Math.max(1,r.ranking.length)]),color:css('--teal'),mark:1}],{bar:bar,xfmt:v=>'gen '+Math.round(v),empty:'First generation in progress'});
  $('evolog').textContent=L.slice(-12).flatMap(r=>[`generation ${r.generation}: best ${r.ranking[0]?r.ranking[0].id+' '+Math.round(r.ranking[0].fitness):'-'}`,...r.events.map(e=>`   ${e.replaced} (${Math.round(e.fitness_was||0)}) replaced by a mutant of ${e.by} (${Math.round(e.parent_fitness)}): learning speed ${e.new_genes.lr}, randomness ${e.new_genes.ent}, plans ${Math.round(1/(1-e.new_genes.gamma))} moves ahead`)]).join('\n')||'No generations finished yet.'}
 $('subtitle').textContent=kit?(d.platform=='atari'?'Atari 2600 game. Every dot on the score graph is one real, full game it played.':'Arcade game. Every dot on the score graph is one real, full game it played.')
  :(d.kind=='atari'?'PPO + IMPALA-CNN, game and brain both on the GPU (CuLE). Judged on 64-game tests with sticky buttons.':'PPO + IMPALA-CNN brain on the GPU, real FBNeo cabinets on the CPU. Every dot is a real full game.');
 const ev=d.evals||[],lastE=ev[ev.length-1];
 let mean=null,best=null,corner=null;
 if(d.kind=='atari'){mean=lastE?lastE.mean:null;best=ev.length?Math.max(...ev.map(e=>e.max)):null;corner=lastE&&cfg.player_x_ram!=null?lastE.corner_right_pct+lastE.corner_left_pct:null;
  $('m1l').textContent='Latest test average';$('m1s').textContent=lastE?`64 games at ${big(lastE.step)} moves, middle score ${fmt(lastE.median)}`:'first test at 10M moves'}
 else{mean=d.last;best=d.best;const cs=d.corner_series||[];corner=cs.length?cs[cs.length-1][1]:null;$('m1l').textContent='Average, last 100 games';$('m1s').textContent=d.games_played?`${fmt(d.games_played)} real games played`:'no games yet'}
 $('m1').textContent=fmt(mean);$('m2').textContent=fmt(best);
 const tops=d.top10||[];$('m2s').textContent=d.kind=='atari'?'best single test game':(tops[0]?`game ${fmt(tops[0].game_no)}, at ${big(tops[0].step)} moves`:'');
 // "versus the bar" and "corner camping" only mean something for games that have them
 $('m3box').style.display=bar!=null?'':'none';
 if(bar!=null&&mean!=null){const r=mean/bar*100;$('m3').innerHTML=`<span class="${r>=100?'good':r>=70?'warn':'bad'}">${fmt(r)}%</span>`;$('m3s').textContent=`${cfg.bar_label||'bar'} ~${fmt(bar)}`}
 else{$('m3').textContent='—';$('m3s').textContent=cfg.bar_label||'no bar set'}
 const cs=d.kind=='atari'?ev.filter(e=>cfg.player_x_ram!=null).map(e=>[e.step,e.corner_right_pct+e.corner_left_pct]):(d.corner_series||[]);
 $('m4box').style.display=corner!=null?'':'none';$('cornerpanel').style.display=cs.length?'':'none';$('grid').classList.toggle('nocorner',!cs.length);
 if(corner!=null){$('m4').innerHTML=`<span class="${corner<20?'good':corner<40?'warn':'bad'}">${fmt(corner,1)}%</span>`;$('m4s').textContent=corner<20?'moving about, not camping':corner<40?'leaning on the corners':'camping!'}
 $('m5').textContent=fmt(st.sps);$('m5s').textContent=d.live?'live':(cap(st.state)||'not started');
 const tgt=st.target_steps||cfg.target||0,step=st.step||d.step||0;$('m6').textContent=big(step);
 $('prog').style.width=tgt?Math.min(100,step/tgt*100)+'%':'0';$('prog').parentNode.style.display=tgt?'':'none';
 let eta='';if(d.live&&st.sps>0&&tgt>step){const h=(tgt-step)/st.sps/3600;eta=`, about ${h<1?Math.round(h*60)+' min':h.toFixed(1)+' hours'} to go`}
 $('m6s').innerHTML=tgt?`of ${big(tgt)}${eta}`:'keeps going until you press Stop';
 const vtxt=[],how=kit?"Press Train on the kit's page":'Double-click TRAIN in its folder';
 if(!d.live)vtxt.push(st.state?`<b class="warn">Not training</b> (${esc(st.state)}). ${how} to carry on.`:`<b class="quiet">Set up, never trained.</b> ${how} to start.`);
 else vtxt.push('<b class="good">Training live.</b>');
 if(bar!=null&&mean!=null)vtxt.push(mean>=bar?`<b class="good">Beating the bar</b> (${fmt(mean)} vs ${fmt(bar)}).`:`${fmt(bar-mean)} points short of the bar.`);
 if(corner!=null&&corner>=40)vtxt.push('<b class="bad">Camping warning</b>: it spends a lot of time parked in a corner.');
 $('verdict').innerHTML=vtxt.join(' &nbsp; ');
 // the score graph
 const s1=[];let leg='';
 if(d.kind=='atari'){s1.push({pts:ev.map(e=>[e.step,e.mean]),hi:ev.map(e=>[e.step,e.max]),lo:ev.map(e=>[e.step,e.min]),band:css('--c-band'),color:css('--s-avg'),w:2.5,mark:1});
  leg='<span><i class="key" style="background:var(--s-avg)"></i>test average (shaded: worst to best game)</span>'}
 else{s1.push({pts:d.scatter||[],dots:1,color:css('--c-dots')});s1.push({pts:d.rolling||[],color:css('--s-avg'),w:2.5});
  leg='<span><i class="key" style="background:var(--teal)"></i>each real game</span><span><i class="key" style="background:var(--s-avg)"></i>average of the last 100</span>'}
 $('lg1').innerHTML=leg+(bar!=null?'<span><i class="key" style="background:var(--c-bar)"></i>bar to beat</span>':'');
 const toH=XMODE=='hours'?hoursOf(d.hours_map):null;      // ALFRED 23 Oct 2026: Moves / Hours along the bottom
 $('xsw_m').className=toH?'':'on';$('xsw_h').className=toH?'on':'';$('xsw_h').disabled=!d.hours_map||d.hours_map.length<2;
 if(toH)s1.forEach(s=>['pts','hi','lo'].forEach(k=>{if(s[k])s[k]=s[k].map(p=>[toH(p[0]),p[1]])}));
 chart($('c1'),s1,Object.assign({bar:bar,empty:'No games yet - press Train'},toH?{xticks:hourTicks,xfmt:hrs}:{}));
 const th=d.train_hours!=null?d.train_hours:d.wall_hours;
 $('c1note').textContent=(toH?'Left to right: hours of training (the time it was stopped is left out). ':'Left to right: moves of practice. ')+(th?`${th.toFixed(1)} hours of training so far.`:'');
 $('c2t').textContent=d.kind=='atari'?'Average and middle score':'Last 500 games';$('c2s').textContent=d.kind=='atari'?'orange average, green middle score':(bar!=null?'how the scores spread out (green beats the bar)':'how the scores spread out');
 hist($('c2'),d.recent_scores||null,bar);
 if(d.kind=='atari'&&!d.recent_scores){const [g,W,H]=setup($('c2'));g.clearRect(0,0,W,H);if(lastE){chart($('c2'),[{pts:ev.map(e=>[e.step,e.median]),color:css('--lime'),mark:1},{pts:ev.map(e=>[e.step,e.mean]),color:css('--s-avg'),mark:1}],{bar:bar})}}
 if(cs.length)chart($('c3'),[{pts:cs,color:css('--pink'),w:2}],{ymin:0,ymax:100,pct:1,empty:'Not measured for this game'});
 const ts=d.train_series||{};chart($('c4'),[{pts:ts.entropy||[],color:css('--violet')},{pts:(ts.expl_var||[]).map(p=>[p[0],Math.max(-1,p[1])]),color:css('--teal')}],{ymin:(ts.expl_var||[]).some(p=>p[1]<0)?-1:0,empty:'No training yet'});
 chart($('c5'),[{pts:ts.sps||[],color:css('--teal'),w:1.5}],{ymin:0,empty:''});
 if(d.kind=='atari'){$('tabletitle').textContent='Every test';$('tablesub').textContent='64 full games each';
  $('tbl').innerHTML='<tr><th>Moves</th><th>Average</th><th>Middle</th><th>Best</th><th>Worst</th>'+(cfg.player_x_ram!=null?'<th>Corner %</th>':'')+'</tr>'+[...ev].reverse().map(e=>`<tr><td>${big(e.step)}</td><td>${fmt(e.mean)}</td><td>${fmt(e.median)}</td><td>${fmt(e.max)}</td><td>${fmt(e.min)}</td>${cfg.player_x_ram!=null?`<td>${fmt(e.corner_right_pct+e.corner_left_pct,1)}</td>`:''}</tr>`).join('')}
 else{$('tabletitle').textContent='Hall of fame';$('tablesub').textContent='its 10 best real games';
  $('tbl').innerHTML='<tr><th>#</th><th>Score</th><th>Game</th><th>At move</th></tr>'+tops.map((t,i)=>`<tr><td>${i+1}</td><td><b>${fmt(t.score)}</b></td><td>${fmt(t.game_no)}</td><td>${big(t.step)}</td></tr>`).join('')}
 const con=$('console');const atBottom=con.scrollTop+con.clientHeight>=con.scrollHeight-10;con.textContent=(d.console||[]).join('\n');if(atBottom)con.scrollTop=con.scrollHeight;
 $('updated').textContent='Updated '+new Date().toLocaleTimeString()}
async function sys(){try{const s=await getJSON('/api/system');const m=(id,v,t)=>{$(id).style.width=(v||0)+'%';$(id+'t').textContent=t};
 m('gpu',s.gpu,s.gpu!=null?`${fmt(s.gpu)}%  ${s.gpu_temp||''}°C  ${s.gpu_power?fmt(+s.gpu_power)+' W':''}`:'n/a');m('vram',s.vram,s.vram_text||'n/a');m('cpu',s.cpu,s.cpu!=null?fmt(s.cpu)+'%':'n/a');m('ram',s.ram,s.ram_text||'n/a');$('gpuname').textContent=s.gpu_name||''}catch(e){}}
async function tick(){try{await loadList();await load();$('conn').innerHTML='<i class="dot live"></i>Live'}catch(e){$('conn').innerHTML='<i class="dot"></i>Report stopped'}}
$('watchbtn').onclick=async()=>{const b=$('watchbtn');b.textContent='Opening…';try{const r=await getJSON('/api/watch?id='+encodeURIComponent(current));b.textContent=r.ok?'▶ Opening the window…':'No watch for this one'}catch(e){b.textContent='Failed'}setTimeout(()=>b.innerHTML='&#9654; Watch it play',4000)};
setTheme();setInterval(tick,5000);setInterval(sys,3000);setInterval(()=>$('clock').textContent=new Date().toLocaleTimeString(),1000);
window.onresize=()=>load();window.onhashchange=()=>{const h=decodeURIComponent(location.hash.slice(1));if(h&&h!=current){current=h;tabs();load()}};tick();sys();
</script></body></html>"""

if __name__ == "__main__":
    main()

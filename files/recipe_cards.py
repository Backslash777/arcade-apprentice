"""Arcade Apprentice recipe cards: safe JSON export/import, with no ROMs or private PC data."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

FORMAT_VERSION = 1
KIT_VERSION = "1.2"
MAX_RECIPE_BYTES = 2_000_000
LENGTHS = {"taster": 5e6, "overnight": 100e6, "forever": 1e12, "more5": 5e6}

# Internal dial keys only. Values are clamped to the ranges offered by the page.
DIAL_RULES = {
    "reward_style": ("choice", ("log", "equal", "raw")),
    "life_penalty": ("float", 0.0, 2.0),
    "curiosity": ("float", 0.003, 0.03),
    "gamma": ("float", 0.97, 0.999),
    "sticky": ("bool",),
    "explorer": ("float", 0.0, 3.0),
    "ignore_losses": ("bool",),
    "learning_rate": ("float", 0.0001, 0.0005),
    "length": ("choice", ("taster", "more5", "overnight", "forever")),
    "go_explore": ("float", 0.0, 0.75),
    "brain_size": ("choice", ("auto", "normal", "large")),
    "learn_from_me": ("float", 0.0, 2.0),
    "learn_style": ("choice", ("practice", "evolution")),
    "learn_from_best": ("bool",),
    "best_study_times": ("int", 1, 5),
    "lives_at_game_over": ("bool",),
    "miss_price": ("float", 0.0, 0.2),
    "practice": ("float", 0.0, 0.5),
}
DEFAULTS = {
    "reward_style": "log", "life_penalty": 0.0, "curiosity": 0.01, "gamma": 0.99,
    "sticky": True, "explorer": 0.0, "ignore_losses": False, "learning_rate": 0.00025,
    "length": "overnight", "go_explore": 0.0, "brain_size": "auto", "learn_from_me": 1.0,
    "learn_style": "practice", "learn_from_best": False, "best_study_times": 1,
    "lives_at_game_over": False, "miss_price": 0.0, "practice": 0.0,
}


def _ai_folder(projects: Path, pid: str) -> Path:
    if not pid or "/" in pid or "\\" in pid or ".." in pid:
        raise ValueError("That isn't an AI in this kit.")
    folder = (projects / pid).resolve()
    if folder.parent != projects.resolve() or not (folder / "project.json").exists():
        raise ValueError("That isn't an AI in this kit.")
    return folder


def _config(projects: Path, pid: str) -> dict:
    return json.loads((_ai_folder(projects, pid) / "project.json").read_text(encoding="utf-8-sig"))


def _rom_fingerprint(roms: Path, cfg: dict) -> str:
    name = cfg.get("rom_file")
    if not name:
        raise ValueError("This AI has no ROM filename recorded, so its fingerprint can't be checked.")
    rom = (roms / name).resolve()
    if roms.resolve() not in rom.parents or not rom.is_file():
        raise ValueError("The ROM file isn't in the kit's ROMs folder any more. Put it back and Scan again.")
    digest = hashlib.sha256()
    with rom.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _clean_setting(key, value):
    rule = DIAL_RULES[key]
    kind = rule[0]
    if kind == "choice":
        if value not in rule[1]:
            raise ValueError(f"{key} has an unknown option: {value!r}")
        return value
    if kind == "bool":
        return _bool(value)
    number = float(value)
    number = max(rule[1], min(rule[2], number))
    return int(round(number)) if kind == "int" else number


def _settings(cfg: dict) -> dict:
    return {key: _clean_setting(key, cfg.get(key, DEFAULTS[key])) for key in DIAL_RULES}


def _results(folder: Path) -> dict:
    run = folder / "files" / "run"
    status = {}
    try:
        status = json.loads((run / "status.json").read_text(encoding="utf-8-sig"))
    except Exception:
        pass
    latest = {}
    try:
        import csv
        with (run / "train_log.csv").open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if rows:
            latest = rows[-1]
    except Exception:
        pass

    def number(value):
        try:
            value = float(value)
            return value if value == value and abs(value) != float("inf") else None
        except (TypeError, ValueError):
            return None

    return {
        "average_score": number(status.get("score_mean100", latest.get("score_mean100"))),
        "best_score": number(status.get("best_game", latest.get("best_game"))),
        "best_average": number(status.get("best_mean100")),
        "shot_accuracy": number(status.get("accuracy", latest.get("accuracy"))),
        "total_moves_trained": number(status.get("step", latest.get("step"))) or 0,
        "moves_per_second": number(status.get("sps", latest.get("sps"))),
    }


def _pc_specs(cfg: dict, results: dict) -> dict:
    hw = cfg.get("hardware") or {}
    # Deliberately allowlisted: no PC name, username, serial, path, IP or other identifier.
    return {
        "cpu_model": hw.get("cpu_name"),
        "cpu_cores": hw.get("cpu_cores", hw.get("cpu_threads")),
        "cpu_threads": hw.get("cpu_threads"),
        "ram_gb": hw.get("ram_gb"),
        "gpu_model": hw.get("gpu_name"),
        "windows_version": hw.get("os"),
        "measured_moves_per_second": results.get("moves_per_second"),
    }


def _filename(game_name: str) -> str:
    game = re.sub(r"[^\w \-]", "", game_name).strip().replace(" ", "-") or "game"
    return f"{game}-{time.strftime('%Y-%m-%d')}.aarecipe"


def build_recipe(projects: Path, roms: Path, pid: str, notes="", include_specs=False) -> dict:
    folder = _ai_folder(projects, pid)
    cfg = _config(projects, pid)
    results = _results(folder)
    recipe = {
        "format_version": FORMAT_VERSION,
        "kit_version": KIT_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "game": {
            "name": cfg.get("game_name", pid),
            "game_id": cfg.get("game_id"),
            "platform": cfg.get("platform", "atari"),
            "rom_sha256": _rom_fingerprint(roms, cfg),
        },
        "settings": _settings(cfg),
        "results": {k: v for k, v in results.items() if k != "moves_per_second"},
        "notes": str(notes or "")[:4000],
    }
    if include_specs:
        recipe["pc_specs"] = _pc_specs(cfg, results)
    return recipe


def preview(projects: Path, roms: Path, pid: str, notes="", include_specs=False) -> dict:
    recipe = build_recipe(projects, roms, pid, notes, include_specs)
    text = json.dumps(recipe, indent=2, ensure_ascii=False) + "\n"
    return {"ok": True, "recipe": recipe, "json": text, "filename": _filename(recipe["game"]["name"])}


def _parse(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_RECIPE_BYTES:
        raise ValueError("The recipe must be valid JSON and no bigger than 2 MB.")
    recipe = json.loads(value)
    if not isinstance(recipe, dict):
        raise ValueError("The recipe must be a JSON object.")
    return recipe


def _check_recipe(recipe: dict):
    if recipe.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported recipe format. This kit accepts format_version {FORMAT_VERSION} only.")
    if not isinstance(recipe.get("game"), dict) or not isinstance(recipe.get("settings"), dict):
        raise ValueError("This doesn't look like an Arcade Apprentice recipe card.")


def export(projects: Path, roms: Path, pid: str, text: str, share=False) -> dict:
    recipe = _parse(text)
    _check_recipe(recipe)
    cfg = _config(projects, pid)
    if recipe["game"].get("rom_sha256") != _rom_fingerprint(roms, cfg):
        raise ValueError("The recipe is for a different ROM, so it wasn't saved.")
    unknown = sorted(set(recipe["settings"]) - set(DIAL_RULES))
    if unknown:
        raise ValueError("The preview has changed and now contains unknown settings. Preview it again.")
    recipe["settings"] = {key: _clean_setting(key, value) for key, value in recipe["settings"].items()}
    exact = json.dumps(recipe, indent=2, ensure_ascii=False) + "\n"
    if exact != text:
        raise ValueError("The preview text has changed. Preview it again before saving.")
    destination = projects.parent / "Recipe Cards"
    destination.mkdir(exist_ok=True)
    path = destination / _filename(recipe["game"]["name"])
    if path.exists():
        path = destination / f"{path.stem}-{time.strftime('%H%M%S')}{path.suffix}"
    if path.exists():
        raise ValueError("A recipe with that name already exists. Export it again in a moment.")
    path.write_text(exact, encoding="utf-8")
    response = {"ok": True, "path": str(path), "filename": path.name,
                "msg": f"Saved {path.name} in Recipe Cards. No ROM was included."}
    if share:
        webbrowser.open("https://forum.arcadeapprentice.co.uk/composer?tag=recipes", new=1)
        if hasattr(os, "startfile"):
            os.startfile(str(destination))
        response["msg"] = f"Saved {path.name}. The Recipes forum and its folder are open - attach it there."
    return response


def import_preview(projects: Path, roms: Path, pid: str, value, override_fingerprint=False) -> dict:
    recipe = _parse(value)
    _check_recipe(recipe)
    cfg = _config(projects, pid)
    actual = _rom_fingerprint(roms, cfg)
    mismatch = recipe["game"].get("rom_sha256") != actual
    if mismatch and not override_fingerprint:
        return {"ok": False, "fingerprint_mismatch": True,
                "msg": "ROM fingerprint mismatch. This recipe came from a different game or a different copy of the ROM."}
    before = _settings(cfg)
    unknown = sorted(set(recipe["settings"]) - set(DIAL_RULES))
    cleaned = {}
    for key, value in recipe["settings"].items():
        if key in DIAL_RULES:
            cleaned[key] = _clean_setting(key, value)
    diff = [{"key": key, "before": before[key], "after": cleaned[key], "different": before[key] != cleaned[key]}
            for key in DIAL_RULES if key in cleaned and before[key] != cleaned[key]]
    return {"ok": True, "recipe": recipe, "diff": diff, "ignored_keys": unknown,
            "fingerprint_mismatch": mismatch,
            "msg": f"{len(diff)} dial{'s' if len(diff) != 1 else ''} will change. Tick only the ones you want."}


def _backup(folder: Path, label: str) -> str:
    dev = folder / "files" / "_dev"
    dev.mkdir(exist_ok=True)
    name = f"project.json.bak-{time.strftime('%Y-%m-%d-%H%M%S')}-{label}"
    shutil.copy2(folder / "project.json", dev / name)
    return name


def import_apply(projects: Path, roms: Path, pid: str, value, keys, override_fingerprint=False) -> dict:
    preview_data = import_preview(projects, roms, pid, value, override_fingerprint)
    if not preview_data["ok"]:
        return preview_data
    allowed = {row["key"] for row in preview_data["diff"]}
    chosen = set(keys) & allowed
    if not chosen:
        return {"ok": False, "msg": "No changed dials were ticked, so nothing was applied."}
    folder = _ai_folder(projects, pid)
    cfg = _config(projects, pid)
    backup = _backup(folder, "recipe-import")
    recipe = preview_data["recipe"]
    for key in chosen:
        cfg[key] = _clean_setting(key, recipe["settings"][key])
    if "length" in chosen:
        cfg["total_steps"] = LENGTHS.get(cfg.get("length"), cfg.get("total_steps", 1e12))
    cfg["changed"] = time.strftime("%Y-%m-%d %H:%M")
    (folder / "project.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    with (folder / "files" / "run" / "console.log").open("a", encoding="utf-8") as f:
        f.write(f"RECIPE IMPORT {cfg['changed']}: {', '.join(sorted(chosen))}; backup={backup}\n")
    return {"ok": True, "backup": backup, "applied": sorted(chosen),
            "msg": f"Applied {len(chosen)} dial{'s' if len(chosen) != 1 else ''}. Undo is one click, backup {backup}."}


def undo(projects: Path, pid: str, backup: str) -> dict:
    folder = _ai_folder(projects, pid)
    if not isinstance(backup, str) or not backup.startswith("project.json.bak-") or backup != Path(backup).name:
        raise ValueError("That isn't a recipe import backup.")
    source = folder / "files" / "_dev" / backup
    if not source.is_file():
        raise ValueError("That undo backup no longer exists.")
    _backup(folder, "before-recipe-undo")
    shutil.copy2(source, folder / "project.json")
    with (folder / "files" / "run" / "console.log").open("a", encoding="utf-8") as f:
        f.write(f"RECIPE IMPORT UNDONE: restored {backup}\n")
    return {"ok": True, "msg": "Import undone. Your previous settings are back."}

# Arcade Apprentice

Teach an AI to play Atari 2600 and arcade games from scratch, on your own Windows PC. No cloud training account, no command-line setup and no bundled games.

**[Download v1.2](https://arcadeapprentice.co.uk/download/Arcade-Apprentice-v1.2.zip)** · [Project website](https://arcadeapprentice.co.uk/) · [Community forum](https://forum.arcadeapprentice.co.uk/)

## What it does

- Trains game-playing AIs locally with PPO.
- Supports Atari 2600 plus MAME/FBNeo arcade games.
- Includes score-finding tools for arcade games that are not already configured.
- Adapts the training workload to your graphics card, processor and memory.
- Lets you watch the latest AI play and inspect its report card.
- Exports and imports shareable **recipe cards**: training dials, results and a ROM fingerprint, never the ROM itself.

## Quick start

1. Download and extract [Arcade Apprentice v1.2](https://arcadeapprentice.co.uk/download/Arcade-Apprentice-v1.2.zip) somewhere you can write to, such as Desktop or Documents. Do not use Program Files.
2. Double-click **`START HERE.bat`**.
3. On first run, the kit downloads its own private Python and AI libraries. Allow roughly **4 GB** and **5–15 minutes**, depending on your connection and PC.
4. Put game files you are legally entitled to use in **`ROMs - put your games here`**, then scan that folder from the kit's page.
5. Choose a game and training settings, create your AI and press **Train**.

Closing the browser does not stop training. Use **`REOPEN THE PAGE.bat`** to return to the dashboard, or **`CLOSE THE KIT.bat`** to save and stop the kit.

## Requirements

- Windows 10 or 11.
- 8 GB RAM minimum; 16 GB recommended.
- About 6 GB disk space after setup.
- NVIDIA, supported newer AMD Radeon, or many AMD/Intel GPUs through DirectML. CPU-only training works but is slower.

The download is approximately 53 MB. It contains no trained AIs, ROMs or private Python runtime.

## Recipe cards

Open an AI's **Settings → Recipe cards** to preview exactly what will be exported. Cards use the `.aarecipe` format and contain the selected dials, results and the game's SHA-256 fingerprint.

You can import only the dials you want. The kit rejects unknown keys, clamps out-of-range values, warns about ROM mismatches, backs up your existing settings and provides one-click undo. PC specifications are optional; PC names, usernames, serial numbers and file paths are not exported.

Share cards and results in the [Recipes forum](https://forum.arcadeapprentice.co.uk/t/recipes).

## Help and feedback

- [Help forum](https://forum.arcadeapprentice.co.uk/t/help)
- [Suggestions](https://forum.arcadeapprentice.co.uk/t/suggestions)
- [Bugfixes](https://forum.arcadeapprentice.co.uk/t/bugfixes)
- In the kit, use **Make a problem report** to create a diagnostic text file.

## Legal and privacy

Use only game files you are entitled to use. ROMs and trained AIs stay on your computer unless you choose to share something yourself. Recipe cards contain no ROM contents.

Copyright © 2026 Jonathan Terrington. All rights reserved unless otherwise stated.

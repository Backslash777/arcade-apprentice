ARCADE APPRENTICE v1.2
Teach an AI to play Atari 2600 and arcade games from scratch - on your own computer.  (c) 2026 Jonathan Terrington

0. Extract this folder anywhere you can write to (Desktop, Documents, Downloads...) - not inside Program Files.
1. Double-click  START HERE.bat
   If Windows shows a blue "Windows protected your PC" box, click More info -> Run anyway (normal for downloaded files).
   The first time, it downloads its own private copy of Python and the AI libraries (about 4 GB, 5-15 minutes).
   Nothing is installed on your system. To remove it all later, double-click UNINSTALL.bat.
2. A page opens in your browser. It checks your computer (graphics card, processor, memory) and sizes the job to it.
   Works with NVIDIA graphics cards, newer AMD Radeon cards and many AMD/Intel cards via DirectML - or just the
   processor if there's no usable card (slower, still learns).
3. Put your game files in the folder  "ROMs - put your games here":
     - Atari 2600 games (.bin / .a26, or zipped)
     - Arcade games (.zip, MAME/FBNeo sets). Ready to go: Space Invaders, Ms. Pac-Man, Donkey Kong, Track & Field, Pac-Man.
       ANY OTHER arcade game: press "Find the score" next to it on the page. You play it for a few minutes and
       now and then type the score you can see; the kit finds where the game keeps it, and from then on it's
       a normal game you can train.
       Fighting games with health bars (Street Fighter II...): same window, but press H / J / K / U to say
       what just happened (round start / you got hit / you hit them / nothing happened). Street Fighter II
       is already set up this way.
   Games are recognised by their fingerprint, so the file name doesn't matter. Sub-folders are fine.
   Press "Scan the ROMs folder" on the page. Your games are sorted into three lists: Atari 2600, Arcade - ready
   to train (score: automatic) and Arcade - score needs finding (you show it the score first, once).
4. Pick a game, choose how the AI is rewarded, choose how long it practises, press Create my AI.
5. Press Train. A training window opens - leave it running. Press Stop any time; it saves and carries on later.
   Closed the browser by accident? Double-click REOPEN THE PAGE.bat - it brings the page back (training carries on).
   Shut the page without pressing "Close the kit"? Double-click CLOSE THE KIT.bat - it stops any training
   (each AI saves first) and closes everything the kit left open.
6. Press Watch it play to see the latest brain in action. Open the report card for live charts.

RECIPE CARDS (new in v1.2):
   Open an AI's Settings and look for Recipe cards.
   - Preview & export settings shows exactly what goes in the .aarecipe file first: every dial, the ROM's
     SHA-256 fingerprint and your results. The ROM itself is never included. Notes are optional, and PC specs are
     off unless you tick them (never PC name, username, serial numbers or file paths).
   - Share to forum saves the card, opens the Recipes section of forum.arcadeapprentice.co.uk and opens its folder.
     Attach the card there. No forum login or API token is stored in the kit.
   - Import settings checks the format and ROM fingerprint, warns on a mismatch, ignores unknown keys and clamps
     values. Tick only the dials you want. Your current settings are backed up first, so Undo is one click.
   - Cards are saved in "Recipe Cards" next to START HERE.bat.

HELP: press Help at the top of the page - it explains everything and plays the help video (it's a song).
ABOUT / BUY JONNY A COFFEE: press About at the top of the page.

SOMETHING NOT WORKING?  Press "Make a problem report" on the page (or double-click MAKE PROBLEM REPORT.bat).
It makes one text file next to START HERE.bat describing your PC and what went wrong - send that to Jonny.

TO REMOVE IT: double-click UNINSTALL.bat. It clears out everything the kit downloaded (even if setup had to put
it in your AppData folder because this folder sits deep inside others) and the scratch files it used. It asks
whether to keep your trained AIs and games or delete the lot.

What you need: Windows 10/11, 8 GB memory (16 GB better), about 6 GB of disk space.
Only use game files you are entitled to.

How it learns (in one breath): it sees the screen, guesses which button to press, and after every few
hundred moves a "coach" looks back at what happened and nudges it towards the moves that scored more than
expected. Repeat a few million times. That's PPO - the same family of methods used to train game-playing
AIs in research labs.

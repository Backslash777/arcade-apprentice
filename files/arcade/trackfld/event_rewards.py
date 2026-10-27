"""Contract v3 event reward components (trackfld-run-macro-v3).

Jonny's spec of 2026-10-05 (see CONTRACT_V3_DRAFT.md): the native game score
is telemetry-only under v3; these components ARE the learning reward. Only
detectors backed by validated telemetry ship enabled here:

- stall penalty (100m): agent's own action stream + frame clock.
- qualifier bonus: the env's ten-frame-confirmed event transition.
- world record bonus: display_score against the lineage benchmark file.
- foul board / board precision (long jump): the run-up counter, foul enforced
  by the game at exactly 721 units (validated 10/10, evidence/telemetry-hunt).

Pending detectors (GO signal, angles, javelin set) and blocked events
(hurdles, hammer, high jump) are NOT implemented here; they arrive after
their phase-1/phase-2 hunts. Constants carry Jonny's exact numbers.
"""
import json
import math
import os

# Jonny's spec values (raw reward units; there is no native backbone in v3).
STALL_PENALTY_PER_SECOND = 1.0
QUALIFIER_BONUS = 20.0
WORLD_RECORD_BONUS = 50.0
FOUL_BOARD_PENALTY = 50.0
BOARD_PRECISION_BONUS = 10.0

# Validated telemetry constants.
FOUL_LINE_UNITS = 721          # game enforces the foul at exactly 721 (10/10)
BOARD_WINDOW = (710, 721)      # Jonny ruling 19:49: jump frame in [710, 721)
FINISH_LINE_UNITS = 1720       # 100m counter parks at 1720-1722 past the finish
DEFAULT_WR_BENCHMARK = 40800   # lineage best at v3 adoption (Jonny: "previous generations")

EVENT_100M = 0
EVENT_LONG_JUMP = 1
RUN_SPAM = 8                   # ACTION_NAMES index of the macro
JUMP_THROW = 4                 # ACTION_NAMES index of jump_throw
MODE_ACTIVE = 3


class StallPenalty:
    """-1.0/sec while the macro is off during a live 100m (before the finish).

    Per-frame: -(1/fps) for every frame the decision action is not run_spam,
    gated to event 0 in active play before the finish line. The end-of-race
    dive (jump_throw) costs a couple of frames of tax by design; waiting for
    the event transition after finishing is free (distance >= 1720).
    Episodes currently begin AT the gun, so no pre-GO window exists; the GO
    probe (phase 1) revisits this gate if reset moves earlier.
    """

    name = "stall"

    def __init__(self, fps):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("stall penalty fps must be positive and finite")
        self.per_frame = -STALL_PENALTY_PER_SECOND / float(fps)
        self.reset()

    def reset(self):
        self.total = 0.0

    def frame(self, native, action):
        if native["mode"] != MODE_ACTIVE or native["event_index"] != EVENT_100M:
            return 0.0
        if native["race_distance_units"] >= FINISH_LINE_UNITS:
            return 0.0
        # "running" = the macro, or any raw action with a run bit (raw actions
        # 1..3 and combos 5..7). Jonny's no-macro experiment uses raw buttons.
        if action == RUN_SPAM or (isinstance(action, int) and action & 3):
            return 0.0
        self.total += self.per_frame
        return self.per_frame


class QualifierBonus:
    """+20 per confirmed event advance (beat the qualifier, move on)."""

    name = "qualifier"

    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0

    def on_confirmed_transition(self):
        self.total += QUALIFIER_BONUS
        return QUALIFIER_BONUS


class WorldRecordBonus:
    """+50 once per episode when display_score beats the lineage benchmark.

    The benchmark is shared across worker processes through a locked JSON
    file and updates immediately when beaten (Jonny's moving goalpost). The
    lock serializes simultaneous crossings: the loser re-reads the raised
    benchmark and is not paid.
    """

    name = "world_record"

    def __init__(self, benchmark_path):
        self.path = str(benchmark_path)
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            self._write(DEFAULT_WR_BENCHMARK)
        self.benchmark = self._read()
        self.reset()

    def _read(self):
        with open(self.path, "r", encoding="utf-8") as handle:
            return int(json.load(handle)["benchmark"])

    def _write(self, value):
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump({"benchmark": int(value)}, handle)

    def _locked_crossing(self, score):
        # Returns (paid, benchmark): pay only if the score still beats the
        # benchmark under the lock, then raise the benchmark to the score.
        import msvcrt
        with open(self.path, "r+", encoding="utf-8") as handle:
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                handle.seek(0)
                benchmark = int(json.load(handle)["benchmark"])
                if score <= benchmark:
                    return False, benchmark
                handle.seek(0)
                handle.truncate()
                json.dump({"benchmark": int(score)}, handle)
                handle.flush()
                return True, int(score)
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

    def reset(self):
        self.total = 0.0
        self.paid_this_episode = False

    def frame(self, native, action):
        score = int(native["display_score"])
        if score <= self.benchmark:
            return 0.0
        # Always push the benchmark to the true maximum (even after this
        # episode's payout) so the file tracks the lifetime best; the +50
        # itself pays only once per episode. Review item 19.
        paid, self.benchmark = self._locked_crossing(score)
        if not paid or self.paid_this_episode:
            return 0.0
        self.paid_this_episode = True
        self.total += WORLD_RECORD_BONUS
        return WORLD_RECORD_BONUS


class _LongJumpAttemptTracker:
    """Per-attempt state machine for the event-1 run-up counter.

    Validated facts (evidence/lj-takeoff-*.json, 2026-10-06):
    - the counter holds the stale 100m finish (~1722) for ~74 decisions on
      entry, then drops to ~0 to start the run-up (drop = new attempt);
    - the counter advances ~11 units per decision during the run-up, so
      Jonny's 11-unit window is about one decision wide at press time;
    - press-to-takeoff latency is ~10-11 units: a press registering at 711
      fouls at 721; the counter FREEZES at the takeoff point during flight;
    - a freeze AT 721 is a real foul (game stops the runner at the line).
    Semantics: foul = a rise THROUGH 721 in a live run-up; precision = the
    freeze (actual takeoff point) in [710, 721) with no foul this attempt.
    """

    def __init__(self):
        self.previous_counter = None
        self.reset_attempt()

    def reset_attempt(self):
        self.live = False           # a sub-721 run-up reading has been seen
        self.pressed = False
        self.press_counter = None
        self.static_frames = 0
        self.freeze_value = None
        self.fouled = False

    def frame(self, native, action):
        """Returns (counter, crossed, just_froze). Tracks press/freeze state."""
        counter = int(native["race_distance_units"])
        if native["mode"] != MODE_ACTIVE or native["event_index"] != EVENT_LONG_JUMP:
            self.previous_counter = None
            self.reset_attempt()
            return None, False, False
        previous = self.previous_counter
        if previous is not None and counter < previous:
            self.reset_attempt()    # counter reset between attempts
        if counter < FOUL_LINE_UNITS:
            self.live = True
        crossed = (self.live and previous is not None and previous < counter
                   and previous < FOUL_LINE_UNITS <= counter)
        if crossed:
            self.fouled = True
        if action == JUMP_THROW and not self.pressed and self.live:
            self.pressed = True
            self.press_counter = counter
        just_froze = False
        if self.pressed and self.freeze_value is None:
            if counter == previous:
                self.static_frames += 1
                if self.static_frames >= 3:
                    self.freeze_value = counter
                    just_froze = True
            else:
                self.static_frames = 0
        self.previous_counter = counter
        return counter, crossed, just_froze


class LongJumpBoard:
    """Foul -50 and board precision +10 sharing one attempt tracker.

    foul_board: the counter rises through 721 during a live run-up (never on
    entry residue, never on a parked freeze). board_precision: the takeoff
    freeze lands in [710, 721) and no foul this attempt — a foul pays NOTHING,
    killing the old +10-on-a-foul mismatch (probe 2026-10-06: presses
    registering 711+ foul at 721 yet collected precision under the old rule).
    """

    name = "long_jump_board"

    def __init__(self):
        self.tracker = _LongJumpAttemptTracker()
        self.reset()

    def reset(self):
        self.ledger = {"foul_board": 0.0, "board_precision": 0.0}
        self.precision_paid = False
        self.tracker.previous_counter = None
        self.tracker.reset_attempt()

    @property
    def total(self):
        return sum(self.ledger.values())

    def frame(self, native, action):
        counter, crossed, just_froze = self.tracker.frame(native, action)
        if counter is None:
            return 0.0
        reward = 0.0
        if crossed:
            self.ledger["foul_board"] -= FOUL_BOARD_PENALTY
            reward -= FOUL_BOARD_PENALTY
        if self.tracker.freeze_value is None:
            self.precision_paid = False   # attempt still open / new attempt re-arm
        if (just_froze and not self.tracker.fouled and not self.precision_paid
                and BOARD_WINDOW[0] <= self.tracker.freeze_value < BOARD_WINDOW[1]):
            self.precision_paid = True
            self.ledger["board_precision"] += BOARD_PRECISION_BONUS
            reward += BOARD_PRECISION_BONUS
        return reward


class EventRewardMachine:
    """Sums the validated v3 components per frame; env adds qualifier payouts."""

    def __init__(self, fps=60.0, benchmark_path=None):
        self.components = [
            StallPenalty(fps),
            QualifierBonus(),
            WorldRecordBonus(benchmark_path) if benchmark_path is not None else None,
            LongJumpBoard(),
        ]
        self.components = [component for component in self.components if component is not None]

    def reset(self, native):
        for component in self.components:
            component.reset()

    def step_frame(self, native, action):
        return sum(component.frame(native, action)
                   for component in self.components if hasattr(component, "frame"))

    def on_confirmed_transition(self):
        for component in self.components:
            if isinstance(component, QualifierBonus):
                return component.on_confirmed_transition()
        return 0.0

    def totals(self):
        out = {}
        for component in self.components:
            if hasattr(component, "ledger"):
                out.update(component.ledger)
            else:
                out[component.name] = component.total
        return out

    def total(self):
        return sum(self.totals().values())

    def benchmark(self):
        for component in self.components:
            if isinstance(component, WorldRecordBonus):
                return component.benchmark
        return None

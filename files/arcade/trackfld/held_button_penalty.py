"""Learning-only soft cost for carrying controls across decision boundaries.

This does not alter controls, manufacture releases, or preserve the native-score
objective: it is an explicit additional training preference for releasing inputs.
"""
import math
from numbers import Integral


HOLD_PENALTY_CONTRACT = (
    "negative weight per overlapping scoped raw button across consecutive decisions "
    "(run bits always; jump bit only when include_jump); "
    "plus one unreleased-run unit when both decisions contain a run button "
    "but their run bits do not overlap; all events; reset on game reset"
)


class HeldButtonPenalty:
    def __init__(self, weight=0.0, include_jump=False):
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("hold penalty weight must be finite and in [0, 1]")
        self.weight = float(weight)
        if not isinstance(include_jump, bool):
            raise ValueError("hold penalty include_jump must be a boolean")
        self.include_jump = include_jump
        self.reset()

    def reset(self):
        self.previous_action = self.last_held_mask = 0
        self.held_button_count = self.extra_unreleased_run_count = self.units = 0
        self.total_count = self.total_held_button_count = self.total_unreleased_run_count = 0
        self.reward = self.total = 0.0

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action <= 7:
            raise ValueError("hold penalty action must be a raw button combination 0..7")
        action = int(action)
        previous_run, current_run = self.previous_action & 3, action & 3
        self.last_held_mask = self.previous_action & action & (7 if self.include_jump else 3)
        self.held_button_count = self.last_held_mask.bit_count()
        self.extra_unreleased_run_count = int(
            bool(previous_run) and bool(current_run) and not (previous_run & current_run)
        )
        self.units = self.held_button_count + self.extra_unreleased_run_count
        self.reward = -self.weight * self.units
        self.total += self.reward
        self.total_count += self.units
        self.total_held_button_count += self.held_button_count
        self.total_unreleased_run_count += self.extra_unreleased_run_count
        self.previous_action = action
        return self.reward

"""Bounded potential shaping from verified P1 first-race distance, not inputs.

F(s,s') = gamma * Phi(s') - Phi(s). Phi is zero outside the 100m and
at terminal/truncated boundaries. Negative settling terms are intentional.
The 2048-unit normalizer is a conservative bound, not a metres conversion.
"""
import math

DISTANCE_NORMALIZER = 2048.0


class RaceProgressShaper:
    def __init__(self, weight=0.25, gamma=0.999):
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('shaping weight must be finite and in [0, 1]')
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError('shaping gamma must be finite and in (0, 1]')
        self.weight, self.gamma = float(weight), float(gamma)
        self.previous = self.total = self.positive_total = self.negative_total = 0.0
        self.last_distance = None

    def potential(self, native, done=False):
        if self.weight == 0 or done or native['mode'] != 3 or native['event_index'] != 0:
            return 0.0
        distance = int(native['race_distance_units'])
        if not 0 <= distance <= 2048:
            raise RuntimeError(f'100m distance out of verified range: {distance}')
        return self.weight * distance / DISTANCE_NORMALIZER

    def reset(self, native):
        self.previous = self.potential(native)
        self.total = self.positive_total = self.negative_total = 0.0
        self.last_distance = native['race_distance_units'] if self.weight and native['event_index'] == 0 else None

    def step(self, native, done=False):
        # Native-only WATCH must not depend on shaping-specific telemetry guards.
        if self.weight == 0:
            return 0.0
        active = not done and native['mode'] == 3 and native['event_index'] == 0
        distance = int(native['race_distance_units']) if active else None
        if active and self.last_distance is not None and distance < self.last_distance:
            raise RuntimeError('100m distance unexpectedly moved backwards; refusing fabricated shaping')
        current = self.potential(native, done)
        value = self.gamma * current - self.previous
        self.previous, self.last_distance = current, distance
        self.total += value
        self.positive_total += max(0.0, value)
        self.negative_total += min(0.0, value)
        return value

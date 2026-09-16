import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from . import callbacks


class FixedQNetwork(nn.Module):
    """Always prefer UP, then WAIT, for deterministic action tests."""

    def forward(self, features):
        values = torch.tensor(
            [10.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            dtype=torch.float32,
            device=features.device,
        )
        return values.repeat(features.shape[0], 1)


def corridor_state(bomb_timer):
    """Make UP walkable while a bomb threatens that destination."""
    field = np.full((7, 7), -1, dtype=np.int8)
    field[3, 3] = 0
    field[3, 2] = 0
    field[2, 2] = 0
    return {
        "round": 1,
        "step": 1,
        "field": field,
        "self": ("lsk_task4_agent", 0, False, (3, 3)),
        "others": [],
        "bombs": [((2, 2), bomb_timer)],
        "coins": [],
        "explosion_map": np.zeros_like(field),
    }


def timer_one_trap_state():
    """Make UP enter a timer-one blast with no available exit next turn."""
    state = corridor_state(bomb_timer=1)
    # The agent has just left its own bomb, so returning DOWN will be blocked.
    state["bombs"].append(((3, 3), 3))
    # RIGHT has a corner escape and is therefore the surviving alternative.
    state["field"][4, 3] = 0
    state["field"][4, 4] = 0
    return state


class ImmediateSafetyTests(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleNamespace(
            train=False,
            epsilon=0.0,
            policy_net=FixedQNetwork(),
        )

    def test_act_does_not_enter_timer_zero_blast(self):
        state = corridor_state(bomb_timer=0)
        self.assertEqual(callbacks.act(self.agent, state), "WAIT")

    def test_timer_one_blast_is_not_treated_as_immediate(self):
        state = corridor_state(bomb_timer=1)
        self.assertEqual(callbacks.act(self.agent, state), "UP")

    def test_timer_one_dead_end_is_rejected(self):
        state = timer_one_trap_state()
        self.assertEqual(callbacks.act(self.agent, state), "RIGHT")

    def test_wait_is_not_added_during_normal_escape(self):
        state = corridor_state(bomb_timer=3)
        features = np.zeros(callbacks.FEATURE_DIM, dtype=np.int8)
        features[0] = 1
        features[8] = 1
        features[9] = 1
        candidates = callbacks.valid_action_indices(features)
        self.assertEqual(candidates, [0])
        self.assertEqual(
            callbacks.survival_safe_action_indices(state, candidates),
            [0],
        )

    def test_safety_blast_passes_through_crates(self):
        field = np.zeros((7, 7), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        field[3, 2] = 1
        self.assertIn((3, 3), callbacks._real_blast_cells(field, (3, 1)))

    def test_checkpoint_features_keep_legacy_crate_blocking(self):
        field = np.zeros((7, 7), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        field[3, 2] = 1
        self.assertNotIn((3, 3), callbacks._blast_cells(field, (3, 1)))


if __name__ == "__main__":
    unittest.main()

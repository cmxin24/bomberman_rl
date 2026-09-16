import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import events as e

from . import callbacks, train
from agent_code.lsk_task4_agent import callbacks as legacy_callbacks


class HybridObjectiveTests(unittest.TestCase):
    def test_nearby_coin_has_priority_over_opponent(self):
        target, target_kind = callbacks._nearest_target(
            5,
            5,
            coins=[(8, 5)],
            crates=[],
            opponents=[(6, 5)],
        )
        self.assertEqual(target, (8, 5))
        self.assertEqual(target_kind, 1)

    def test_distant_coin_does_not_replace_opponent(self):
        target, target_kind = callbacks._nearest_target(
            1,
            1,
            coins=[(7, 1)],
            crates=[],
            opponents=[(3, 1)],
        )
        self.assertEqual(target, (3, 1))
        self.assertEqual(target_kind, 3)

    def test_collected_coin_matches_environment_score(self):
        reward = train.reward_from_events(
            SimpleNamespace(),
            [e.COIN_COLLECTED],
        )
        self.assertAlmostEqual(reward, 0.99)

    def test_unaligned_bomb_has_extra_cost(self):
        useful = np.zeros(callbacks.FEATURE_DIM, dtype=np.int8)
        useful[callbacks.BOMB_HITS_TARGET_INDEX] = 1
        wasteful = useful.copy()
        wasteful[callbacks.BOMB_HITS_TARGET_INDEX] = 0

        useful_reward = train.reward_from_events(
            SimpleNamespace(),
            [e.BOMB_DROPPED],
            old_features=useful,
        )
        wasteful_reward = train.reward_from_events(
            SimpleNamespace(),
            [e.BOMB_DROPPED],
            old_features=wasteful,
        )
        self.assertAlmostEqual(useful_reward - wasteful_reward, 0.15)

    def test_trapping_bomb_has_small_bonus(self):
        escapable = np.zeros(callbacks.FEATURE_DIM, dtype=np.int8)
        escapable[callbacks.BOMB_HITS_TARGET_INDEX] = 1
        escapable[callbacks.BOMB_OPPONENT_COUNT_INDEX] = 1
        escapable[callbacks.OPPONENT_ESCAPE_ROUTE_COUNT_INDEX] = 3
        trapped = escapable.copy()
        trapped[callbacks.OPPONENT_ESCAPE_ROUTE_COUNT_INDEX] = 0

        escapable_reward = train.reward_from_events(
            SimpleNamespace(),
            [e.BOMB_DROPPED],
            old_features=escapable,
        )
        trapped_reward = train.reward_from_events(
            SimpleNamespace(),
            [e.BOMB_DROPPED],
            old_features=trapped,
        )
        self.assertAlmostEqual(trapped_reward - escapable_reward, 0.30)


class VNextFeatureTests(unittest.TestCase):
    def test_feature_dimension_is_37(self):
        self.assertEqual(callbacks.FEATURE_DIM, 37)

    def test_bfs_first_action_routes_around_wall(self):
        field = np.zeros((7, 7), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        field[2, 1:5] = -1
        first_actions, distance = callbacks._shortest_route_features(
            field,
            start=(1, 1),
            goals={(3, 1)},
            occupied=set(),
        )
        self.assertEqual(first_actions, [0, 0, 1, 0])
        self.assertEqual(distance, 10)

    def test_many_crates_take_priority_over_opponent_route(self):
        field = np.zeros((9, 9), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        crates = [(x, y) for x in range(2, 7) for y in range(2, 7)][:19]
        first, distance, kind = callbacks._path_route_features(
            field,
            start=(1, 1),
            coins=[],
            crates=crates,
            opponents=[(7, 7)],
            occupied={(7, 7)},
        )
        self.assertEqual(kind, 2)
        self.assertGreater(distance, 0)
        self.assertTrue(any(first))

    def test_few_crates_switch_to_opponent_route(self):
        field = np.zeros((9, 9), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        _, _, kind = callbacks._path_route_features(
            field,
            start=(1, 1),
            coins=[],
            crates=[(4, 4)],
            opponents=[(7, 7)],
            occupied={(7, 7)},
        )
        self.assertEqual(kind, 3)

    def test_combat_gate_reduces_residual_and_adjacent_disables_it(self):
        checkpoint = torch.load(
            Path(callbacks.__file__).resolve().parent.parent
            / "lsk_task4_agent"
            / "dqn_model_task4_hybrid_1k.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = callbacks.DQN()
        callbacks._load_legacy_policy(model, checkpoint["policy_state_dict"])
        with torch.no_grad():
            model.path_head[-1].bias.fill_(1.0)

        resource = torch.zeros(1, callbacks.FEATURE_DIM)
        hunting = resource.clone()
        hunting[0, callbacks.PATH_TARGET_IS_OPPONENT_INDEX] = 1
        adjacent = hunting.clone()
        adjacent[0, callbacks.OPPONENT_ADJACENT_INDEX] = 1
        with torch.no_grad():
            base = model.base_network(resource[:, : callbacks.LEGACY_FEATURE_DIM])
            adjacent_base = model.base_network(
                adjacent[:, : callbacks.LEGACY_FEATURE_DIM]
            )
            resource_delta = (model(resource) - base).abs().mean()
            hunting_delta = (model(hunting) - base).abs().mean()
            adjacent_delta = (model(adjacent) - adjacent_base).abs().mean()
        self.assertGreater(resource_delta, hunting_delta)
        self.assertAlmostEqual(
            float(hunting_delta / resource_delta),
            callbacks.COMBAT_RESIDUAL_SCALE,
            places=5,
        )
        self.assertEqual(float(adjacent_delta), 0.0)

    def test_loop_break_uses_non_returning_bfs_action(self):
        field = np.zeros((7, 7), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        state = {
            "field": field,
            "self": ("agent", 0, True, (3, 3)),
            "others": [],
            "bombs": [],
            "coins": [(5, 3)],
            "explosion_map": np.zeros_like(field),
        }
        features = np.zeros(callbacks.FEATURE_DIM, dtype=np.int8)
        features[callbacks.PATH_FIRST_ACTION_START + 1] = 1
        q_values = np.array([0.0, 1.0, 0.0, 10.0, 0.0, 0.0])
        recent = deque([(2, 3), (3, 3)] * 3, maxlen=6)
        action = callbacks._loop_break_action(
            state,
            features,
            q_values,
            valid_indices=[1, 3, 4],
            recent=recent,
        )
        self.assertEqual(action, 1)

    def test_coin_commit_uses_safe_close_q_route(self):
        field = np.zeros((7, 7), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        state = {
            "field": field,
            "self": ("agent", 0, True, (3, 3)),
            "others": [],
            "bombs": [],
            "coins": [(5, 3)],
            "explosion_map": np.zeros_like(field),
        }
        features = np.zeros(callbacks.FEATURE_DIM, dtype=np.int8)
        features[callbacks.PATH_TARGET_IS_COIN_INDEX] = 1
        features[callbacks.PATH_DISTANCE_INDEX] = 2
        features[callbacks.PATH_FIRST_ACTION_START + 1] = 1
        features[callbacks.CRATES_REMAINING_INDEX] = (
            callbacks.CRATE_HARVEST_THRESHOLD + 1
        )
        q_values = np.array([1.0, 0.9, 0.0, 0.0, 0.0, -1.0])
        action = callbacks._coin_commit_action(
            state,
            features,
            q_values,
            valid_indices=[0, 1, 2, 3, 4],
        )
        self.assertEqual(action, 1)

    def test_coin_commit_does_not_override_close_combat(self):
        field = np.zeros((7, 7), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        state = {
            "field": field,
            "self": ("agent", 0, True, (3, 3)),
            "others": [("enemy", 0, True, (3, 5))],
            "bombs": [],
            "coins": [(5, 3)],
            "explosion_map": np.zeros_like(field),
        }
        features = np.zeros(callbacks.FEATURE_DIM, dtype=np.int8)
        features[callbacks.PATH_TARGET_IS_COIN_INDEX] = 1
        features[callbacks.PATH_DISTANCE_INDEX] = 2
        features[callbacks.PATH_FIRST_ACTION_START + 1] = 1
        features[callbacks.CRATES_REMAINING_INDEX] = (
            callbacks.CRATE_HARVEST_THRESHOLD + 1
        )
        self.assertIsNone(
            callbacks._coin_commit_action(
                state,
                features,
                np.zeros(6),
                valid_indices=[0, 1, 2, 3, 4],
            )
        )

    def test_open_room_has_four_escape_routes(self):
        field = np.zeros((9, 9), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        distance, routes = callbacks._bomb_escape_metrics(
            field,
            (4, 4),
            occupied=set(),
            existing_danger=set(),
        )
        self.assertEqual(distance, 2)
        self.assertEqual(routes, 4)

    def test_corridor_has_one_escape_route(self):
        field = np.full((9, 9), -1, dtype=np.int8)
        for position in ((4, 4), (4, 3), (4, 2), (3, 2)):
            field[position] = 0
        distance, routes = callbacks._bomb_escape_metrics(
            field,
            (4, 4),
            occupied=set(),
            existing_danger=set(),
        )
        self.assertEqual(distance, 3)
        self.assertEqual(routes, 1)

    def test_state_contains_bomb_quality_features(self):
        field = np.zeros((9, 9), dtype=np.int8)
        field[[0, -1], :] = -1
        field[:, [0, -1]] = -1
        field[4, 2] = 1
        field[4, 1] = 1
        state = {
            "round": 1,
            "step": 1,
            "field": field,
            "self": ("vnext", 0, 1, (4, 4)),
            "others": [("opponent", 0, 1, (6, 4))],
            "bombs": [],
            "coins": [],
            "explosion_map": np.zeros_like(field),
        }
        features = callbacks.state_to_features(state)
        self.assertEqual(features.shape, (37,))
        self.assertEqual(features[callbacks.BOMB_TARGET_COUNT_INDEX], 3)
        self.assertEqual(features[callbacks.ESCAPE_ROUTE_COUNT_INDEX], 4)
        self.assertEqual(features[callbacks.BOMB_OPPONENT_COUNT_INDEX], 1)
        self.assertEqual(
            features[callbacks.OPPONENT_ESCAPE_ROUTE_COUNT_INDEX],
            4,
        )
        self.assertEqual(features[callbacks.PATH_FIRST_ACTION_START + 1], 1)
        self.assertEqual(features[callbacks.PATH_DISTANCE_INDEX], 1)
        self.assertEqual(features[callbacks.PATH_TARGET_IS_OPPONENT_INDEX], 1)

    def test_legacy_migration_preserves_action_ranking(self):
        legacy_checkpoint = (
            Path(callbacks.__file__).resolve().parent.parent
            / "lsk_task4_agent"
            / "dqn_model_task4_hybrid_1k.pt"
        )
        checkpoint = torch.load(
            legacy_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        legacy = legacy_callbacks.DQN()
        legacy.load_state_dict(checkpoint["policy_state_dict"])
        migrated = callbacks.DQN()
        callbacks._load_legacy_policy(
            migrated,
            checkpoint["policy_state_dict"],
        )

        old_features = torch.randn(8, callbacks.LEGACY_FEATURE_DIM)
        new_features = torch.cat(
            [
                old_features,
                torch.zeros(
                    8,
                    callbacks.FEATURE_DIM - callbacks.LEGACY_FEATURE_DIM,
                ),
            ],
            dim=1,
        )
        with torch.no_grad():
            old_q = legacy(old_features)
            new_q = migrated(new_features)
        self.assertTrue(torch.allclose(old_q, new_q))
        self.assertTrue(
            torch.equal(
                migrated.path_head[-1].weight,
                torch.zeros_like(migrated.path_head[-1].weight),
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in migrated.base_network.parameters()
            )
        )


class NStepTests(unittest.TestCase):
    @staticmethod
    def transition(reward, done=False, marker=0):
        state = np.full(callbacks.FEATURE_DIM, marker, dtype=np.int8)
        return train.Transition(
            state=state,
            action=0,
            next_state=None if done else state + 1,
            reward=reward,
            done=done,
            discount=callbacks.GAMMA,
        )

    def test_five_step_return(self):
        transitions = [self.transition(1.0, marker=i) for i in range(5)]
        result = train._aggregate_n_step(transitions)
        expected = sum(callbacks.GAMMA ** i for i in range(5))
        self.assertAlmostEqual(result.reward, expected)
        self.assertAlmostEqual(result.discount, callbacks.GAMMA ** 5)
        self.assertFalse(result.done)
        np.testing.assert_array_equal(result.next_state, transitions[-1].next_state)

    def test_terminal_return_stops_early(self):
        transitions = [
            self.transition(1.0, marker=0),
            self.transition(2.0, done=True, marker=1),
            self.transition(100.0, marker=2),
        ]
        result = train._aggregate_n_step(transitions)
        self.assertAlmostEqual(result.reward, 1.0 + callbacks.GAMMA * 2.0)
        self.assertAlmostEqual(result.discount, callbacks.GAMMA ** 2)
        self.assertTrue(result.done)
        self.assertIsNone(result.next_state)


if __name__ == "__main__":
    unittest.main()

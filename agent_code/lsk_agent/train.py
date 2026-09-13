import pickle
from typing import List

import events as e
import numpy as np

from .callbacks import (
    ACTIONS,
    ALPHA,
    GAMMA,
    MODEL_PATH,
    TARGET_DISTANCE_INDEX,
    TARGET_IS_OPPONENT_INDEX,
    state_to_features,
    state_to_key,
    valid_action_indices,
)


def setup_training(self):
    """Initialize the values used by the online Q-learning update."""
    if not isinstance(self.model, dict):
        self.model = {}
    self.alpha = ALPHA
    self.gamma = GAMMA


def _q_values(self, state_key):
    """Return the Q-value vector for a state, creating it if needed."""
    if state_key not in self.model:
        self.model[state_key] = np.zeros(len(ACTIONS), dtype=np.float32)
    return self.model[state_key]


def _update_q_value(self, old_features, action, new_features, reward):
    """Apply Q(s,a) <- Q(s,a) + alpha * TD-error."""
    if old_features is None or action not in ACTIONS:
        return

    old_key = state_to_key(old_features)
    old_q_values = _q_values(self, old_key)
    action_index = ACTIONS.index(action)

    if new_features is None:
        target = reward
    else:
        new_key = state_to_key(new_features)
        next_q_values = _q_values(self, new_key)
        next_valid = valid_action_indices(new_features)
        best_next_q = max(next_q_values[index] for index in next_valid)
        target = reward + self.gamma * best_next_q

    old_q_values[action_index] += self.alpha * (
        target - old_q_values[action_index]
    )


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Convert one game transition into a Q-learning update."""
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events, old_features, new_features)
    _update_q_value(self, old_features, self_action, new_features, reward)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Apply the terminal update and save the Task 3 Q-table."""
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events, last_features, None)
    _update_q_value(self, last_features, last_action, None, reward)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as file:
        pickle.dump(self.model, file)

    self.logger.info("Saved %d Task 3 states to %s", len(self.model), MODEL_PATH)


def reward_from_events(self, events: List[str], old_features=None, new_features=None) -> float:
    """Reward opponent hunting and survival while penalizing unsafe play."""
    event_rewards = {
        # Task 3's main objective is to eliminate opponents.  Make this
        # substantially more valuable than collecting coins or crates.
        e.KILLED_OPPONENT: 15.0,
        e.OPPONENT_ELIMINATED: 3.0,
        # Crates are useful for exploration, but should not dominate the
        # opponent-hunting objective.
        e.CRATE_DESTROYED: 0.25,
        e.COIN_FOUND: 0.25,
        e.COIN_COLLECTED: 0.25,
        e.BOMB_DROPPED: 0.0,
        e.INVALID_ACTION: -0.30,
        e.WAITED: -0.05,
        # Penalize both kinds of death: self-inflicted explosions and being
        # killed by another agent.
        e.KILLED_SELF: -10.0,
        e.GOT_KILLED: -8.0,
        e.SURVIVED_ROUND: 2.0,
    }

    # Small time cost discourages aimless waiting.
    reward = -0.01
    for event in events:
        reward += event_rewards.get(event, 0.0)

    # Potential-style shaping: strongly encourage approaching opponents, but
    # keep crate/coin shaping weak so those fallback targets do not dominate
    # Task 3's opponent-hunting objective.
    if old_features is not None and new_features is not None:
        old_distance = int(old_features[TARGET_DISTANCE_INDEX])
        new_distance = int(new_features[TARGET_DISTANCE_INDEX])
        if (
            int(old_features[TARGET_IS_OPPONENT_INDEX]) == 1
            and int(new_features[TARGET_IS_OPPONENT_INDEX]) == 1
        ):
            reward += 0.25 * (old_distance - new_distance)
        else:
            reward += 0.05 * (old_distance - new_distance)

        # Give a small additional signal for leaving an active danger zone.
        if int(old_features[8]) == 1 and int(new_features[8]) == 0:
            reward += 0.5

    return reward

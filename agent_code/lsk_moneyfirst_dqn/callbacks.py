import os
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Keep the same action order in callbacks.py and train.py.
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTION_INDICES = list(range(len(ACTIONS)))

# Task 4 keeps all six actions, including BOMB.
TASK4_ACTION_INDICES = ACTION_INDICES

# DQN hyperparameters shared with train.py.
GAMMA = 0.95
LEARNING_RATE = 2e-4
LEGACY_FEATURE_DIM = 23
PHASE_FEATURE_DIM = 37
FEATURE_DIM = 43
EPSILON_START = 1.00
EPSILON_WARM_START = 0.10
EPSILON_MIN = 0.05
# From 1.0 this reaches roughly 0.22 after 5,000 rounds and 0.05 after 10,000.
EPSILON_DECAY = 0.9995
BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
REPLAY_WARMUP = 1_000
TRAIN_EVERY = 4
TARGET_UPDATE_INTERVAL = 1_000
SAVE_INTERVAL = 100

# Path-aware VNext is independent from the accepted Hybrid 1k agent.  The
# legacy network is frozen and a bounded residual head learns only corrections
# from the new route/phase/bomb-quality features.
MODEL_PATH = Path(__file__).resolve().parent / "dqn_model_task4_phase.pt"
BASE_MODEL_PATH = Path(__file__).resolve().parent / "dqn_model_task4_path.pt"
DEVICE = torch.device("cpu")

# Feature indices used by helper functions and reward shaping.
TARGET_DISTANCE_INDEX = 15
TARGET_IS_COIN_INDEX = 16
BOMBS_LEFT_INDEX = 17
SAFE_NEIGHBORS_INDEX = 18
ESCAPE_AFTER_BOMB_INDEX = 19
BOMB_HITS_TARGET_INDEX = 20
TARGET_IS_OPPONENT_INDEX = 21
OPPONENT_ADJACENT_INDEX = 22
PATH_FIRST_ACTION_START = 23
PATH_DISTANCE_INDEX = 27
PATH_TARGET_IS_COIN_INDEX = 28
PATH_TARGET_IS_OPPONENT_INDEX = 29
OPPONENTS_ALIVE_INDEX = 30
CRATES_REMAINING_INDEX = 31
VISIBLE_COINS_INDEX = 32
BOMB_TARGET_COUNT_INDEX = 33
ESCAPE_ROUTE_COUNT_INDEX = 34
BOMB_OPPONENT_COUNT_INDEX = 35
OPPONENT_ESCAPE_ROUTE_COUNT_INDEX = 36
COIN_RACE_MARGIN_INDEX = 37
COIN_RACE_FIRST_ACTION_START = 38
COIN_RACE_DISTANCE_INDEX = 42
COIN_PRIORITY_DISTANCE = 5
PATH_DISTANCE_LIMIT = 15
RESIDUAL_SCALE = 1.0
COMBAT_RESIDUAL_SCALE = 0.35
CRATE_HARVEST_THRESHOLD = 18
ENDGAME_CRATE_THRESHOLD = 0
# With strict first place as the only objective, protect even a one-point lead
# once the board has reached the low-crate endgame.
SAFE_SCORE_LEAD = 1
COIN_COMMIT_DISTANCE = int(os.environ.get("PATH_COIN_DISTANCE", "6"))
COIN_Q_MARGIN_STD = float(os.environ.get("PATH_COIN_Q_MARGIN", "2.0"))

# Directions follow the action order UP, RIGHT, DOWN, LEFT.
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class DQN(nn.Module):
    """Frozen Hybrid base plus the accepted phase-aware residual head."""

    def __init__(self):
        super().__init__()
        self.base_network = nn.Sequential(
            nn.Linear(LEGACY_FEATURE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(ACTIONS)),
        )
        self.path_head = nn.Sequential(
            nn.Linear(PHASE_FEATURE_DIM - LEGACY_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, len(ACTIONS)),
        )
        nn.init.zeros_(self.path_head[-1].weight)
        nn.init.zeros_(self.path_head[-1].bias)
        for parameter in self.base_network.parameters():
            parameter.requires_grad = False

    def forward(self, features):
        base_q = self.base_network(features[:, :LEGACY_FEATURE_DIM])
        phase_features = features[:, :PHASE_FEATURE_DIM]
        residual = torch.tanh(
            self.path_head(phase_features[:, LEGACY_FEATURE_DIM:])
        )
        # Resource navigation receives the full learned correction. During
        # opponent hunting preserve most of the accepted Hybrid policy, and
        # hand adjacent combat back to it completely.
        hunting = features[:, PATH_TARGET_IS_OPPONENT_INDEX] > 0.5
        adjacent_combat = features[:, OPPONENT_ADJACENT_INDEX] > 0.0
        gate = torch.ones_like(features[:, 0])
        gate = torch.where(
            hunting,
            torch.full_like(gate, COMBAT_RESIDUAL_SCALE),
            gate,
        )
        gate = torch.where(adjacent_combat, torch.zeros_like(gate), gate)
        return base_q + RESIDUAL_SCALE * gate.unsqueeze(1) * residual


def _load_legacy_policy(policy_net, legacy_state_dict):
    """Load the accepted policy into the frozen base without changing Q."""
    with torch.no_grad():
        if legacy_state_dict["network.0.weight"].shape[1] != LEGACY_FEATURE_DIM:
            raise ValueError(
                f"Expected {LEGACY_FEATURE_DIM} legacy inputs, "
                f"got {legacy_state_dict['network.0.weight'].shape[1]}"
            )
        base_state = {
            key.removeprefix("network."): value
            for key, value in legacy_state_dict.items()
        }
        policy_net.base_network.load_state_dict(base_state)
        nn.init.zeros_(policy_net.path_head[-1].weight)
        nn.init.zeros_(policy_net.path_head[-1].bias)


# Most features are already in [-1, 1]. Normalize the three count/distance
# features so one input cannot dominate merely because it has a larger scale.
FEATURE_SCALE = np.ones(FEATURE_DIM, dtype=np.float32)
FEATURE_SCALE[TARGET_DISTANCE_INDEX] = 5.0
FEATURE_SCALE[SAFE_NEIGHBORS_INDEX] = 4.0
FEATURE_SCALE[OPPONENT_ADJACENT_INDEX] = 4.0
FEATURE_SCALE[PATH_DISTANCE_INDEX] = float(PATH_DISTANCE_LIMIT)
FEATURE_SCALE[OPPONENTS_ALIVE_INDEX] = 3.0
FEATURE_SCALE[CRATES_REMAINING_INDEX] = 40.0
FEATURE_SCALE[VISIBLE_COINS_INDEX] = 9.0
FEATURE_SCALE[BOMB_TARGET_COUNT_INDEX] = 4.0
FEATURE_SCALE[ESCAPE_ROUTE_COUNT_INDEX] = 4.0
FEATURE_SCALE[BOMB_OPPONENT_COUNT_INDEX] = 3.0
FEATURE_SCALE[OPPONENT_ESCAPE_ROUTE_COUNT_INDEX] = 4.0
FEATURE_SCALE[COIN_RACE_MARGIN_INDEX] = 5.0
FEATURE_SCALE[COIN_RACE_DISTANCE_INDEX] = float(PATH_DISTANCE_LIMIT)


def normalize_features(features):
    """Return normalized float32 features accepted by the DQN."""
    values = np.asarray(features, dtype=np.float32)
    if values.shape[-1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {values.shape[-1]}")
    return values / FEATURE_SCALE


def setup(self):
    """Create the DQN and load its independent Task 4 checkpoint if present."""
    self.policy_net = DQN().to(DEVICE)
    self.checkpoint = None
    self.episodes = 0
    self.warm_started = False

    if MODEL_PATH.exists():
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            self.checkpoint = checkpoint
            self.episodes = int(checkpoint.get("episodes", 0))
            self.logger.info("Loaded Task 4 DQN checkpoint from %s", MODEL_PATH)
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
            self.logger.warning("Could not load Task 4 DQN checkpoint: %s", error)
            self.logger.info("Starting a new Task 4 DQN instead.")
    elif BASE_MODEL_PATH.exists():
        try:
            checkpoint = torch.load(
                BASE_MODEL_PATH,
                map_location=DEVICE,
                weights_only=False,
            )
            base_state = checkpoint["policy_state_dict"]
            if "base_network.0.weight" in base_state:
                self.policy_net.load_state_dict(base_state)
            else:
                _load_legacy_policy(self.policy_net, base_state)
            self.warm_started = True
            self.logger.info(
                "Warm-started hybrid Task 4 DQN from %s",
                BASE_MODEL_PATH,
            )
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
            self.logger.warning("Could not warm-start hybrid DQN: %s", error)
            self.logger.info("Starting a new hybrid Task 4 DQN instead.")
    else:
        self.logger.info("Starting a new Task 4 DQN.")

    # Keep the conventional attribute available for framework/debugging code.
    self.model = self.policy_net
    self.recent_positions = deque(maxlen=6)
    self.memory_round = None
    if self.train:
        if self.checkpoint is not None:
            saved_epsilon = self.checkpoint.get("epsilon", EPSILON_START)
        elif self.warm_started:
            saved_epsilon = EPSILON_WARM_START
        else:
            saved_epsilon = EPSILON_START
        self.epsilon = min(max(float(saved_epsilon), EPSILON_MIN), EPSILON_START)
        self.policy_net.train()
    else:
        self.epsilon = 0.0
        self.policy_net.eval()


def valid_action_indices(features):
    """Return actions that are physically possible in the current state.

    Movement into walls/crates is excluded using the first four features.
    BOMB is excluded when no bomb is available. WAIT is always possible.
    """
    if features is None:
        return [4]

    movement_indices = [index for index in range(4) if int(features[index]) == 1]
    # If already in danger, prefer directions that are not in the predicted
    # blast area. Fall back to all walkable directions if no safe move exists.
    in_danger = int(features[8]) == 1
    if in_danger:
        safe_movement = [
            index for index in movement_indices
            if int(features[9 + index]) == 0
        ]
        valid = safe_movement or movement_indices
    else:
        valid = movement_indices
    # Waiting is removed while a safe escape direction exists.
    if not (in_danger and valid):
        valid.append(4)  # WAIT
    # Permit useful bombs when a short escape route exists. Opponents within
    # three cells are attack targets even before the current ray overlaps them.
    if (
        int(features[BOMBS_LEFT_INDEX]) > 0
        and int(features[ESCAPE_AFTER_BOMB_INDEX]) == 1
        and int(features[8]) == 0
        and (
            int(features[BOMB_HITS_TARGET_INDEX]) == 1
            or (
                int(features[TARGET_IS_OPPONENT_INDEX]) == 1
                and int(features[TARGET_DISTANCE_INDEX]) <= 3
            )
        )
    ):
        valid.append(5)  # BOMB
    return valid


def survival_safe_action_indices(game_state, candidate_indices):
    """Remove actions with no surviving route through visible bomb timers.

    The learned 23-feature policy still evaluates future bomb danger. This
    final safety layer changes no network inputs: it only rejects a candidate
    when a short time-expanded search cannot survive the visible explosions.
    """
    if game_state is None:
        return [4]

    candidates = list(dict.fromkeys(int(index) for index in candidate_indices))
    safe_candidates = [
        index
        for index in candidates
        if _action_has_survival_route(game_state, index)
    ]

    if safe_candidates:
        return safe_candidates

    # Only restore WAIT as a rescue action after every learned candidate has
    # been rejected. Adding it earlier can make the agent linger beside its
    # own counting bomb instead of following the learned escape sequence.
    if 4 not in candidates and _action_has_survival_route(game_state, 4):
        return [4]

    # If every possible action, including waiting, is immediately lethal,
    # retain the original candidates as a last resort.
    return candidates or [4]


def _sync_position_memory(self, game_state):
    """Reset episode memory and remember the current position."""
    if not hasattr(self, "recent_positions"):
        self.recent_positions = deque(maxlen=6)
    if not hasattr(self, "memory_round"):
        self.memory_round = None
    round_number = int(game_state.get("round", 0))
    if self.memory_round != round_number:
        self.memory_round = round_number
        self.recent_positions.clear()
    self.recent_positions.append(tuple(game_state["self"][3]))


def _two_tile_loop_detected(recent_positions):
    positions = list(recent_positions)
    if len(positions) < 6:
        return False
    tail = positions[-6:]
    return (
        tail[0] == tail[2] == tail[4]
        and tail[1] == tail[3] == tail[5]
        and tail[0] != tail[1]
    )


def _board_has_active_danger(game_state):
    if game_state.get("bombs"):
        return True
    explosion_map = game_state.get("explosion_map")
    return explosion_map is not None and bool(np.any(explosion_map > 0))


def _candidate_position(game_state, action_index):
    x, y = game_state["self"][3]
    if 0 <= action_index < 4:
        dx, dy = DIRECTIONS[action_index]
        return x + dx, y + dy
    return x, y


def _loop_break_action(game_state, features, q_values, valid_indices, recent):
    """Break a confirmed calm resource loop without changing combat policy."""
    if int(features[PATH_TARGET_IS_OPPONENT_INDEX]) == 1:
        return None
    if _board_has_active_danger(game_state):
        return None
    if not _two_tile_loop_detected(recent):
        return None

    previous_position = list(recent)[-2]
    movement = [
        index
        for index in valid_indices
        if 0 <= index < 4
        and _candidate_position(game_state, index) != previous_position
    ]
    if not movement:
        return None
    routed = [
        index
        for index in movement
        if int(features[PATH_FIRST_ACTION_START + index]) == 1
    ]
    candidates = routed or movement
    best_q = max(float(q_values[index]) for index in candidates)
    best = [
        index
        for index in candidates
        if np.isclose(float(q_values[index]), best_q)
    ]
    return random.choice(best)


def _coin_commit_action(game_state, features, q_values, valid_indices):
    """Take a nearby safe coin route unless Hybrid strongly disagrees."""
    if int(features[PATH_TARGET_IS_COIN_INDEX]) != 1:
        return None

    own_score = int(game_state["self"][1])
    opponent_scores = [
        int(agent[1]) for agent in game_state.get("others", [])
    ]
    not_leading = bool(opponent_scores) and own_score <= max(opponent_scores)
    opening = int(features[CRATES_REMAINING_INDEX]) > CRATE_HARVEST_THRESHOLD
    # During the opening, economy is always valuable. Later, only a tied or
    # trailing agent commits to a coin route; a leader keeps its survival plan.
    if not opening and not not_leading:
        return None
    distance = int(features[COIN_RACE_DISTANCE_INDEX])
    if distance <= 0 or distance > COIN_COMMIT_DISTANCE:
        return None

    x, y = game_state["self"][3]
    nearest_opponent = min(
        (
            abs(agent[3][0] - x) + abs(agent[3][1] - y)
            for agent in game_state.get("others", [])
        ),
        default=99,
    )
    # A nearby opponent used to cancel every coin commitment.  Keep yielding
    # contested coins when the opponent is at least as close, but take the
    # safe route when our BFS race margin says that we can arrive first.
    if (
        nearest_opponent <= 3
        and int(features[COIN_RACE_MARGIN_INDEX]) <= 0
    ):
        return None

    routed = [
        index
        for index in valid_indices
        if 0 <= index < 4
        and int(features[COIN_RACE_FIRST_ACTION_START + index]) == 1
    ]
    if not routed:
        return None
    routed_best = max(float(q_values[index]) for index in routed)
    valid_values = np.asarray(
        [float(q_values[index]) for index in valid_indices],
        dtype=float,
    )
    q_margin = COIN_Q_MARGIN_STD * max(float(np.std(valid_values)), 1e-6)
    if routed_best < float(np.max(valid_values)) - q_margin:
        return None
    best = [
        index
        for index in routed
        if np.isclose(float(q_values[index]), routed_best)
    ]
    return random.choice(best)


def _score_aware_endgame_action(
    game_state,
    features,
    q_values,
    valid_indices,
    proposed_action,
):
    """Protect a real late-game lead without making a loser hide forever.

    The override is intentionally narrow. It activates only in a calm endgame
    with a lead of at least two points. A normal non-bomb action that does not
    approach the closest opponent is preserved exactly. Otherwise, choose a
    survivable non-bomb action that maximizes opponent distance and exits.
    """
    opponents = list(game_state.get("others", []))
    if not opponents or int(features[8]) == 1:
        return None
    if int(features[CRATES_REMAINING_INDEX]) > ENDGAME_CRATE_THRESHOLD:
        return None

    own_score = int(game_state["self"][1])
    score_lead = own_score - max(int(agent[1]) for agent in opponents)
    if score_lead < SAFE_SCORE_LEAD:
        return None

    # Visible bombs already have a time-expanded safety policy. Do not replace
    # its learned escape sequence with the calm-board endgame heuristic.
    if _board_has_active_danger(game_state):
        return None

    opponent_positions = [tuple(agent[3]) for agent in opponents]
    current_position = tuple(game_state["self"][3])

    def opponent_distance(action_index):
        position = _candidate_position(game_state, action_index)
        return min(
            abs(position[0] - ox) + abs(position[1] - oy)
            for ox, oy in opponent_positions
        )

    current_distance = min(
        abs(current_position[0] - ox) + abs(current_position[1] - oy)
        for ox, oy in opponent_positions
    )

    if (
        proposed_action != 5
        and opponent_distance(proposed_action) >= current_distance
    ):
        return proposed_action

    non_bomb = [index for index in valid_indices if index != 5]
    if not non_bomb:
        return None
    non_approaching = [
        index
        for index in non_bomb
        if opponent_distance(index) >= current_distance
    ]
    candidates = non_approaching or non_bomb

    field = game_state["field"]
    occupied = {
        tuple(position) for position, _ in game_state.get("bombs", [])
    } | set(opponent_positions)

    def exit_count(action_index):
        x, y = _candidate_position(game_state, action_index)
        return sum(
            int(_is_walkable(field, occupied, x + dx, y + dy))
            for dx, dy in DIRECTIONS
        )

    return max(
        candidates,
        key=lambda index: (
            opponent_distance(index),
            exit_count(index),
            float(q_values[index]),
        ),
    )


def _economy_priority_action(
    game_state,
    features,
    q_values,
    valid_indices,
    proposed_action,
):
    """Harvest coins and crates before joining moving-opponent combat.

    This mirrors the useful ordering in the teammate's BC teacher without
    importing its model: take a safe opportunistic kill, claim a winnable
    visible coin, bomb crates, then route to a productive crate-bomb tile.
    """
    if int(features[CRATES_REMAINING_INDEX]) <= 0:
        return None
    if int(features[8]) == 1 or _board_has_active_danger(game_state):
        return None

    # "边抢边杀": keep a direct safe bomb opportunity, but never leave the
    # economy route merely to chase a moving opponent while crates remain.
    if (
        proposed_action == 5
        and proposed_action in valid_indices
        and int(features[BOMB_OPPONENT_COUNT_INDEX]) > 0
        and int(features[ESCAPE_ROUTE_COUNT_INDEX]) >= 2
    ):
        return proposed_action

    coin_distance = int(features[COIN_RACE_DISTANCE_INDEX])
    coin_margin = int(features[COIN_RACE_MARGIN_INDEX])
    if coin_distance > 0 and coin_margin > 0:
        coin_actions = [
            index
            for index in valid_indices
            if 0 <= index < 4
            and int(features[COIN_RACE_FIRST_ACTION_START + index]) == 1
        ]
        if coin_actions:
            return max(coin_actions, key=lambda index: float(q_values[index]))

    field = game_state["field"]
    current_position = tuple(game_state["self"][3])
    if 5 in valid_indices:
        crates_hit = sum(
            int(field[x, y] == 1)
            for x, y in _real_blast_cells(field, current_position)
        )
        if crates_hit > 0 and int(features[ESCAPE_ROUTE_COUNT_INDEX]) >= 1:
            return 5

    bombs = {tuple(position) for position, _ in game_state.get("bombs", [])}
    opponents = {tuple(agent[3]) for agent in game_state.get("others", [])}
    crates = [tuple(position) for position in zip(*np.where(field == 1))]
    productive_actions, productive_distance = _productive_bomb_route_features(
        field,
        current_position,
        crates,
        bombs | opponents,
    )
    if productive_distance > 0:
        routed = [
            index
            for index in valid_indices
            if 0 <= index < 4 and int(productive_actions[index]) == 1
        ]
        if routed:
            return max(routed, key=lambda index: float(q_values[index]))

    return None


def act(self, game_state: dict) -> str:
    """Choose a valid Task 4 action using epsilon-greedy DQN values."""
    features = state_to_features(game_state)
    if features is None:
        return "WAIT"

    _sync_position_memory(self, game_state)

    valid_indices = survival_safe_action_indices(
        game_state,
        valid_action_indices(features),
    )
    if not valid_indices:
        return "WAIT"

    q_values = None
    if self.train and random.random() < self.epsilon:
        action_index = random.choice(valid_indices)
    else:
        feature_tensor = torch.as_tensor(
            normalize_features(features), dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(feature_tensor).squeeze(0).cpu().numpy()
        valid_q_values = q_values[valid_indices]
        best_value = float(np.max(valid_q_values))
        best_indices = [
            index for index in valid_indices
            if np.isclose(q_values[index], best_value)
        ]
        # Random tie-breaking avoids a fixed preference for one action.
        action_index = random.choice(best_indices or valid_indices)

    if not self.train and q_values is not None:
        economy_action = _economy_priority_action(
            game_state=game_state,
            features=features,
            q_values=q_values,
            valid_indices=valid_indices,
            proposed_action=action_index,
        )
        if economy_action is not None:
            action_index = economy_action
        else:
            score_action = _score_aware_endgame_action(
                game_state=game_state,
                features=features,
                q_values=q_values,
                valid_indices=valid_indices,
                proposed_action=action_index,
            )
            if score_action is not None:
                action_index = score_action
            else:
                coin_action = _coin_commit_action(
                    game_state,
                    features,
                    q_values,
                    valid_indices,
                )
                if coin_action is not None:
                    action_index = coin_action
                else:
                    loop_action = _loop_break_action(
                        game_state,
                        features,
                        q_values,
                        valid_indices,
                        self.recent_positions,
                    )
                    if loop_action is not None:
                        action_index = loop_action

    return ACTIONS[action_index]


def _in_bounds(field, x, y):
    width, height = field.shape
    return 0 <= x < width and 0 <= y < height


def _is_walkable(field, occupied, x, y):
    """A walkable cell is floor and is not occupied by an agent or bomb."""
    return _in_bounds(field, x, y) and field[x, y] == 0 and (x, y) not in occupied


def _blast_cells(field, origin, power=3):
    """Return the legacy blast features used to train this checkpoint."""
    cells = [origin]
    ox, oy = origin
    for dx, dy in DIRECTIONS:
        for distance in range(1, power + 1):
            x, y = ox + dx * distance, oy + dy * distance
            if not _in_bounds(field, x, y) or field[x, y] == -1:
                break
            cells.append((x, y))
            if field[x, y] == 1:
                break
    return cells


def _real_blast_cells(field, origin, power=3):
    """Return environment-accurate blast cells for the safety layer."""
    cells = [origin]
    ox, oy = origin
    for dx, dy in DIRECTIONS:
        for distance in range(1, power + 1):
            x, y = ox + dx * distance, oy + dy * distance
            if not _in_bounds(field, x, y) or field[x, y] == -1:
                break
            cells.append((x, y))
    return cells


def _danger_map(game_state):
    """Estimate cells that are exploding or will soon be hit by a bomb."""
    field = game_state["field"]
    danger = set()

    # Positive explosion_map values indicate active explosions.
    explosion_map = game_state.get("explosion_map")
    if explosion_map is not None:
        for x, y in zip(*np.where(explosion_map > 0)):
            danger.add((int(x), int(y)))

    # A bomb is considered dangerous from its initial four-step timer.
    for bomb_position, timer in game_state.get("bombs", []):
        if timer <= 4:
            danger.update(_blast_cells(field, tuple(bomb_position)))

    return danger


def _immediate_danger_map(game_state):
    """Return cells lethal immediately after the current action."""
    field = game_state["field"]
    danger = set()

    explosion_map = game_state.get("explosion_map")
    if explosion_map is not None:
        for x, y in zip(*np.where(explosion_map > 0)):
            danger.add((int(x), int(y)))

    # In this environment a bomb observed at timer zero explodes after the
    # agents choose their current actions.
    for bomb_position, timer in game_state.get("bombs", []):
        if int(timer) <= 0:
            danger.update(_real_blast_cells(field, tuple(bomb_position)))

    return danger


def _timed_danger_schedule(game_state, place_bomb=False):
    """Map future action offsets to environment-accurate lethal cells."""
    field = game_state["field"]
    schedule = {0: _immediate_danger_map(game_state)}
    timed_bombs = []

    for bomb_position, timer in game_state.get("bombs", []):
        position = tuple(bomb_position)
        timer = int(timer)
        timed_bombs.append((position, timer))
        if timer > 0:
            blast = set(_real_blast_cells(field, position))
            schedule.setdefault(timer, set()).update(blast)
            schedule.setdefault(timer + 1, set()).update(blast)

    if place_bomb:
        position = tuple(game_state["self"][3])
        timed_bombs.append((position, 4))
        blast = set(_real_blast_cells(field, position))
        schedule.setdefault(4, set()).update(blast)
        schedule.setdefault(5, set()).update(blast)

    return schedule, timed_bombs


def _action_has_survival_route(game_state, action_index):
    """Return whether one action leaves a route through visible explosions."""
    field = game_state["field"]
    _, _, _, (x, y) = game_state["self"]
    if 0 <= action_index < 4:
        dx, dy = DIRECTIONS[action_index]
        start = (x + dx, y + dy)
    elif action_index in (4, 5):
        start = (x, y)
    else:
        return False

    schedule, timed_bombs = _timed_danger_schedule(
        game_state,
        place_bomb=(action_index == 5),
    )
    if start in schedule.get(0, set()):
        return False

    last_danger_step = max(schedule, default=0)
    reachable = {start}
    for step in range(1, last_danger_step + 1):
        occupied_bombs = {
            position
            for position, timer in timed_bombs
            if step <= timer
        }
        next_reachable = set()
        for px, py in reachable:
            for dx, dy in DIRECTIONS + [(0, 0)]:
                nx, ny = px + dx, py + dy
                position = (nx, ny)
                if not _in_bounds(field, nx, ny) or field[nx, ny] != 0:
                    continue
                if position in occupied_bombs:
                    continue
                if position in schedule.get(step, set()):
                    continue
                next_reachable.add(position)

        reachable = next_reachable
        if not reachable:
            return False

    return True


def _bomb_escape_metrics(field, start, occupied, existing_danger, max_steps=4):
    """Return minimum escape distance and independent first-step routes.

    Routes are counted by distinct first moves rather than by destination, so
    a large room behind one bottleneck still counts as one escape option. The
    agent may cross its future blast while the bomb is counting down but must
    leave that blast within ``max_steps``.
    """
    hypothetical_blast = set(_blast_cells(field, start))
    successful_routes = 0
    minimum_distance = None
    sx, sy = start

    for first_dx, first_dy in DIRECTIONS:
        first = (sx + first_dx, sy + first_dy)
        if not _is_walkable(field, occupied, *first):
            continue
        if first in existing_danger:
            continue

        queue = deque([(first, 1)])
        visited = {start, first}
        route_distance = None
        while queue:
            (x, y), depth = queue.popleft()
            if (x, y) not in hypothetical_blast:
                route_distance = depth
                break
            if depth >= max_steps:
                continue
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                position = (nx, ny)
                if position in visited:
                    continue
                if not _is_walkable(field, occupied, nx, ny):
                    continue
                if position in existing_danger:
                    continue
                visited.add(position)
                queue.append((position, depth + 1))

        if route_distance is not None:
            successful_routes += 1
            minimum_distance = (
                route_distance
                if minimum_distance is None
                else min(minimum_distance, route_distance)
            )

    return (minimum_distance or 0), successful_routes


def _escape_after_bomb(field, start, occupied, existing_danger, max_steps=4):
    """Compatibility wrapper for the original binary escape feature."""
    distance, _ = _bomb_escape_metrics(
        field,
        start,
        occupied,
        existing_danger,
        max_steps=max_steps,
    )
    return distance > 0


def _nearest_target(x, y, coins, crates, opponents, prefer_adjacent_crates=False):
    """Prefer nearby visible coins, then opponents, crates and distant coins.

    Target kind: 1 = coin, 2 = crate, 3 = opponent.
    """
    nearby_coins = [
        tuple(coin)
        for coin in coins
        if abs(coin[0] - x) + abs(coin[1] - y) <= COIN_PRIORITY_DISTANCE
    ]
    targets = [(coin, 1) for coin in nearby_coins]
    if not targets:
        targets = [(tuple(opponent), 3) for opponent in opponents]
    if not targets and prefer_adjacent_crates:
        targets = [(tuple(crate), 2) for crate in crates]
    if not targets:
        targets = [(tuple(crate), 2) for crate in crates]
    if not targets:
        targets = [(tuple(coin), 1) for coin in coins]
    if not targets:
        return None, 0

    target, target_kind = min(
        targets,
        key=lambda item: abs(item[0][0] - x) + abs(item[0][1] - y),
    )
    return target, int(target_kind)


def _shortest_route_features(field, start, goals, occupied):
    """Return shortest-path first actions and distance to any static goal."""
    goals = {tuple(goal) for goal in goals}
    if not goals:
        return [0, 0, 0, 0], 0
    if tuple(start) in goals:
        return [0, 0, 0, 0], 0

    distances = {}
    sx, sy = start
    for action_index, (dx, dy) in enumerate(DIRECTIONS):
        first = (sx + dx, sy + dy)
        if not _is_walkable(field, occupied, *first):
            continue
        if first in goals:
            distances[action_index] = 1
            continue

        queue = deque([(first, 1)])
        visited = {tuple(start), first}
        while queue:
            (x, y), distance = queue.popleft()
            if distance >= PATH_DISTANCE_LIMIT:
                continue
            for next_dx, next_dy in DIRECTIONS:
                position = (x + next_dx, y + next_dy)
                if position in visited:
                    continue
                if not _is_walkable(field, occupied, *position):
                    continue
                if position in goals:
                    distances[action_index] = distance + 1
                    queue.clear()
                    break
                visited.add(position)
                queue.append((position, distance + 1))

    if not distances:
        return [0, 0, 0, 0], 0
    shortest = min(distances.values())
    first_actions = [
        int(distances.get(index) == shortest) for index in range(4)
    ]
    return first_actions, min(shortest, PATH_DISTANCE_LIMIT)


def _path_route_features(field, start, coins, crates, opponents, occupied):
    """Build a route target with resource priority and phase information."""
    if coins:
        goals = {tuple(coin) for coin in coins}
        target_kind = 1
    elif crates and (len(crates) > CRATE_HARVEST_THRESHOLD or not opponents):
        goals = {
            (crate[0] + dx, crate[1] + dy)
            for crate in crates
            for dx, dy in DIRECTIONS
            if _in_bounds(field, crate[0] + dx, crate[1] + dy)
            and field[crate[0] + dx, crate[1] + dy] == 0
        }
        target_kind = 2 if goals else 0
    elif opponents:
        goals = {
            (opponent[0] + dx, opponent[1] + dy)
            for opponent in opponents
            for dx, dy in DIRECTIONS
            if _in_bounds(field, opponent[0] + dx, opponent[1] + dy)
            and field[opponent[0] + dx, opponent[1] + dy] == 0
        }
        target_kind = 3
    else:
        goals = set()
        target_kind = 0

    first_actions, distance = _shortest_route_features(
        field,
        start,
        goals,
        occupied,
    )
    return first_actions, distance, target_kind


def _escape_routes_from_blast(field, start, blast, occupied, max_steps=4):
    """Count distinct first moves that can leave one hypothetical blast."""
    if start not in blast:
        return 4
    routes = 0
    sx, sy = start
    for first_dx, first_dy in DIRECTIONS:
        first = (sx + first_dx, sy + first_dy)
        if not _is_walkable(field, occupied, *first):
            continue
        if first not in blast:
            routes += 1
            continue

        queue = deque([(first, 1)])
        visited = {start, first}
        escaped = False
        while queue and not escaped:
            (x, y), depth = queue.popleft()
            if depth >= max_steps:
                continue
            for dx, dy in DIRECTIONS:
                position = (x + dx, y + dy)
                if position in visited:
                    continue
                if not _is_walkable(field, occupied, *position):
                    continue
                if position not in blast:
                    escaped = True
                    break
                visited.add(position)
                queue.append((position, depth + 1))
        routes += int(escaped)
    return routes


def _opponent_bomb_escape_features(field, origin, opponents, occupied):
    """Describe how strongly a bomb here constrains currently hit opponents."""
    blast = set(_real_blast_cells(field, origin))
    threatened = [position for position in opponents if position in blast]
    if not threatened:
        return 0, 0

    route_counts = []
    for position in threatened:
        blockers = set(occupied)
        blockers.discard(position)
        route_counts.append(
            _escape_routes_from_blast(
                field,
                position,
                blast,
                blockers,
            )
        )
    return min(len(threatened), 3), min(route_counts)


def _best_coin_race_features(field, start, coins, opponents, occupied):
    """Route to one coin and compare every agent's distance to that coin.

    Prefer the nearest coin that we can reach before all opponents. If no
    visible coin is currently winnable, retain the closest reachable coin so
    the caller can decide whether the ordinary Phase policy should handle it.
    """
    candidates = []
    for coin in map(tuple, coins):
        first_actions, own_distance = _shortest_route_features(
            field,
            start,
            {coin},
            occupied,
        )
        if own_distance <= 0:
            continue

        opponent_distances = []
        for opponent in opponents:
            blockers = set(occupied)
            blockers.discard(tuple(opponent))
            _, distance = _shortest_route_features(
                field,
                tuple(opponent),
                {coin},
                blockers,
            )
            if distance > 0:
                opponent_distances.append(distance)
        opponent_distance = min(
            opponent_distances,
            default=PATH_DISTANCE_LIMIT,
        )
        margin = int(np.clip(opponent_distance - own_distance, -5, 5))
        candidates.append((first_actions, own_distance, margin, coin))

    if not candidates:
        return [0, 0, 0, 0], 0, 0
    winnable = [candidate for candidate in candidates if candidate[2] > 0]
    pool = winnable or candidates
    first_actions, distance, margin, _ = min(
        pool,
        key=lambda candidate: (
            candidate[1],
            -candidate[2],
            candidate[3],
        ),
    )
    return first_actions, min(distance, PATH_DISTANCE_LIMIT), margin


def _productive_bomb_route_features(field, start, crates, occupied):
    """Route to the nearest floor tile whose blast reaches a crate."""
    if not crates:
        return [0, 0, 0, 0], 0
    crate_set = set(map(tuple, crates))
    goals = {
        (int(x), int(y))
        for x, y in zip(*np.where(field == 0))
        if any(
            cell in crate_set
            for cell in _real_blast_cells(field, (int(x), int(y)))
        )
    }
    return _shortest_route_features(field, start, goals, occupied)


def state_to_features(game_state: dict) -> np.ndarray:
    """Extract compact Task 4 features for DQN.

    Feature layout:
      0-3: free movement in UP, RIGHT, DOWN, LEFT
      4-7: crate directly in UP, RIGHT, DOWN, LEFT
      8: danger at the current cell
      9-12: danger in UP, RIGHT, DOWN, LEFT
      13-14: sign of target direction (dx, dy)
      15: target Manhattan distance, clipped at 5
      16: 1 when target is a visible coin, otherwise 0
      17: number of bombs available, clipped to 1
      18: number of currently safe neighboring cells, clipped at 4
      19: whether a route exists after placing a bomb here
      20: whether the bomb blast would hit a crate or opponent
      21: whether the current target is an opponent
      22: number of adjacent opponents, clipped at 4
      23-26: first moves on a shortest static route to the phase target
      27: BFS route distance, clipped at 15
      28-29: whether the route target is a coin or opponent
      30-32: live opponents, remaining crates and visible coins
      33: number of crates/opponents in the accurate blast, clipped at 4
      34: number of distinct safe first-step escape routes, clipped at 4
      35: opponents currently covered by a bomb placed here, clipped at 3
      36: minimum escape-route count among those threatened opponents
      37: same-coin race margin versus the closest opponent, clipped ±5
      38-41: BFS first moves toward that exact race coin
      42: BFS distance to that exact race coin, clipped at 15
    """
    if game_state is None:
        return None

    field = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]
    bombs = game_state.get("bombs", [])
    coins = game_state.get("coins", [])
    others = game_state.get("others", [])

    bomb_positions = {tuple(position) for position, _ in bombs}
    other_positions = {tuple(agent[3]) for agent in others}
    occupied = bomb_positions | other_positions
    danger = _danger_map(game_state)

    free = []
    crates_adjacent = []
    adjacent_crate_positions = []
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        free.append(int(_is_walkable(field, occupied, nx, ny)))
        has_crate = _in_bounds(field, nx, ny) and field[nx, ny] == 1
        crates_adjacent.append(int(has_crate))
        if has_crate:
            adjacent_crate_positions.append((nx, ny))

    danger_features = [int((x, y) in danger)]
    danger_features.extend(
        int((x + dx, y + dy) in danger) for dx, dy in DIRECTIONS
    )

    crates = [tuple(position) for position in zip(*np.where(field == 1))]
    opponent_positions = [tuple(agent[3]) for agent in others]
    # Nearby visible coins may temporarily outrank an opponent. Otherwise the
    # opponent remains primary, with crates and distant coins as fallbacks.
    target, target_kind = _nearest_target(
        x,
        y,
        coins,
        adjacent_crate_positions
        if (bombs_left > 0 and adjacent_crate_positions)
        else crates,
        opponent_positions,
        prefer_adjacent_crates=(
            bombs_left > 0
            and not opponent_positions
            and bool(adjacent_crate_positions)
        ),
    )
    target_is_coin = int(target_kind == 1)
    target_is_opponent = int(target_kind == 3)
    if target is None:
        target_dx = target_dy = target_distance = 0
    else:
        target_dx = int(np.sign(target[0] - x))
        target_dy = int(np.sign(target[1] - y))
        target_distance = min(
            abs(target[0] - x) + abs(target[1] - y),
            5,
        )

    safe_neighbors = sum(
        int(free[index] == 1 and danger_features[index + 1] == 0)
        for index in range(4)
    )

    escape_distance, escape_route_count = _bomb_escape_metrics(
        field,
        (x, y),
        occupied,
        danger,
    )
    escape_after_bomb = int(escape_distance > 0)
    blast = _blast_cells(field, (x, y))
    bomb_hits_target = int(
        any(field[cx, cy] == 1 for cx, cy in blast)
        or any(position in blast for position in opponent_positions)
    )
    opponent_adjacent = min(
        sum(
            int((x + dx, y + dy) in opponent_positions)
            for dx, dy in DIRECTIONS
        ),
        4,
    )
    accurate_blast = _real_blast_cells(field, (x, y))
    bomb_target_count = min(
        sum(int(field[cx, cy] == 1) for cx, cy in accurate_blast)
        + sum(int(position in accurate_blast) for position in opponent_positions),
        4,
    )
    path_first_actions, path_distance, path_target_kind = _path_route_features(
        field,
        (x, y),
        coins,
        crates,
        opponent_positions,
        occupied,
    )
    bomb_opponent_count, opponent_escape_routes = (
        _opponent_bomb_escape_features(
            field,
            (x, y),
            opponent_positions,
            occupied,
        )
    )
    (
        coin_race_first_actions,
        coin_race_distance,
        coin_race_margin,
    ) = _best_coin_race_features(
        field,
        (x, y),
        coins,
        opponent_positions,
        occupied,
    )

    return np.array(
        free
        + crates_adjacent
        + danger_features
        + [
            target_dx,
            target_dy,
            target_distance,
            target_is_coin,
            min(int(bombs_left), 1),
            min(safe_neighbors, 4),
            escape_after_bomb,
            bomb_hits_target,
            target_is_opponent,
            opponent_adjacent,
            *path_first_actions,
            min(path_distance, PATH_DISTANCE_LIMIT),
            int(path_target_kind == 1),
            int(path_target_kind == 3),
            min(len(opponent_positions), 3),
            min(len(crates), 40),
            min(len(coins), 9),
            bomb_target_count,
            min(escape_route_count, 4),
            bomb_opponent_count,
            min(opponent_escape_routes, 4),
            coin_race_margin,
            *coin_race_first_actions,
            coin_race_distance,
        ],
        dtype=np.int8,
    )

import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np


# Keep the same action order in callbacks.py and train.py.
ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
ACTION_INDICES = list(range(len(ACTIONS)))

# Task 3 keeps all six actions, including BOMB.
TASK3_ACTION_INDICES = ACTION_INDICES

ALPHA = 0.10       # learning rate
GAMMA = 0.95       # discount factor
EPSILON = 0.20     # exploration probability during training

# Keep the Task 3 table separate from the Task 1 and Task 2 tables.
MODEL_PATH = Path(__file__).resolve().parent / "q_table_task3.pkl"

# Feature indices used by helper functions and reward shaping.
TARGET_DISTANCE_INDEX = 15
BOMBS_LEFT_INDEX = 17
ESCAPE_AFTER_BOMB_INDEX = 19
BOMB_HITS_TARGET_INDEX = 20
TARGET_IS_OPPONENT_INDEX = 21
OPPONENT_ADJACENT_INDEX = 22

# Directions follow the action order UP, RIGHT, DOWN, LEFT.
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def state_to_key(features):
    """Convert a feature vector into a hashable Q-table key."""
    if features is None:
        return None
    return tuple(int(value) for value in features)


def setup(self):
    """Create a fresh Q-table for training or load one for evaluation."""
    if self.train:
        self.logger.info("Starting a new Task 3 Q-learning table.")
        self.model = {}
    elif MODEL_PATH.exists():
        self.logger.info("Loading Task 3 Q-table from %s", MODEL_PATH)
        with MODEL_PATH.open("rb") as file:
            loaded_model = pickle.load(file)
        self.model = loaded_model if isinstance(loaded_model, dict) else {}
    else:
        self.logger.warning("No Task 3 Q-table found; using an empty table.")
        self.model = {}


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
    # A bomb is useful when it can hit a target and a route exists to leave
    # the hypothetical blast area before the timer expires.  When an opponent
    # is within three cells, allow the bomb even if the current blast does not
    # yet overlap the opponent; this gives the agent a chance to attack while
    # it is approaching a moving target.
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


def act(self, game_state: dict) -> str:
    """Choose a Task 3 action with epsilon-greedy Q-learning."""
    features = state_to_features(game_state)
    if features is None:
        return "WAIT"

    state_key = state_to_key(features)
    if state_key not in self.model:
        self.model[state_key] = np.zeros(len(ACTIONS), dtype=np.float32)

    q_values = self.model[state_key]
    valid_indices = valid_action_indices(features)

    if self.train and random.random() < EPSILON:
        action_index = random.choice(valid_indices)
    else:
        valid_q_values = [q_values[index] for index in valid_indices]
        best_value = max(valid_q_values)
        best_indices = [
            index for index in valid_indices
            if q_values[index] == best_value
        ]
        # Random tie-breaking avoids a fixed preference for one action.
        action_index = random.choice(best_indices)

    return ACTIONS[action_index]


def _in_bounds(field, x, y):
    width, height = field.shape
    return 0 <= x < width and 0 <= y < height


def _is_walkable(field, occupied, x, y):
    """A walkable cell is floor and is not occupied by an agent or bomb."""
    return _in_bounds(field, x, y) and field[x, y] == 0 and (x, y) not in occupied


def _blast_cells(field, origin, power=3):
    """Return cells affected by a bomb, stopping at walls and crates."""
    cells = [origin]
    ox, oy = origin
    for dx, dy in DIRECTIONS:
        for distance in range(1, power + 1):
            x, y = ox + dx * distance, oy + dy * distance
            if not _in_bounds(field, x, y) or field[x, y] == -1:
                break
            cells.append((x, y))
            if field[x, y] == 1:  # crate is destroyed but blocks further blast
                break
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


def _escape_after_bomb(field, start, occupied, existing_danger, max_steps=4):
    """Return whether a short route exists out of a hypothetical blast.

    The agent may cross a future blast cell while the bomb is counting down,
    but must reach a walkable cell outside that blast within ``max_steps``.
    Existing bombs/explosions remain blocked throughout the search.
    """
    hypothetical_blast = set(_blast_cells(field, start))
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        (x, y), depth = queue.popleft()
        for dx, dy in DIRECTIONS:
            nx, ny = x + dx, y + dy
            position = (nx, ny)
            if position in visited or not _is_walkable(field, occupied, nx, ny):
                continue
            if position in existing_danger:
                continue
            visited.add(position)
            if position not in hypothetical_blast:
                return True
            if depth + 1 < max_steps:
                queue.append((position, depth + 1))

    return False


def _nearest_target(x, y, coins, crates, opponents, prefer_adjacent_crates=False):
    """Prefer opponents, then adjacent crates, coins, and other crates.

    Target kind: 1 = coin, 2 = crate, 3 = opponent.
    """
    targets = [(tuple(opponent), 3) for opponent in opponents]
    if not targets and prefer_adjacent_crates:
        targets = [(tuple(crate), 2) for crate in crates]
    if not targets:
        targets = [(tuple(coin), 1) for coin in coins]
    if not targets:
        targets = [(tuple(crate), 2) for crate in crates]
    if not targets:
        return None, 0

    target, target_kind = min(
        targets,
        key=lambda item: abs(item[0][0] - x) + abs(item[0][1] - y),
    )
    return target, int(target_kind)


def state_to_features(game_state: dict) -> np.ndarray:
    """Extract compact Task 3 features for tabular Q-learning.

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
    # In Task 3, an opponent is the primary target. If no opponent remains,
    # retain the Task 2 crate/coin behavior as a useful fallback.
    target, target_kind = _nearest_target(
        x,
        y,
        coins,
        adjacent_crate_positions if (bombs_left > 0 and adjacent_crate_positions) else crates,
        opponent_positions,
        prefer_adjacent_crates=(bombs_left > 0 and not opponent_positions and bool(adjacent_crate_positions)),
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

    escape_after_bomb = int(
        _escape_after_bomb(field, (x, y), occupied, danger)
    )
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
        ],
        dtype=np.int8,
    )

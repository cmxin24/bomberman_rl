import random
from collections import deque, namedtuple
from typing import List

import events as e
import numpy as np
import torch
from torch import nn

from .callbacks import (
    ACTIONS,
    BATCH_SIZE,
    DEVICE,
    DQN,
    EPSILON_DECAY,
    EPSILON_MIN,
    FEATURE_DIM,
    GAMMA,
    LEARNING_RATE,
    MODEL_PATH,
    REPLAY_CAPACITY,
    REPLAY_WARMUP,
    SAVE_INTERVAL,
    BOMB_HITS_TARGET_INDEX,
    BOMB_OPPONENT_COUNT_INDEX,
    OPPONENT_ESCAPE_ROUTE_COUNT_INDEX,
    PATH_DISTANCE_INDEX,
    PATH_TARGET_IS_COIN_INDEX,
    PATH_TARGET_IS_OPPONENT_INDEX,
    TARGET_DISTANCE_INDEX,
    TARGET_IS_COIN_INDEX,
    TARGET_IS_OPPONENT_INDEX,
    TARGET_UPDATE_INTERVAL,
    TRAIN_EVERY,
    normalize_features,
    state_to_features,
    valid_action_indices,
)


Transition = namedtuple(
    "Transition",
    ["state", "action", "next_state", "reward", "done", "discount"],
)
N_STEP = 5


def setup_training(self):
    """Create replay memory, target network, optimizer and counters."""
    self.replay_memory = deque(maxlen=REPLAY_CAPACITY)
    self.n_step_buffer = deque()
    self.target_net = DQN().to(DEVICE)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    trainable_parameters = [
        parameter
        for parameter in self.policy_net.parameters()
        if parameter.requires_grad
    ]
    self.optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )
    if self.checkpoint is not None and "optimizer_state_dict" in self.checkpoint:
        try:
            self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError) as error:
            self.logger.warning("Could not restore DQN optimizer state: %s", error)

    self.environment_steps = 0
    self.training_steps = (
        int(self.checkpoint.get("training_steps", 0))
        if self.checkpoint is not None
        else 0
    )
    self.last_loss = None


def _aggregate_n_step(transitions):
    """Aggregate up to four raw steps into one discounted transition."""
    if not transitions:
        raise ValueError("Cannot aggregate an empty transition sequence")

    total_reward = 0.0
    final_transition = transitions[0]
    steps_used = 0
    for step, transition in enumerate(transitions[:N_STEP]):
        total_reward += (GAMMA ** step) * float(transition.reward)
        final_transition = transition
        steps_used += 1
        if transition.done:
            break

    first = transitions[0]
    return Transition(
        state=first.state,
        action=first.action,
        next_state=final_transition.next_state,
        reward=total_reward,
        done=final_transition.done,
        discount=GAMMA ** steps_used,
    )


def _commit_oldest_n_step(self):
    """Commit the oldest buffered decision and advance one raw step."""
    aggregated = _aggregate_n_step(list(self.n_step_buffer))
    self.replay_memory.append(aggregated)
    self.n_step_buffer.popleft()


def _store_transition(self, old_features, action, new_features, reward, done):
    """Buffer a raw step and store four-step returns in replay memory."""
    if old_features is None or action not in ACTIONS:
        return

    raw_transition = Transition(
        state=np.asarray(old_features, dtype=np.int8).copy(),
        action=ACTIONS.index(action),
        next_state=(
            None
            if new_features is None
            else np.asarray(new_features, dtype=np.int8).copy()
        ),
        reward=float(reward),
        done=bool(done),
        discount=GAMMA,
    )
    self.n_step_buffer.append(raw_transition)
    if len(self.n_step_buffer) >= N_STEP:
        _commit_oldest_n_step(self)
    if done:
        while self.n_step_buffer:
            _commit_oldest_n_step(self)

    self.environment_steps += 1
    if self.environment_steps % TRAIN_EVERY == 0:
        _optimize_model(self)


def _optimize_model(self):
    """Train one Double-DQN minibatch sampled from replay memory."""
    if len(self.replay_memory) < max(BATCH_SIZE, REPLAY_WARMUP):
        return

    transitions = random.sample(self.replay_memory, BATCH_SIZE)
    states = torch.as_tensor(
        normalize_features(np.stack([item.state for item in transitions])),
        dtype=torch.float32,
        device=DEVICE,
    )
    actions = torch.as_tensor(
        [item.action for item in transitions], dtype=torch.long, device=DEVICE
    ).unsqueeze(1)
    rewards = torch.as_tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=DEVICE
    )
    done = torch.as_tensor(
        [item.done for item in transitions], dtype=torch.bool, device=DEVICE
    )
    discounts = torch.as_tensor(
        [item.discount for item in transitions],
        dtype=torch.float32,
        device=DEVICE,
    )

    predicted_q = self.policy_net(states).gather(1, actions).squeeze(1)
    next_values = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=DEVICE)

    non_terminal_indices = [
        index
        for index, item in enumerate(transitions)
        if not item.done and item.next_state is not None
    ]
    if non_terminal_indices:
        raw_next_states = np.stack(
            [transitions[index].next_state for index in non_terminal_indices]
        )
        next_states = torch.as_tensor(
            normalize_features(raw_next_states),
            dtype=torch.float32,
            device=DEVICE,
        )

        # The policy network selects the action while the slower target network
        # evaluates it. This is Double DQN and reduces optimistic Q estimates.
        with torch.no_grad():
            policy_next_q = self.policy_net(next_states)
            valid_mask = torch.zeros(
                (len(non_terminal_indices), len(ACTIONS)),
                dtype=torch.bool,
                device=DEVICE,
            )
            for row, raw_state in enumerate(raw_next_states):
                valid_mask[row, valid_action_indices(raw_state)] = True
            policy_next_q = policy_next_q.masked_fill(~valid_mask, -torch.inf)
            next_actions = policy_next_q.argmax(dim=1, keepdim=True)
            target_next_q = self.target_net(next_states)
            selected_next_q = target_next_q.gather(1, next_actions).squeeze(1)
            next_values[non_terminal_indices] = selected_next_q

    targets = rewards + (~done).float() * discounts * next_values
    loss = nn.functional.smooth_l1_loss(predicted_q, targets)

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
    self.optimizer.step()

    self.training_steps += 1
    self.last_loss = float(loss.item())
    if self.training_steps % TARGET_UPDATE_INTERVAL == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def game_events_occurred(
    self,
    old_game_state: dict,
    self_action: str,
    new_game_state: dict,
    events: List[str],
):
    """Store one game transition and perform an occasional DQN update."""
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)
    reward = reward_from_events(self, events, old_features, new_features)
    _store_transition(
        self, old_features, self_action, new_features, reward, done=False
    )


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """Store the terminal transition, decay epsilon and save a checkpoint."""
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events, last_features, None)
    _store_transition(self, last_features, last_action, None, reward, done=True)

    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
    self.episodes += 1

    if self.episodes == 1 or self.episodes % SAVE_INTERVAL == 0:
        _save_checkpoint(self)


def _save_checkpoint(self):
    """Persist DQN training state without touching older model files."""
    checkpoint = {
        "policy_state_dict": self.policy_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "epsilon": self.epsilon,
        "episodes": self.episodes,
        "training_steps": self.training_steps,
        "last_loss": self.last_loss,
        "feature_dim": FEATURE_DIM,
        "n_step": N_STEP,
        "actions": ACTIONS,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(MODEL_PATH)
    self.checkpoint = checkpoint
    self.logger.info(
        "Saved Task 4 DQN after %d episodes (epsilon=%.4f, loss=%s) to %s",
        self.episodes,
        self.epsilon,
        "n/a" if self.last_loss is None else f"{self.last_loss:.4f}",
        MODEL_PATH,
    )


def reward_from_events(
    self, events: List[str], old_features=None, new_features=None
) -> float:
    """Reward opponent hunting and survival while penalizing unsafe play."""
    event_rewards = {
        e.KILLED_OPPONENT: 15.0,
        e.OPPONENT_ELIMINATED: 3.0,
        e.CRATE_DESTROYED: 0.25,
        e.COIN_FOUND: 0.10,
        e.COIN_COLLECTED: 1.00,
        e.BOMB_DROPPED: -0.05,
        e.INVALID_ACTION: -0.30,
        e.WAITED: -0.05,
        e.KILLED_SELF: -10.0,
        e.GOT_KILLED: -8.0,
        e.SURVIVED_ROUND: 2.0,
    }

    reward = -0.01
    for event in events:
        reward += event_rewards.get(event, 0.0)

    if old_features is not None and new_features is not None:
        old_distance = int(old_features[TARGET_DISTANCE_INDEX])
        new_distance = int(new_features[TARGET_DISTANCE_INDEX])
        if (
            int(old_features[TARGET_IS_COIN_INDEX]) == 1
            and int(new_features[TARGET_IS_COIN_INDEX]) == 1
        ):
            reward += 0.20 * (old_distance - new_distance)
        elif (
            int(old_features[TARGET_IS_OPPONENT_INDEX]) == 1
            and int(new_features[TARGET_IS_OPPONENT_INDEX]) == 1
        ):
            reward += 0.25 * (old_distance - new_distance)
        else:
            reward += 0.05 * (old_distance - new_distance)

        if int(old_features[8]) == 1 and int(new_features[8]) == 0:
            reward += 0.5

        # Manhattan direction aliases states on opposite sides of walls. The
        # path-aware residual instead learns from progress along a real BFS
        # route. Clip one-step progress so target switches cannot dominate.
        old_path_distance = int(old_features[PATH_DISTANCE_INDEX])
        new_path_distance = int(new_features[PATH_DISTANCE_INDEX])
        same_coin_target = (
            int(old_features[PATH_TARGET_IS_COIN_INDEX]) == 1
            and int(new_features[PATH_TARGET_IS_COIN_INDEX]) == 1
            and e.COIN_COLLECTED not in events
        )
        same_opponent_target = (
            int(old_features[PATH_TARGET_IS_OPPONENT_INDEX]) == 1
            and int(new_features[PATH_TARGET_IS_OPPONENT_INDEX]) == 1
        )
        if old_path_distance > 0 and new_path_distance > 0:
            progress = float(
                np.clip(old_path_distance - new_path_distance, -1, 1)
            )
            if same_coin_target:
                reward += 0.18 * progress
            elif same_opponent_target:
                reward += 0.08 * progress
            elif not (
                int(old_features[PATH_TARGET_IS_COIN_INDEX])
                or int(old_features[PATH_TARGET_IS_OPPONENT_INDEX])
            ):
                reward += 0.06 * progress

    # Every bomb has an opportunity cost. Useful bombs retain only the small
    # base cost above; bombs not currently aligned with a crate or opponent
    # receive an additional penalty to improve bomb efficiency.
    if (
        e.BOMB_DROPPED in events
        and old_features is not None
        and int(old_features[BOMB_HITS_TARGET_INDEX]) == 0
    ):
        reward -= 0.15

    # Reward bombs that genuinely restrict an opponent's escape choices. The
    # five-step return still carries the eventual kill back to the bomb action.
    if e.BOMB_DROPPED in events and old_features is not None:
        threatened = int(old_features[BOMB_OPPONENT_COUNT_INDEX])
        opponent_routes = int(
            old_features[OPPONENT_ESCAPE_ROUTE_COUNT_INDEX]
        )
        if threatened > 0:
            reward += 0.10 * threatened * max(0, 3 - opponent_routes)

    return reward

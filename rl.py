import logging
import random
from typing import List
from functools import partial
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import control
import sumo


class History:
    def __init__(self, state, next_state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state


def get_vehicle_x_state(ego_car_position):
    """Discretize ego position."""
    ego_x = ego_car_position[0]
    if ego_x < -66:
        return 2
    elif ego_x < -51:
        return 1
    else:
        return 0


def get_vehicle_speed_state(ego_car_speed):
    """Discretize ego speed."""
    if ego_car_speed < 2:
        return 0
    elif ego_car_speed < 5:
        return 1
    elif ego_car_speed < 10:
        return 2
    elif ego_car_speed < 20:
        return 3
    elif ego_car_speed < 30:
        return 4
    else:
        return 5


def get_relative_x_state(ego_position, other_position):
    """Discretize front/back car position."""
    ego_x = ego_position[0]
    other_x = other_position[0]
    difference = abs(ego_x - other_x)
    if difference < 5:
        return 0
    elif difference < 10:
        return 1
    elif difference < 20:
        return 2
    elif difference < 50:
        return 3
    else:
        return 4


def get_relative_speed_state(ego_speed, other_speed):
    """Discretize front/back car speed."""
    relative_speed = other_speed - ego_speed
    if relative_speed < -15:
        return 0
    elif relative_speed < -5:
        return 1
    elif relative_speed < 0:
        return 2
    elif relative_speed < 5:
        return 3
    elif relative_speed < 15:
        return 4
    else:
        return 5


def get_jerk(action):
    return control.Settings.JERK_VALUES.get(action, 0)


def get_undiscretized_state():
    vehicle_ids = control.get_vehicle_ids()
    positions = control.get_vehicle_positions(vehicle_ids)
    speeds = control.get_vehicle_speeds(vehicle_ids)
    before_car, after_car = control.get_closest_vehicles(positions, speeds)
    before_position = positions.get(before_car, (-np.inf, 0))
    after_position = positions.get(after_car, (np.inf, 0))
    before_speed = speeds.get(before_car, 0)
    after_speed = speeds.get(after_car, 0)
    ego_position = positions.get("ego", (-np.inf, 0))
    ego_speed = speeds.get("ego", 0)
    logging.debug("Ego Pos: {}, Ego speed: {}".format(ego_position[0], ego_speed))
    logging.debug("Before Pos: {}, Before speed: {}".format(before_position[0], before_speed))
    logging.debug("After Pos: {}, After speed: {}".format(after_position[0], after_speed))
    return ego_position, ego_speed, before_position, after_position, before_speed, after_speed


def get_state_tuple():
    ego_position, ego_speed, before_position, after_position, before_speed, after_speed = get_undiscretized_state()
    x_state = get_vehicle_x_state(ego_position)
    speed_state = get_vehicle_speed_state(ego_speed)
    before_x_state = get_relative_x_state(ego_position, before_position)
    after_x_state = get_relative_x_state(ego_position, after_position)
    before_speed_state = get_relative_speed_state(ego_speed, before_speed)
    after_speed_state = get_relative_speed_state(ego_speed, after_speed)
    return x_state, speed_state, before_x_state, after_x_state, before_speed_state, after_speed_state


def get_q_values(q, state):
    return q[state]


def get_visited_q_values(q, visits, state):
    values = q[state]
    state_visits = visits[state]
    visited_modifier = np.zeros(values.size)
    visited_modifier[state_visits == 0] = np.inf
    return values - visited_modifier


def continuous_reward(state, jerk, crashed, arrived):

    if crashed:
        safety_metric = -100
        efficiency_metric = 0
        smoothness_metric = 0
    elif arrived:
        safety_metric = 0
        efficiency_metric = 0
        smoothness_metric = 0
    else:
        smoothness_metric = - abs(jerk)
        ego_position, ego_speed, before_position, after_position, before_speed, after_speed = state
        efficiency_metric = - control.Settings.TICK_LENGTH * abs(ego_speed - control.Settings.DESIRED_SPEED)

        front_distance = after_position[0] - ego_position[0] - control.Settings.CAR_LENGTH
        front_speed_diff = ego_speed - after_speed
        if front_distance < 0:
            time_to_collide_front = 0.01
        elif front_speed_diff > 0 and not np.isinf(front_distance):
            time_to_collide_front = front_distance / front_speed_diff
        else:
            time_to_collide_front = np.inf

        back_distance = ego_position[0] - before_position[0] - control.Settings.CAR_LENGTH
        back_speed_diff = before_speed - ego_speed
        if back_distance < 0:
            time_to_collide_back = 0.01
        elif back_speed_diff > 0 and not np.isinf(back_distance):
            time_to_collide_back = back_distance / back_speed_diff
        else:
            time_to_collide_back = np.inf

        front_ratio = min(time_to_collide_front / control.Settings.DESIRED_TTC, 1)
        back_ratio = min(time_to_collide_back / control.Settings.DESIRED_TTC, 1)

        safety_metric = (np.log(front_ratio) + np.log(back_ratio))

    return control.Settings.WT_SMOOTH * smoothness_metric + control.Settings.WT_SAFE * safety_metric + control.Settings.WT_EFFICIENT * efficiency_metric


def slotted_reward(state, jerk, crashed, arrived):
    if crashed:
        return control.Settings.CRASH_REWARD
    elif arrived:
        return control.Settings.SUCCESS_REWARD
    else:
        return control.Settings.TIME_REWARD * control.Settings.TICK_LENGTH


def get_control(next_state, Q, visits=None, epsilon=0.0):
    if random.random() < epsilon:
        next_action = random.choice(list(control.Settings.JERK_VALUES.keys()))
    else:
        if visits is not None and control.Settings.AVOID_UNVISITED_STATES:
            next_action = np.argmax(get_visited_q_values(Q, visits, next_state))
        else:
            next_action = np.argmax(get_q_values(Q, next_state))
    jerk = get_jerk(next_action)
    control.set_ego_jerk(jerk)
    return next_action


def get_state():
    return get_state_tuple()


def get_history(episode_stats, reward_function):
    jerk_history = episode_stats["jerk_history"]
    state_history = episode_stats["state_history"]
    control_history = episode_stats["control_history"]
    crashed = episode_stats["crashed"]
    merged = episode_stats["merged"]
    episode_history = []
    episode_length = len(state_history)
    for i in range(episode_length - 1):
        previous_state = state_history[i]
        next_state = state_history[i+1]
        previous_action = control_history[i]
        next_jerk = jerk_history[i+1]
        if i == episode_length - 2:
            current_crashed = crashed
            current_merged = merged
        else:
            current_crashed = False
            current_merged = False
        reward = reward_function(next_state, next_jerk, current_crashed, current_merged)
        episode_history.append(History(previous_state, next_state, previous_action, reward))
    return episode_history


def do_q_update(episode_history: List[History], Q, visits, discount_factor, step_size):
    for step in reversed(episode_history):
        state_action_pair = step.state + (step.action,)
        target = step.reward
        if step.next_state is not None:
            target += discount_factor * np.max(get_q_values(Q, step.next_state))
        visits[state_action_pair] += 1
        Q[state_action_pair] = (1 - step_size)*Q[state_action_pair] + step_size * target


def initialize_q():
    return np.zeros((3, 6, 5, 5, 6, 6, len(control.Settings.JERK_VALUES.keys())))


def load_q_model(model_name):
    return np.load("{}.npy".format(model_name))


def get_rl_custom_episode_stats(episode_stats, reward_function):
    episode_history = get_history(episode_stats, reward_function)
    total_reward = sum([history.reward for history in episode_history])
    custom_stats = {
        "total_reward": total_reward
    }
    return custom_stats


def evaluate_q_model_and_log_metrics(Q, iteration, writer, visits, reward_function=slotted_reward):
    sumo.change_step_size(control.Settings.EVALUATION_TICK_LENGTH)
    control_function = partial(get_control, Q=Q, visits=visits, epsilon=0.0)
    evaluation_stats = control.evaluate_control(
        control_function=control_function,
        state_function=get_state_tuple,
        custom_stats_function=partial(get_rl_custom_episode_stats, reward_function=reward_function),
        max_episode_length=control.Settings.EVALUATION_EPISODE_LENGTH,
        num_episodes=control.Settings.NUM_EVALUATION_EPISODES,
        wait_before_start=20
    )
    metrics = evaluation_stats.get_stat_averages()
    metrics["unvisited_states"] = visits.size - np.count_nonzero(visits)

    for metric in metrics:
        writer.add_scalar(metric, metrics[metric], iteration)
    logging.info(metrics)
    sumo.change_step_size(control.Settings.TRAINING_TICK_LENGTH)


def learn_q_model(model_name):
    if control.Settings.REWARD_FUNCTION == "Continuous":
        reward_function = continuous_reward
    elif control.Settings.REWARD_FUNCTION == "Slotted":
        reward_function = slotted_reward
    else:
        raise ValueError("Invalid reward function {} specified in settings.".format(control.Settings.REWARD_FUNCTION))
    writer = SummaryWriter(comment=model_name)
    for key, value in control.Settings.export_settings().items():
        writer.add_text(key, str(value))

    if control.Settings.INIT_MODEL_NAME:
        Q = load_q_model(control.Settings.INIT_MODEL_NAME)
    else:
        Q = initialize_q()
        
    visits = initialize_q()

    for i in tqdm(range(control.Settings.NUM_TRAINING_EPISODES)):
        if i % control.Settings.EVALUATION_PERIOD == 0 and i != 0:
            logging.info("Evaluating model at training episode {}".format(i))
            evaluate_q_model_and_log_metrics(Q, i, writer, visits, reward_function=reward_function)

        control_function = partial(get_control, Q=Q, visits=visits, epsilon=1.0)
        episode_state = control.run_episode(
            control_function=control_function,
            state_function=get_state_tuple,
            max_episode_length=control.Settings.MAX_EPISODE_LENGTH,
            wait_before_start=20,
            limit_metrics=True
        )
        episode_history = get_history(episode_state, reward_function)
        do_q_update(episode_history, Q, visits, control.Settings.GAMMA, control.Settings.STEP_SIZE)

        if i % control.Settings.STEP_SIZE_HALF_PER_EPISODES == 0 and i != 0:
            control.Settings.STEP_SIZE /= 2

    evaluate_q_model_and_log_metrics(Q, control.Settings.NUM_TRAINING_EPISODES, writer, visits, reward_function=reward_function)
    np.save(model_name, Q)
    writer.close()

import copy
import logging
import os
import random
from collections import deque
from distutils.dir_util import copy_tree
from functools import partial
from shutil import rmtree
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import control
import rl
import sumo
from config import Settings
from stats import StatsAggregator
import st
import prediction


class RLAgent(ABC):
    """
    An abstract agent that controls the car using RL
    """

    def __init__(self):
        self.takeover_history = []
        self.all_xs = []
        self.takeover_xs = []

    @classmethod
    @abstractmethod
    def load(cls, path):
        """
        Load the agent from file
        :param path: Path to the directory containing the agent's parameters
        :return: A fully initialized RLAgent object
        """
        pass

    @classmethod
    @abstractmethod
    def train(cls, num_frames: int):
        """
        Train the agent from scratch and save the network in the agent's run directory
        :param num_frames: The number of simulation frames to use to train the agent
        :return: None
        """
        pass

    @classmethod
    @abstractmethod
    def resume_training(cls, path, num_frames: int):
        """
        Train the agent, starting from an already trained model at the given path
        :param path: The path of the directory containing the agent's parameters (same as RLAgent.load)
        :param num_frames: The number of simulation frames to use to train the agent
        :return: None
        """
        pass

    @abstractmethod
    def get_control(self, state: prediction.HighwayState) -> float:
        """
        Get the control signal (in jerk) to use for the current state
        :param state: The input state of the traffic system
        :return: The jerk produced by the RL model, in m/s
        """
        pass

    def _setup(self):
        """
        This gets run before a testing run to set up the environment
        :return: None
        """
        self.takeover_history = []
        self.all_xs = []
        self.takeover_xs = []

    def _cleanup(self):
        """
        This gets run after a testing run to clean up the environment
        :return: None
        """
        pass

    def do_control(self, state: prediction.HighwayState) -> float:
        return control.set_ego_jerk(self.get_control(state))

    def end_episode_callback(self, last_state):
        pass

    def combined_stats_callback(self, episode_stats):
        # We are just calculating the percent of time the ST solver takes over in the combination
        position_history = episode_stats["position_history"]
        total_num = len(position_history)
        takeover_num = 0
        for i, position in enumerate(position_history):
            self.all_xs.append(position[0])
            if self.takeover_history[i]:
                takeover_num += 1
                self.takeover_xs.append(position[0])
            else:
                pass
        percent_st = takeover_num / max(total_num, 1)
        self.takeover_history.clear()
        return {"percent st solver": percent_st}

    def do_combined_control(self, state: prediction.HighwayState) -> float:
        start_state = state
        first_action = self.get_control(state)
        future_action = first_action
        rollout_s_history = [control.get_ego_s(start_state.ego_position)]
        crash_predicted = False
        test_state = None
        selected_speed = 0
        last_choice_rl = True
        if len(self.takeover_history) > 0 and self.takeover_history[-1]:
            last_choice_rl = False
        i = 0
        while not (crash_predicted or i >= max(Settings.ROLLOUT_LENGTH, 1)):
            i += 1
            if i != 1:
                future_action = self.get_control(state)
            current_speed = state.ego_speed
            current_acceleration = state.ego_acceleration
            selected_speed = control.get_ego_speed_from_jerk(current_speed, current_acceleration, future_action)
            state, crash_predicted = state.predict_step_with_ego(selected_speed, Settings.TICK_LENGTH, Settings.COMBINATION_MIN_DISTANCE)
            if i == Settings.ST_TEST_ROLLOUTS:
                test_state = state
            rollout_s_history.append(control.get_ego_s(state.ego_position))
            if state.ego_position[0] > Settings.STOP_X:
                break
        if test_state is None:
            test_state = state
        if Settings.CHECK_ROLLOUT_CRASH and crash_predicted:
            print("Crash predicted")
            self.takeover_history.append(True)
            return st.do_st_control(start_state)
        elif Settings.LIMIT_DQN_SPEED and selected_speed > Settings.DESIRED_SPEED:
            print("DDPG going too fast")
            self.takeover_history.append(True)
            return st.do_st_control(start_state)
        elif Settings.TEST_ROLLOUT_STATE and st.test_guaranteed_crash_from_state(test_state):
            print("ST solver not happy with rollout state")
            self.takeover_history.append(True)
            return st.do_st_control(start_state)
        elif Settings.TEST_ST_STRICTLY_BETTER:
            s_sequence, obstacles, s_values, t_values, distances = st.get_appropriate_base_st_path_and_obstacles(start_state)
            end_point = len(s_sequence)
            while s_sequence[end_point - 1] == 0:
                end_point -= 1
            s_sequence = s_sequence[:end_point]
            if Settings.TICK_LENGTH < Settings.T_DISCRETIZATION:
                s_sequence = st.finer_fit(s_sequence, Settings.TICK_LENGTH, Settings.T_DISCRETIZATION, start_state.ego_speed,
                                          start_state.ego_acceleration)

            # We are essentially on top of another car, so there's not much we can do anyway
            if len(s_sequence) <= 1:
                self.takeover_history.append(False)
                return control.set_ego_jerk(first_action)

            min_planning_length = min(len(s_sequence), len(rollout_s_history))
            st_jerk = st.get_path_mean_abs_jerk(s_sequence[:min_planning_length], start_state.ego_speed, start_state.ego_acceleration, Settings.TICK_LENGTH)
            rl_jerk = st.get_path_mean_abs_jerk(rollout_s_history[:min_planning_length], start_state.ego_speed, start_state.ego_acceleration, Settings.TICK_LENGTH)
            st_distance = s_sequence[min_planning_length - 1] - s_sequence[0]
            rl_distance = rollout_s_history[min_planning_length - 1] - rollout_s_history[0]
            if last_choice_rl or not Settings.REMEMBER_LAST_CHOICE_FOR_SWITCHING_COMBINED:
                if (st_jerk < rl_jerk and st_distance > rl_distance) or rl_distance == 0:
                    print("ST Path deemed better")
                    self.takeover_history.append(True)
                    planned_distance_first_step = s_sequence[1] - s_sequence[0]
                    end_speed_first_step = planned_distance_first_step / (Settings.TICK_LENGTH)
                    control.set_ego_speed(end_speed_first_step)
                    return end_speed_first_step
                else:
                    self.takeover_history.append(False)
                    return control.set_ego_jerk(first_action)
            else:
                if rl_jerk < st_jerk and rl_distance > st_distance:
                    print("RL path deemed better")
                    self.takeover_history.append(False)
                    return control.set_ego_jerk(first_action)
                else:
                    self.takeover_history.append(True)
                    planned_distance_first_step = s_sequence[1] - s_sequence[0]
                    end_speed_first_step = planned_distance_first_step / (Settings.TICK_LENGTH)
                    control.set_ego_speed(end_speed_first_step)
                    return end_speed_first_step
        else:
            self.takeover_history.append(False)
            return control.set_ego_jerk(first_action)

    def evaluate(self, num_episodes):
        self._setup()
        output = control.evaluate_control(
            control_function=self.do_control,
            state_function=prediction.HighwayState.from_sumo,
            crash_callback=None,
            num_episodes=num_episodes,
            end_episode_callback=self.end_episode_callback,
            verbose=True
        )
        output.print_stats()
        self._cleanup()

    def plot_st_proportion(self):
        default_bins = np.arange(-220, 61, 20)
        hist_all, bins = np.histogram(self.all_xs, bins=default_bins)
        hist_st, _ = np.histogram(self.takeover_xs, bins=default_bins)
        proportions = hist_st / hist_all
        plt.figure()
        plt.bar(bins[:-1], proportions, width=np.diff(bins), edgecolor="black", align="edge")
        plt.savefig(os.path.join(Settings.FULL_LOG_DIR, "plots", "proportion_using_st"))
        plt.close()
        logging.info("ST solver usage: ")
        logging.info(proportions)
        logging.info(bins)

    def evaluate_combined(self, num_episodes):
        self._setup()
        output = control.evaluate_control(
            control_function=self.do_combined_control,
            state_function=prediction.HighwayState.from_sumo,
            crash_callback=None,
            end_episode_callback=self.end_episode_callback,
            num_episodes=num_episodes,
            custom_stats_function=self.combined_stats_callback,
            verbose=True
        )
        self._cleanup()
        output.print_stats()
        self.plot_st_proportion()


class DQNAgent(RLAgent):

    def __init__(self, model=None):
        super().__init__()
        self.model = model
        if self.model is None:
            self.model = DQN(dropout=Settings.USE_DROPOUT)

    @classmethod
    def load(cls, path):
        agent = cls(DQN.load(path))
        return agent

    def _train(self, num_frames: int):
        dqn = self.model
        writer = SummaryWriter(log_dir=Settings.FULL_LOG_DIR)

        reward_function = get_reward_function()
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(dqn.parameters(), lr=Settings.LEARNING_RATE)
        target_dqn = copy.deepcopy(dqn)
        target_dqn.eval()  # No dropout for the target network

        if Settings.USE_PRIORITIZED_ER:
            history = SumTree(capacity=Settings.REPLAY_BUFFER_SIZE)
        else:
            history = deque(maxlen=Settings.REPLAY_BUFFER_SIZE)

        for iteration in tqdm(range(Settings.NUM_TRAINING_EPISODES)):

            # Decay the chance to make a random move to a minimum of 0.1
            epsilon = Settings.EPS_END + (Settings.EPS_START - Settings.EPS_END) * np.exp(
                -Settings.EPS_DECAY_COEFFICIENT * np.floor(iteration / Settings.EPS_DECAY_RATE))

            if iteration % Settings.TARGET_NET_FREEZE_PERIOD == 0 and iteration != 0:
                target_dqn = copy.deepcopy(dqn)
                target_dqn.eval()

            if iteration % Settings.EVALUATION_PERIOD == 0 and iteration != 0:
                evaluate_q_model_and_log_metrics(dqn, iteration, writer, reward_function)
                writer.add_scalar("epsilon", epsilon, iteration)
                dqn.checkpoint("checkpoint_{}".format(iteration))

            control_function = partial(do_dqn_control, dqn=dqn, epsilon=epsilon)
            episode_metrics = control.run_episode(
                control_function=control_function,
                state_function=prediction.HighwayState.from_sumo,
                max_episode_length=Settings.TRAINING_EPISODE_LENGTH,
                limit_metrics=True
            )

            # Turn the state history into state vectors from the state objects
            state_history = episode_metrics["state_history"]
            state_history = [get_state_vector_from_base_state(state) for state in state_history]
            episode_metrics["state_history"] = state_history

            episode_history = rl.get_history(episode_metrics, reward_function)

            if Settings.USE_PRIORITIZED_ER:
                for item in episode_history:
                    history.add_node(item, Settings.PER_MAX_PRIORITY ** Settings.PER_ALPHA)
            else:
                history.extend(episode_history)

            if iteration % 10 == 0:
                writer.add_scalar("Length", len(episode_history), iteration)

            total_loss = 0
            for train_index in range(Settings.TRAINING_STEPS_PER_EPISODE):
                # Choose a (state, action, reward, state) tuple from some random trajectories in the replay buffer
                if Settings.USE_PRIORITIZED_ER:
                    train_sars = []
                    train_indices = []
                    for k in range(min(len(history), Settings.BATCH_SIZE)):
                        position, sars = history.sample()
                        train_sars.append(sars)
                        train_indices.append(position)
                else:
                    train_sars = random.choices(history, k=min(len(history), Settings.BATCH_SIZE))
                    train_indices = []

                # Calculate target = r + gamma * max_a q(s+, a)
                targets = get_targets(train_sars, dqn, target_dqn, gamma=Settings.DISCOUNT_FACTOR)
                target_tensor = dqn.get_target_tensor_bulk(targets)

                # Convert the states and actions to pytorch tensors
                state_tensor = dqn.get_q_tensor_bulk([item.state for item in train_sars])
                action_tensor = dqn.get_action_tensor_bulk([item.action for item in train_sars]).reshape((-1, 1))

                optimizer.zero_grad()

                # Calculate Q(s, a)
                outputs = dqn.forward(state_tensor)
                q_values = torch.gather(outputs, 1, action_tensor).flatten()

                # Gradient descent step
                loss = criterion(q_values, target_tensor)
                loss.backward()
                optimizer.step()

                if Settings.USE_PRIORITIZED_ER:
                    td_errors = torch.abs(q_values - target_tensor)
                    for error_index, error in enumerate(td_errors):
                        priority = min(error + Settings.PER_MIN_PRIORITY,
                                       Settings.PER_MAX_PRIORITY) ** Settings.PER_ALPHA
                        history.update_weight(priority, train_indices[error_index])

                total_loss += loss

            if iteration % 10 == 0:
                writer.add_scalar("Loss", total_loss / Settings.TRAINING_STEPS_PER_EPISODE, iteration)

        evaluate_q_model_and_log_metrics(dqn, Settings.NUM_TRAINING_EPISODES, writer, reward_function)

        dqn.save()
        writer.close()

    @classmethod
    def train(cls, num_frames: int):
        agent = cls()
        agent._setup()
        agent._train(num_frames)
        agent._cleanup()

    @classmethod
    def resume_training(cls, path, num_frames: int):
        agent = cls.load(path)
        agent._setup()
        agent._train(num_frames)
        agent._cleanup()

    def get_control(self, state: prediction.HighwayState) -> float:
        with torch.no_grad():
            state_vector = get_state_vector_from_base_state(state)
            qs = self.model.get_q_values(state_vector)
            action = torch.argmax(qs).item()
        return self.model.get_jerk(action)

    def _setup(self):
        sumo.start_sumo()

    def _cleanup(self):
        sumo.exit_sumo()


def get_state_vector_from_base_state(state: prediction.HighwayState, cars_ahead=None, cars_behind=None):
    if cars_ahead is None or cars_behind is None:
        cars_ahead = Settings.CARS_AHEAD
        cars_behind = Settings.CARS_BEHIND
    front_cars = []
    back_cars = []
    ego_x = state.ego_position[0]
    ego_speed = state.ego_speed
    for i, vehicle_speed in enumerate(state.other_speeds):
        vehicle_x = state.other_xs[i]
        if Settings.USE_ACCELERATION_OF_OTHER_CARS:
            vehicle_acceleration = state.other_accelerations[i]
            if Settings.USE_SPEED_DIFFERENCE:
                vehicle_tuple = np.array([vehicle_acceleration, vehicle_speed - ego_speed, vehicle_x - ego_x, 1])
            else:
                vehicle_tuple = np.array([vehicle_acceleration, vehicle_speed, vehicle_x - ego_x, 1])

        else:
            if Settings.USE_SPEED_DIFFERENCE:
                vehicle_tuple = np.array([vehicle_speed - ego_speed, vehicle_x - ego_x, 1])
            else:
                vehicle_tuple = np.array([vehicle_speed, vehicle_x - ego_x, 1])
        if vehicle_x > ego_x:
            front_cars.append(vehicle_tuple)
        else:
            back_cars.append(vehicle_tuple)
    front_cars.reverse()
    if Settings.USE_ACCELERATION_OF_OTHER_CARS:
        buffer_tuple = np.zeros(4)
    else:
        buffer_tuple = np.zeros(3)
    while len(front_cars) < cars_ahead:
        front_cars.append(buffer_tuple)
    while len(back_cars) < cars_behind:
        back_cars.append(buffer_tuple)
    vehicle_tuples = front_cars[:cars_ahead]
    vehicle_tuples.extend(back_cars[:cars_behind])
    if Settings.NORMALIZE_VECTOR_INPUT:
        for item in vehicle_tuples:
            if Settings.USE_ACCELERATION_OF_OTHER_CARS:
                item[0] /= 9
                offset = 1
            else:
                offset = 0
            item[offset] /= Settings.MAX_SPEED
            offset += 1
            item[offset] /= Settings.SENSOR_RADIUS
    if Settings.NORMALIZE_VECTOR_INPUT:
        vehicle_tuples.append(np.array([ego_speed / Settings.MAX_SPEED, state.ego_acceleration / 9]))
        position = np.array(state.ego_position)
        position[0] /= 300
        position[1] /= 100
        vehicle_tuples.append(position)
    else:
        vehicle_tuples.append(np.array([ego_speed, state.ego_acceleration]))
        vehicle_tuples.append(np.array(state.ego_position))
    flattened = np.concatenate(vehicle_tuples)
    return flattened


def get_reward_function():
    if Settings.REWARD_FUNCTION == "Continuous":
        reward_function = continuous_reward
    elif Settings.REWARD_FUNCTION == "Slotted":
        reward_function = rl.slotted_reward
    elif Settings.REWARD_FUNCTION == "Slotted Jerk":
        reward_function = slotted_reward_with_jerk
    elif Settings.REWARD_FUNCTION == "ST":
        reward_function = st_reward
    else:
        raise ValueError("Invalid reward function {} specified in settings.".format(Settings.REWARD_FUNCTION))
    return reward_function


def continuous_reward(state: prediction.HighwayState, jerk, crashed, arrived):
    absolute_metric = 0
    safety_metric = 0
    efficiency_metric = 0
    smoothness_metric = 0

    if crashed:
        absolute_metric = -10
    elif arrived:
        absolute_metric = 10
    else:
        smoothness_metric = - abs(jerk) * Settings.TICK_LENGTH
        ego_speed = state.ego_speed
        car_ahead, car_behind = state.get_closest_cars()
        ego_position = state.ego_position
        ego_x, ego_y = ego_position
        ego_s = control.get_ego_s(ego_position)

        if ego_s > 0:
            if car_ahead is not None:
                ahead_x, ahead_speed, ahead_acceleration = car_ahead
                front_distance = ahead_x - ego_x - Settings.CAR_LENGTH
            else:
                front_distance = np.inf
            if car_behind is not None:
                behind_x, behind_speed, behind_acceleration = car_behind
                back_distance = ego_x - behind_x - Settings.CAR_LENGTH
            else:
                back_distance = np.inf

            min_distance = min(front_distance, back_distance)
            if min_distance < Settings.MIN_FOLLOW_DISTANCE:
                safety_metric = -1
            elif min_distance == np.inf:
                safety_metric = 0
            elif np.isnan(min_distance):
                safety_metric = 0
            else:
                safety_metric = -1 / min_distance
            safety_metric *= Settings.TICK_LENGTH
        efficiency_metric = -Settings.TICK_LENGTH * np.abs(ego_speed - Settings.DESIRED_SPEED)

    return Settings.WT_SMOOTH * smoothness_metric + Settings.WT_SAFE * safety_metric + Settings.WT_EFFICIENT * efficiency_metric + absolute_metric


def st_reward(state, jerk, crashed, arrived):
    absolute_metric = 0
    speed_metric = 0
    acceleration_metric = 0
    jerk_metric = 0
    distance_metric = 0

    if crashed:
        absolute_metric = -10
    elif arrived:
        absolute_metric = 10
    else:
        jerk_metric = -jerk ** 2 * Settings.TICK_LENGTH
        ego_speed = state.ego_speed
        car_ahead, car_behind = state.get_closest_cars()
        ego_acceleration = state.ego_acceleration
        ego_position = state.ego_position
        ego_x, ego_y = ego_position
        ego_s = control.get_ego_s(ego_position)
        speed_metric = -Settings.TICK_LENGTH * (ego_speed - Settings.DESIRED_SPEED) ** 2
        acceleration_metric = -Settings.TICK_LENGTH * ego_acceleration ** 2

        if ego_s > 0:
            if car_ahead is not None:
                ahead_x, ahead_speed, ahead_acceleration = car_ahead
                front_distance = ahead_x - ego_x - Settings.CAR_LENGTH
            else:
                front_distance = np.inf
            if car_behind is not None:
                behind_x, behind_speed, behind_acceleration = car_behind
                back_distance = ego_x - behind_x - Settings.CAR_LENGTH
            else:
                back_distance = np.inf

            min_distance = min(front_distance, back_distance)
            if min_distance < Settings.MIN_FOLLOW_DISTANCE:
                distance_metric = -2 / max(min_distance, 1)
            elif min_distance == np.inf:
                distance_metric = 0
            elif np.isnan(min_distance):
                distance_metric = 0
            else:
                distance_metric = -1 / min_distance
            distance_metric *= Settings.TICK_LENGTH

    return Settings.ALT_A_WEIGHT * acceleration_metric + Settings.ALT_D_WEIGHT * distance_metric + \
           Settings.ALT_J_WEIGHT * jerk_metric + Settings.ALT_V_WEIGHT * speed_metric + absolute_metric


def slotted_reward_with_jerk(state, jerk, crashed, arrived):
    if crashed:
        return Settings.CRASH_REWARD
    elif arrived:
        return Settings.SUCCESS_REWARD
    else:
        return Settings.TIME_REWARD * Settings.TICK_LENGTH - Settings.ALT_J_WEIGHT * jerk**2 * Settings.TICK_LENGTH


class DQN(nn.Module):

    def __init__(self, num_hidden=2, hidden_size=256, num_outputs=5, jerk_values=Settings.JERK_VALUES_DQN,
                 dropout=False):
        super().__init__()
        self.jerk_values = jerk_values
        self.num_outputs = num_outputs
        self.use_dropout = dropout
        if Settings.USE_ACCELERATION_OF_OTHER_CARS:
            self.first_hidden_layer = nn.Linear(4 + 4 * (Settings.CARS_AHEAD + Settings.CARS_BEHIND), hidden_size)
        else:
            self.first_hidden_layer = nn.Linear(4 + 3 * (Settings.CARS_AHEAD + Settings.CARS_BEHIND), hidden_size)
        self.hidden_layers = [nn.Linear(hidden_size, hidden_size) for i in range(num_hidden - 2)]
        self.last_hidden_layer = nn.Linear(hidden_size, num_outputs)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if self.use_dropout:
            output = F.relu(self.dropout(self.first_hidden_layer(x)))
        else:
            output = F.relu(self.first_hidden_layer(x))
        for layer in self.hidden_layers:
            if self.use_dropout:
                output = F.relu(self.dropout(layer(output)))
            else:
                output = F.relu(layer(output))
        output = self.last_hidden_layer(output)
        return output

    def get_q_values(self, state_vector):
        tensor = self.get_q_tensor(state_vector)
        return self.forward(tensor)

    @staticmethod
    def get_q_tensor_bulk(state_vectors):
        stacked = np.stack(state_vectors, axis=0)
        tensor = torch.from_numpy(stacked).float()
        if Settings.CUDA:
            tensor.cuda()
        return tensor

    @staticmethod
    def get_target_tensor_bulk(targets):
        tensor = torch.from_numpy(np.asarray(targets)).float()
        if Settings.CUDA:
            tensor.cuda()
        return tensor

    @staticmethod
    def get_q_tensor(state_vector):
        tensor = torch.from_numpy(state_vector).float()
        if Settings.CUDA:
            tensor.cuda()
        return tensor

    @staticmethod
    def get_target_tensor(target):
        tensor = torch.tensor(target).float()
        if Settings.CUDA:
            tensor.cuda()
        return tensor

    @staticmethod
    def get_action_tensor_bulk(actions):
        tensor = torch.tensor(actions).long()
        if Settings.CUDA:
            tensor.cuda()
        return tensor

    def save(self, name=""):
        self.ensure_model_directory()
        torch.save(self, os.path.join(Settings.FULL_LOG_DIR, "q.pt"))
        torch.save(self, os.path.join("models", "q.pt"))

    def checkpoint(self, name):
        self.ensure_model_directory()
        torch.save(self, os.path.join(Settings.FULL_LOG_DIR, "{}.pt".format(name)))

    @classmethod
    def load(cls, path):
        net = torch.load(os.path.join(path, "q.pt".format(path)))
        net.eval()
        return net

    @staticmethod
    def ensure_model_directory():
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists(os.path.join(Settings.FULL_LOG_DIR, "models")):
            os.makedirs(os.path.join(Settings.FULL_LOG_DIR, "models"))

    def get_jerk(self, action):
        return Settings.JERK_VALUES[action]


def do_dqn_control(state: prediction.HighwayState, dqn, epsilon):
    if random.random() < epsilon:
        current_action = random.randrange(dqn.num_outputs)
    else:
        with torch.no_grad():
            state_vector = get_state_vector_from_base_state(state)
            qs = dqn.get_q_values(state_vector)
            current_action = torch.argmax(qs).item()
    control.set_ego_jerk(dqn.get_jerk(current_action))
    return current_action


def get_targets(sars_tuples, current_network, target_network, gamma=0.99):
    intermediate_states = []
    intermediate_rewards = []
    intermediate_indices = []
    end_indices = []
    end_rewards = []
    for index, sars in enumerate(sars_tuples):
        if sars.next_state is None:
            end_rewards.append(sars.reward)
            end_indices.append(index)
        else:
            intermediate_states.append(sars.next_state)
            intermediate_rewards.append(sars.reward)
            intermediate_indices.append(index)
    with torch.no_grad():
        state_tensors = current_network.get_q_tensor_bulk(intermediate_states)
        if Settings.DOUBLE_DQN:
            qs = current_network.forward(state_tensors)
            best_indices = torch.argmax(qs, dim=1, keepdim=True)
            qs2 = target_network.forward(state_tensors)
            intermediate_targets = gamma * torch.gather(qs2, 1, best_indices).flatten().numpy()
        else:
            qs = target_network.forward(state_tensors) * gamma
            intermediate_targets = torch.max(qs, dim=1)[0].numpy()
        if Settings.CLIP_TARGETS:
            intermediate_targets = np.clip(intermediate_targets, Settings.CLIP_MIN_REWARD, Settings.CLIP_MAX_REWARD)
    for i, reward in enumerate(intermediate_rewards):
        intermediate_targets[i] += reward
    end_targets = np.array(end_rewards)
    targets = np.zeros(len(sars_tuples))
    targets[intermediate_indices] = intermediate_targets
    targets[end_indices] = end_targets
    return targets


def evaluate_q_model_and_log_metrics(dqn, iteration, writer, reward_function):
    sumo.change_step_size(Settings.EVALUATION_TICK_LENGTH)
    control_function = partial(do_dqn_control, dqn=dqn, epsilon=0.0)
    evaluation_stats = control.evaluate_control(
        control_function=control_function,
        state_function=prediction.HighwayState.from_sumo,
        custom_stats_function=partial(rl.get_rl_custom_episode_stats, reward_function=reward_function),
        max_episode_length=Settings.EVALUATION_EPISODE_LENGTH,
        num_episodes=Settings.NUM_EVALUATION_EPISODES,
        wait_before_start=20
    )
    metrics = evaluation_stats.get_stat_averages()

    for metric in metrics:
        writer.add_scalar(metric, metrics[metric], iteration)
    logging.info(metrics)
    sumo.change_step_size(Settings.TRAINING_TICK_LENGTH)


class SumTree:

    def __init__(self, capacity):
        self.capacity = 1
        # This implementation assumes a full binary tree
        while self.capacity < capacity:
            self.capacity *= 2
        self.weights = np.zeros(2 * self.capacity - 1)
        self.samples = np.zeros(self.capacity, dtype=np.object)
        self.size = 0
        self.current_index = 0

    def __getitem__(self, position):
        return self.samples[position]

    def __len__(self):
        return self.size

    def add_node(self, item, weight):
        if self.size < self.capacity:
            self.size += 1
        self.set_node(item, weight, self.current_index)
        self.current_index = (self.current_index + 1) % self.capacity

    def set_node(self, item, weight, position):
        if position >= self.size:
            raise IndexError("Requested position is out of range")
        self.samples[position] = item
        self.update_weight(weight, position)

    def get_leaf_index(self, position):
        return position + self.capacity - 1

    def update_weight(self, weight, position):
        if weight < 0:
            raise ValueError("This SumTree is designed for positive weights.")
        weight_index = self.get_leaf_index(position)
        self.weights[weight_index] = weight
        self._update_sum((weight_index - 1) // 2)

    def _update_sum(self, index):
        child_sum = self.weights[index * 2 + 1] + self.weights[index * 2 + 2]
        self.weights[index] = child_sum
        if index != 0:
            self._update_sum((index - 1) // 2)

    def _add_weight(self, index, weight_difference):
        self.weights[index] += weight_difference
        if index != 0:
            self._add_weight((index - 1) // 2, weight_difference)

    def sample(self):
        roll = random.uniform(0, self.weights[0])
        return self._traverse_tree(0, roll)

    def _traverse_tree(self, index, roll):
        if index >= self.capacity - 1:
            position = index - self.capacity + 1
            if position >= self.size:
                logging.log(logging.WARN, "Something went wrong with the SumTree. Trying again instead of crashing.")
                return self.sample()
            return position, self.samples[position]
        left_index = index * 2 + 1
        if roll <= self.weights[left_index]:
            # Left child
            return self._traverse_tree(left_index, roll)
        else:
            return self._traverse_tree(index * 2 + 2, roll - self.weights[left_index])


def train(model_name):
    writer = SummaryWriter(log_dir=Settings.FULL_LOG_DIR)
    for key, value in Settings.export_settings().items():
        writer.add_text(key, str(value))

    if Settings.INIT_MODEL_NAME:
        dqn = DQN.load(Settings.INIT_MODEL_NAME)
    else:
        dqn = DQN(dropout=Settings.USE_DROPOUT)

    reward_function = get_reward_function()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(dqn.parameters(), lr=Settings.LEARNING_RATE)
    target_dqn = copy.deepcopy(dqn)
    target_dqn.eval()  # No dropout for the target network

    if Settings.USE_PRIORITIZED_ER:
        history = SumTree(capacity=Settings.REPLAY_BUFFER_SIZE)
    else:
        history = deque(maxlen=Settings.REPLAY_BUFFER_SIZE)

    for iteration in tqdm(range(Settings.NUM_TRAINING_EPISODES)):

        # Decay the chance to make a random move to a minimum of 0.1
        epsilon = Settings.EPS_END + (Settings.EPS_START - Settings.EPS_END) * np.exp(
            -Settings.EPS_DECAY_COEFFICIENT * np.floor(iteration / Settings.EPS_DECAY_RATE))

        if iteration % Settings.TARGET_NET_FREEZE_PERIOD == 0 and iteration != 0:
            target_dqn = copy.deepcopy(dqn)
            target_dqn.eval()

        if iteration % Settings.EVALUATION_PERIOD == 0 and iteration != 0:
            evaluate_q_model_and_log_metrics(dqn, iteration, writer, reward_function)
            writer.add_scalar("epsilon", epsilon, iteration)
            dqn.checkpoint("{}_checkpoint_{}".format(model_name, iteration))

        control_function = partial(do_dqn_control, dqn=dqn, epsilon=epsilon)
        episode_metrics = control.run_episode(
            control_function=control_function,
            state_function=prediction.HighwayState.from_sumo,
            max_episode_length=Settings.TRAINING_EPISODE_LENGTH,
            limit_metrics=True
        )
        episode_history = rl.get_history(episode_metrics, reward_function)

        if Settings.USE_PRIORITIZED_ER:
            for item in episode_history:
                history.add_node(item, Settings.PER_MAX_PRIORITY ** Settings.PER_ALPHA)
        else:
            history.extend(episode_history)

        if iteration % 10 == 0:
            writer.add_scalar("Length", len(episode_history), iteration)

        total_loss = 0
        for train_index in range(Settings.TRAINING_STEPS_PER_EPISODE):
            # Choose a (state, action, reward, state) tuple from some random trajectories in the replay buffer
            if Settings.USE_PRIORITIZED_ER:
                train_sars = []
                train_indices = []
                for k in range(min(len(history), Settings.BATCH_SIZE)):
                    position, sars = history.sample()
                    train_sars.append(sars)
                    train_indices.append(position)
            else:
                train_sars = random.choices(history, k=min(len(history), Settings.BATCH_SIZE))
                train_indices = []

            # Calculate target = r + gamma * max_a q(s+, a)
            targets = get_targets(train_sars, dqn, target_dqn, gamma=Settings.DISCOUNT_FACTOR)
            target_tensor = dqn.get_target_tensor_bulk(targets)

            # Convert the states and actions to pytorch tensors
            state_tensor = dqn.get_q_tensor_bulk([item.state for item in train_sars])
            action_tensor = dqn.get_action_tensor_bulk([item.action for item in train_sars]).reshape((-1, 1))

            optimizer.zero_grad()

            # Calculate Q(s, a)
            outputs = dqn.forward(state_tensor)
            q_values = torch.gather(outputs, 1, action_tensor).flatten()

            # Gradient descent step
            loss = criterion(q_values, target_tensor)
            loss.backward()
            optimizer.step()

            if Settings.USE_PRIORITIZED_ER:
                td_errors = torch.abs(q_values - target_tensor)
                for error_index, error in enumerate(td_errors):
                    priority = min(error + Settings.PER_MIN_PRIORITY, Settings.PER_MAX_PRIORITY) ** Settings.PER_ALPHA
                    history.update_weight(priority, train_indices[error_index])

            total_loss += loss

        if iteration % 10 == 0:
            writer.add_scalar("Loss", total_loss / Settings.TRAINING_STEPS_PER_EPISODE, iteration)

    evaluate_q_model_and_log_metrics(dqn, Settings.NUM_TRAINING_EPISODES, writer, reward_function)

    dqn.save(model_name)
    writer.close()


def evaluate(model_name, num_iterations):
    dqn = DQN.load(model_name)
    control_function = partial(do_dqn_control, dqn=dqn, epsilon=0.0)
    reward_function = get_reward_function()
    evaluation_stats = control.evaluate_control(
        control_function=control_function,
        state_function=prediction.HighwayState.from_sumo,
        custom_stats_function=partial(rl.get_rl_custom_episode_stats, reward_function=reward_function),
        max_episode_length=Settings.EVALUATION_EPISODE_LENGTH,
        num_episodes=num_iterations,
        wait_before_start=20
    )
    evaluation_stats.print_stats()


def get_tabular_state_from_dqn_state(dqn_state):
    ego_x = dqn_state[-2]
    ego_speed = dqn_state[-4]
    ego_position = (ego_x, 0)

    if dqn_state[2]:
        after_position = (dqn_state[1] + ego_x, 0)
        after_speed = dqn_state[0]
    else:
        after_position = (np.inf, 0)
        after_speed = 0

    if dqn_state[11]:
        before_position = (dqn_state[10] + ego_x, 0)
        before_speed = dqn_state[9]
    else:
        before_position = (-np.inf, 0)
        before_speed = 0

    x_state = rl.get_vehicle_x_state(ego_position)
    speed_state = rl.get_vehicle_speed_state(ego_speed)
    before_x_state = rl.get_relative_x_state(ego_position, before_position)
    after_x_state = rl.get_relative_x_state(ego_position, after_position)
    before_speed_state = rl.get_relative_speed_state(ego_speed, before_speed)
    after_speed_state = rl.get_relative_speed_state(ego_speed, after_speed)

    return x_state, speed_state, before_x_state, after_x_state, before_speed_state, after_speed_state


def get_targets_from_tabular_Q(sars_tuples, tabular_Q):
    targets = []
    for sars in sars_tuples:
        if sars.next_state is None:
            targets.append(sars.reward)
        else:
            rl_state = get_tabular_state_from_dqn_state(sars.next_state)
            targets.append(sars.reward + Settings.GAMMA * np.max(rl.get_q_values(tabular_Q, rl_state)))
    return np.array(targets)


def train_to_tabular_target(model_name):
    writer = SummaryWriter(log_dir=Settings.FULL_LOG_DIR)
    for key, value in Settings.export_settings().items():
        writer.add_text(key, str(value))

    if Settings.INIT_MODEL_NAME:
        dqn = DQN.load(Settings.INIT_MODEL_NAME)
    else:
        dqn = DQN(dropout=Settings.USE_DROPOUT)

    reward_function = get_reward_function()

    criterion = nn.SmoothL1Loss()
    optimizer = optim.RMSprop(dqn.parameters())

    # Set target Q based on tabular
    tabular_Q = rl.load_q_model("q_values_slotted_5action")

    if Settings.USE_PRIORITIZED_ER:
        history = SumTree(capacity=Settings.REPLAY_BUFFER_SIZE)
    else:
        history = deque(maxlen=Settings.REPLAY_BUFFER_SIZE)

    for iteration in tqdm(range(Settings.NUM_TRAINING_EPISODES)):

        # Decay the chance to make a random move to a minimum of 0.1
        epsilon = Settings.EPS_END + (Settings.EPS_START - Settings.EPS_END) * np.exp(
            -Settings.EPS_DECAY_COEFFICIENT * np.floor(iteration / Settings.EPS_DECAY_RATE))

        if iteration % Settings.EVALUATION_PERIOD == 0 and iteration != 0:
            evaluate_q_model_and_log_metrics(dqn, iteration, writer, reward_function)
            dqn.checkpoint("{}_checkpoint_{}".format(model_name, iteration))

        control_function = partial(do_dqn_control, dqn=dqn, epsilon=epsilon)
        episode_metrics = control.run_episode(
            control_function=control_function,
            state_function=prediction.HighwayState.from_sumo,
            max_episode_length=Settings.TRAINING_EPISODE_LENGTH,
            limit_metrics=True
        )
        episode_history = rl.get_history(episode_metrics, reward_function)

        if Settings.USE_PRIORITIZED_ER:
            for item in episode_history:
                history.add_node(item, Settings.PER_MAX_PRIORITY ** Settings.PER_ALPHA)
        else:
            history.extend(episode_history)

        if iteration % 10 == 0:
            writer.add_scalar("Length", len(episode_history), iteration)

        total_loss = 0
        for train_index in range(Settings.TRAINING_STEPS_PER_EPISODE):
            # Choose a (state, action, reward, state) tuple from some random trajectories in the replay buffer
            if Settings.USE_PRIORITIZED_ER:
                train_sars = []
                train_indices = []
                for k in range(min(len(history), Settings.BATCH_SIZE)):
                    position, sars = history.sample()
                    train_sars.append(sars)
                    train_indices.append(position)
            else:
                train_sars = random.choices(history, k=min(len(history), Settings.BATCH_SIZE))
                train_indices = []

            # Calculate target = r + gamma * max_a q(s+, a)
            targets = get_targets_from_tabular_Q(train_sars, tabular_Q)
            target_tensor = dqn.get_target_tensor_bulk(targets)

            # Convert the states and actions to pytorch tensors
            state_tensor = dqn.get_q_tensor_bulk([item.state for item in train_sars])
            action_tensor = dqn.get_action_tensor_bulk([item.action for item in train_sars]).reshape((-1, 1))

            optimizer.zero_grad()

            # Calculate Q(s, a)
            outputs = dqn.forward(state_tensor)
            q_values = torch.gather(outputs, 1, action_tensor).flatten()

            # Gradient descent step
            loss = criterion(q_values, target_tensor)
            loss.backward()
            optimizer.step()

            if Settings.USE_PRIORITIZED_ER:
                td_errors = torch.abs(q_values - target_tensor)
                for error_index, error in enumerate(td_errors):
                    priority = min(error + Settings.PER_MIN_PRIORITY, Settings.PER_MAX_PRIORITY) ** Settings.PER_ALPHA
                    history.update_weight(priority, train_indices[error_index])

            total_loss += loss

        if iteration % 10 == 0:
            writer.add_scalar("Loss", total_loss / Settings.TRAINING_STEPS_PER_EPISODE, iteration)

    evaluate_q_model_and_log_metrics(dqn, Settings.NUM_TRAINING_EPISODES, writer, reward_function)

    dqn.save(model_name)
    writer.close()


def train_dqn_all():
    from all.environments import GymEnvironment
    from all.presets.classic_control import ddqn
    from all.experiments import SingleEnvExperiment

    if Settings.CUDA:
        device = "cuda"
    else:
        device = "cpu"

    env = GymEnvironment(Settings.GYM_ENVIRONMENT, device=device)
    preset = ddqn(
        device=device,
        lr=Settings.LEARNING_RATE,
        initial_exploration=Settings.EPS_START,
        final_exploration=Settings.EPS_END
    )
    experiment = SingleEnvExperiment(preset, env)
    experiment.train(1E6)
    default_log_dir = experiment._writer.log_dir
    copy_tree(default_log_dir, Settings.FULL_LOG_DIR)
    rmtree(default_log_dir)


def resume_dqn_all():
    from all.presets.classic_control import ddqn
    from all.environments import GymEnvironment
    from all.experiments import SingleEnvExperiment

    if Settings.CUDA:
        device = "cuda"
    else:
        device = "cpu"

    env = GymEnvironment('sumo-jerk-v0', device=device)
    lr = 1e-5
    agent = ddqn(device=device, lr=lr)
    q_module = torch.load(os.path.join('models', "q.pt"), map_location='cpu').to(device)

    experiment = SingleEnvExperiment(agent, env)
    agent = experiment._agent
    old_q = agent.q
    old_q.model.load_state_dict(q_module.state_dict())
    experiment.train(frames=1e6)
    default_log_dir = experiment._writer.log_dir
    copy_tree(default_log_dir, Settings.FULL_LOG_DIR)
    rmtree(default_log_dir)


def evaluate_dqn_all(num_test_episodes):
    from all.experiments.watch import GreedyAgent
    from all.environments import GymEnvironment

    if Settings.CUDA:
        device = "cuda"
    else:
        device = "cpu"

    env = GymEnvironment(Settings.GYM_ENVIRONMENT, device=device)
    agent = GreedyAgent.load('models', env)
    num_crashed = 0
    num_arrived = 0
    action = None
    iteration = 0

    rlstats = StatsAggregator()
    episode_reward = 0

    def add_reward(state):
        return {"reward": episode_reward}

    rlstats.add_custom_stat_callback(add_reward)
    rewards = []

    while iteration < num_test_episodes:
        if env.done:
            actualEnv = env.env
            stats = actualEnv.get_stats()
            if len(stats["position_history"]) != 0:
                rlstats.add_episode_stats(stats)
                num_crashed += stats["crashed"]
                num_arrived += stats["merged"]
                iteration += 1
                print(iteration)
                rewards.append(episode_reward)
                episode_reward = 0
            env.reset()
        else:
            env.step(action)
        action = agent.eval(env.state, env.reward)
        episode_reward += env.reward

    logging.info("Rewards: {}".format(rewards))
    rlstats.print_stats()


if __name__ == "__main__":

    # Just a quick test for SumTree, each entry from 1 to 8 has probability proportional to 1/i
    tree = SumTree(8)
    for i in range(1, tree.capacity + 1):
        tree.add_node(i, 1 / i)
    samples = {}
    for i in range(100000):
        random_sample = tree.sample()[1]
        samples[random_sample] = samples.get(random_sample, 0) + 1
    print(samples)
    for sample in samples:
        print(sample, samples[sample] / samples[1])

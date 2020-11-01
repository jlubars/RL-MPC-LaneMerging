import torch
import os
from distutils.dir_util import copy_tree
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
import functools
from typing import List
import logging
from all.presets.continuous import ddpg
from all.experiments import SingleEnvExperiment
from all.experiments.watch import GreedyAgent
from all.environments import GymEnvironment
from all.bodies.time import TimeFeature

from config import Settings
import control
import dqn
import st
import prediction
from dqn import RLAgent


class DDPGAgent(dqn.RLAgent):

    def __init__(self):
        super().__init__()

        if Settings.CUDA:
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.env = GymEnvironment(Settings.GYM_ENVIRONMENT, device=self.device)
        self.agent = None

    @classmethod
    def load(cls, path) -> RLAgent:
        rl_agent = cls()
        agent = GreedyAgent.load(path, rl_agent.env)
        agent = TimeFeature(agent)
        rl_agent.env._lazy_init()
        rl_agent.agent = agent
        return rl_agent

    @classmethod
    def train(cls, num_frames: int):
        rl_agent = cls()
        preset = ddpg(
            device=rl_agent.device,
            lr_q=Settings.LEARNING_RATE,
            lr_pi=Settings.LEARNING_RATE
        )
        experiment = SingleEnvExperiment(preset, rl_agent.env)
        experiment.train(num_frames)
        default_log_dir = experiment._writer.log_dir
        copy_tree(default_log_dir, Settings.FULL_LOG_DIR)
        rmtree(default_log_dir)
        rl_agent.env.close()

    @classmethod
    def resume_training(cls, path, num_frames: int):
        rl_agent = cls()
        lr = Settings.LEARNING_RATE
        agent = ddpg(device=rl_agent.device, lr_q=lr, lr_pi=lr)
        q_module = torch.load(os.path.join(path, "q.pt"), map_location='cpu').to(rl_agent.device)
        policy_module = torch.load(os.path.join(path, "policy.pt"), map_location='cpu').to(rl_agent.device)

        experiment = SingleEnvExperiment(agent, rl_agent.env)
        agent = experiment._agent.agent
        old_q = agent.q
        old_q.model.load_state_dict(q_module.state_dict())
        old_policy = agent.policy
        old_policy.model.load_state_dict(policy_module.state_dict())
        experiment.train(frames=num_frames)
        default_log_dir = experiment._writer.log_dir
        copy_tree(default_log_dir, Settings.FULL_LOG_DIR)
        rmtree(default_log_dir)
        rl_agent.env.close()

    def get_control(self, state: prediction.HighwayState) -> float:
        vector_state = dqn.get_state_vector_from_base_state(state)
        encoded_state = self.env._make_state(vector_state, False)
        first_action = self.agent.eval(encoded_state, 0).item()
        return first_action

    def end_episode_callback(self, last_state):
        self.agent.eval(self.env._make_state(dqn.get_state_vector_from_base_state(last_state), True), 0)

    def _cleanup(self):
        self.env.close()


def train_ddpg_all_with_lr_drop(num_frames, third=False):
    DDPGAgent.train(num_frames)
    Settings.LEARNING_RATE /= 10
    old_log_dir = Settings.FULL_LOG_DIR
    Settings.LOG_DIR = Settings.LOG_DIR + "_extended"
    Settings.setup_logging()
    DDPGAgent.resume_training(old_log_dir, num_frames)
    if third:
        Settings.TASK = "EVALUATE_DDPG"
        Settings.MODEL_NAME = Settings.FULL_LOG_DIR
        eval_agent = DDPGAgent.load(Settings.FULL_LOG_DIR)
        eval_agent.evaluate(Settings.NUM_EPISODES)
        Settings.TASK = "TRAIN_DDPG"
        Settings.LEARNING_RATE /= 10
        old_log_dir = Settings.FULL_LOG_DIR
        Settings.LOG_DIR = Settings.LOG_DIR + "2"
        Settings.setup_logging()
        DDPGAgent.resume_training(old_log_dir, num_frames)
    Settings.TASK = "EVALUATE_DDPG"
    Settings.MODEL_NAME = Settings.FULL_LOG_DIR
    eval_agent = DDPGAgent.load(Settings.FULL_LOG_DIR)
    eval_agent.evaluate(Settings.NUM_EPISODES)


def plot_rollouts(states: List[prediction.HighwayState], agent, env):
    Settings.ensure_run_plot_directory()
    plot_directory = os.path.join(Settings.FULL_LOG_DIR, "plots")
    for j, state in enumerate(states):
        start_state = state
        plt.figure()
        vector_state = dqn.get_state_vector_from_base_state(state)
        encoded_state = env._make_state(vector_state, False)
        first_action = agent.eval(encoded_state, 0).item()
        future_action = first_action
        crash_predicted = False
        i = 0
        state.plot_state(i)
        state_history = []
        while not (crash_predicted or i >= max(Settings.ROLLOUT_LENGTH, 1)):
            i += 1
            if i != 1:
                vector_state = dqn.get_state_vector_from_base_state(state)
                future_action = agent.eval(env._make_state(vector_state, False), 0).item()
            current_speed = state.ego_speed
            current_acceleration = state.ego_acceleration
            selected_speed = control.get_ego_speed_from_jerk(current_speed, current_acceleration,
                                                             future_action)
            state, crash_predicted = state.predict_step_with_ego(selected_speed, Settings.TICK_LENGTH)
            if not crash_predicted:
                state_history.append(state)
            state.plot_state(i)
        plt.savefig("{}/{}".format(plot_directory, j))
        plt.close()
        for i, state in enumerate(state_history):
            s_sequence, obstacles, s_values, t_values, distances = st.get_appropriate_base_st_path_and_obstacles(state)
            st.plot_s_path(obstacles, s_values, t_values, s_sequence)
            plt.savefig("{}/rollout_st_{}_{}".format(plot_directory, j, i))
            if s_sequence[-1] == 0:
                print("Crash predicted {} {}".format(j, i))
            plt.close()

        # Get the ST prediction
        s_sequence, obstacles, s_values, t_values, distances = st.get_appropriate_base_st_path_and_obstacles(start_state)
        st.plot_s_path(obstacles, s_values, t_values, s_sequence)
        plt.savefig("{}/st_{}".format(plot_directory, j))
        plt.close()
    print("Saved crash.")

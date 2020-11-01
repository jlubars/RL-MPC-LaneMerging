import inspect
import logging
import torch
import os


class Settings:

    # Task
    TASK = "ST"  # "ST" or "TRAIN_DQN" or "RESUME_DQN" or "EVALUATE_DQN" or "EVALUATE_COMBINED_DDPG", etc.
    NUM_EPISODES = 2000
    GYM_ENVIRONMENT = "sumo-jerk-continuous-v0"  # "sumo-jerk-continuous-v0" or "sumo-jerk-v0", probably

    # Logging
    LOG_DIR = "last_run"
    FULL_LOG_DIR = "runs"
    LOG_FILE = "out.log"
    LOG_LEVEL = logging.INFO
    MODEL_NAME = "runs/ddpg_simple_traffic_vary_start_extended"
    INIT_MODEL_NAME = ""

    # Randomness
    SEED = "Random"  # An int value or "Random"

    # Sumo
    USE_GUI = False
    SYSTEM = "Linux"  # or Windows

    # Simulation
    TICK_LENGTH = 0.2
    MAX_POSITIVE_ACCELERATION = 4.5
    MAX_NEGATIVE_ACCELERATION = -6.0
    MINIMUM_NEGATIVE_JERK = -5.0
    MAXIMUM_POSITIVE_JERK = 5.0
    MAX_SPEED = 30
    MERGE_POINT_X = -50
    CAR_LENGTH = 5.0
    USE_ALTERNATE_TRAFFIC_DISTRIBUTION = False
    USE_SIMPLE_TRAFFIC_DISTRIBUTION = True
    TRAFFIC_DENSITY = "low"  # "low" or "medium" or "high"

    # Simple traffic distribution
    VARY_TRAFFIC_START_TIMES = True
    BASE_TRAFFIC_INTERVAL = 1.2
    OTHER_CAR_SPEED = 7.0

    # Sensors
    SENSOR_RADIUS = 125
    USE_ACCELERATION_OF_OTHER_CARS = True

    # Random start speed
    START_SPEED = 15
    RANDOMIZE_START_SPEED = True
    START_SPEED_VARIANCE = 5
    MIN_START_SPEED = 5
    MAX_START_SPEED = 25

    # Reward Functions
    REWARD_FUNCTION = "Continuous"           # "Continuous" or "Slotted" or "ST" or "Slotted Jerk"

    # Parameters for slotted Reward
    CRASH_REWARD = -10
    SUCCESS_REWARD = 10
    TIME_REWARD = -0.1
    # Parameters for Continuous Reward
    WT_SMOOTH = 0.1
    WT_SAFE = 0.1
    WT_EFFICIENT = 0.01
    DESIRED_TTC = 3
    # Parameters for alternate continuous reward
    MIN_FOLLOW_DISTANCE = 3
    # Parameters for alternate ST reward
    ALT_V_WEIGHT = 0.0001
    ALT_A_WEIGHT = 0.01
    ALT_J_WEIGHT = 0.05  # Also used for slotted reward + jerk
    ALT_D_WEIGHT = 0.05

    # Tabular RL
    JERK_VALUES = {0: -5, 1: -2.5, 2: 0, 3: 2.5, 4: 5}
    TRAINING_TICK_LENGTH = 0.2
    MAX_EPISODE_LENGTH = 100
    STEP_SIZE = 0.01
    GAMMA = 1.0
    NUM_TRAINING_EPISODES = 150000
    STEP_SIZE_HALF_PER_EPISODES = 20000
    # Parameters for evaluation
    EVALUATION_PERIOD = 2000            # Evaluate after this many training episodes
    NUM_EVALUATION_EPISODES = 100       # Evaluate over this many episodes
    EVALUATION_EPISODE_LENGTH = 50      # Evaluation epsiode length in seconds
    EVALUATION_TICK_LENGTH = 0.2
    AVOID_UNVISITED_STATES = True

    # S-T Solver and Continuous Reward
    DESIRED_SPEED = 30.0

    # S-T Solver
    USE_CYTHON = True
    USE_FAST_ST_SOLVER = True
    S_DISCRETIZATION = 0.05
    T_DISCRETIZATION = 0.30
    FUTURE_S = 150.0
    FUTURE_T = 5.0
    START_UNCERTAINTY = 0.0  # Increases the length of cars by this amount, plus the amount below per second
    UNCERTAINTY_PER_SECOND = 0.0
    V_WEIGHT = 0.5
    A_WEIGHT = 10.0
    J_WEIGHT = 10.0
    D_WEIGHT = 10.0
    MIN_ALLOWED_DISTANCE = 5
    CRASH_MIN_S = 12

    # DQN
    CUDA = torch.cuda.is_available()
    JERK_VALUES_DQN = {0: -5, 1: -2.5, 2: 0, 3: 2.5, 4: 5}
    ACCELERATION_VALUES_DQN = {0: -6.0, 1: -5.5, 2: -5.0, 3: -4.5, 4: -4.0, 5: -3.0, 6: -2.5, 7: -2.0, 8: -1.0, 9: -0.5, 10: 0.0, 11: 0.5, 12: 1.0, 13: 1.5, 14: 2.0, 15: 2.5, 16: 3.0, 17: 3.5, 18: 4.0, 19: 4.5}
    REPLAY_BUFFER_SIZE = 50000
    DISCOUNT_FACTOR = 0.999
    BATCH_SIZE = 50
    TRAINING_EPISODE_LENGTH = 50
    TRAINING_STEPS_PER_EPISODE = 8
    TARGET_NET_FREEZE_PERIOD = 500
    LEARNING_RATE = 2e-4
    USE_PRIORITIZED_ER = True
    PER_MAX_PRIORITY = 4.0
    PER_ALPHA = 0.5
    PER_MIN_PRIORITY = 1E-6
    EPS_DECAY_RATE = 30000
    EPS_DECAY_COEFFICIENT = 0.25
    EPS_START = 1.0
    EPS_END = 0.1
    USE_DROPOUT = False
    DOUBLE_DQN = True
    CLIP_TARGETS = True
    CLIP_MAX_REWARD = 10
    CLIP_MIN_REWARD = -20
    CARS_AHEAD = 2
    CARS_BEHIND = 2
    USE_SPEED_DIFFERENCE = True
    NORMALIZE_VECTOR_INPUT = True
    INVALID_ACTION_PENALTY = 0.0

    # Prediction
    MAX_PREDICTED_DECELERATION = -4

    # Combined
    ROLLOUT_LENGTH = 5
    ST_TEST_ROLLOUTS = 5
    USE_MIN_ALLOWED_DISTANCE_IN_COMBINED_SOLVER = True
    LIMIT_DQN_SPEED = False
    TEST_ST_STRICTLY_BETTER = True
    TEST_ROLLOUT_STATE = True
    CHECK_ROLLOUT_CRASH = True
    COMBINATION_MIN_DISTANCE = 5.1
    STOP_X = 65
    REMEMBER_LAST_CHOICE_FOR_SWITCHING_COMBINED = False

    @classmethod
    def export_settings(cls):
        return {x[0]: x[1] for x in inspect.getmembers(cls, lambda m: not inspect.isroutine(m)) if not x[0].startswith('__')}

    @classmethod
    def load_from_file(cls, filename):
        import json
        file = open(filename, 'rb')
        contents = json.load(file)
        for item in contents:
            value = contents[item]
            if isinstance(value, dict):
                value = {int(x): value[x] for x in value}
            setattr(cls, item, value)

    @staticmethod
    def dump_src(srcdir):
        from shutil import copyfile
        for filename in os.listdir(os.curdir):
            if filename.endswith(".py"):
                copyfile(filename, os.path.join(srcdir, filename))

    @classmethod
    def setup_logging(cls):
        import json
        logdir = os.path.join("runs", cls.LOG_DIR)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        srcdir = os.path.join(logdir, "src")
        if not os.path.exists(srcdir):
            os.makedirs(srcdir)
        cls.dump_src(srcdir)
        logfile = os.path.join(logdir, cls.LOG_FILE)
        settings_file = os.path.join(logdir, "settings.json")
        logging.basicConfig(filename=logfile, level=cls.LOG_LEVEL)
        json.dump(cls.export_settings(), open(settings_file, 'w'), indent=4, sort_keys=True)
        cls.FULL_LOG_DIR = logdir

    @classmethod
    def ensure_run_plot_directory(cls):
        plotdir = cls.get_run_plot_directory()
        if not os.path.exists(plotdir):
            os.mkdir(plotdir)

    @classmethod
    def get_run_plot_directory(cls):
        return os.path.join(cls.FULL_LOG_DIR, "plots")

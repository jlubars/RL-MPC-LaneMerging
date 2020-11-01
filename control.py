import os
import math
import sys
import time
from typing import Callable, Any, Optional, Dict
import random

import numpy as np
from scipy.stats import sem

from config import Settings
import stats
import prediction
import st


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


delay = 0


def get_new_xy(x, y, angle, speed):
    x_speed = speed * np.cos((90 - angle) / 360 * 2 * np.pi)
    y_speed = speed * np.sin((90 - angle) / 360 * 2 * np.pi)
    x += traci.simulation.getDeltaT() * x_speed
    y += traci.simulation.getDeltaT() * y_speed
    return x, y


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def add_ego_car(start_velocity):
    traci.vehicle.add("ego", "rampRoute", "egoCar", departSpeed=start_velocity, departPos=40, arrivalPos=50)
    traci.vehicle.setSpeedMode("ego", 22)
    traci.vehicle.setSpeed("ego", start_velocity)


def remove_ego_car():
    if 'ego' in traci.vehicle.getIDList():
        traci.vehicle.remove("ego")


def get_vehicle_ids():
    return traci.vehicle.getIDList()


def get_vehicle_positions(vehicle_ids):
    return {vehicle_id: traci.vehicle.getPosition(vehicle_id) for vehicle_id in vehicle_ids}


def get_vehicle_speeds(vehicle_ids):
    return {vehicle_id: traci.vehicle.getSpeed(vehicle_id) for vehicle_id in vehicle_ids}


def get_vehicle_accelerations(vehicle_ids):
    return {vehicle_id: traci.vehicle.getAcceleration(vehicle_id) for vehicle_id in vehicle_ids}


def get_closest_vehicles(positions, speeds, use_speeds=True):
    ego_position = positions.get("ego", (-math.inf, 0))
    ego_x = ego_position[0]
    ego_speed = speeds.get("ego", 0)
    distance_to_merge_point = max(Settings.MERGE_POINT_X - ego_x, 0)
    if ego_speed > 0:
        time_to_merge = distance_to_merge_point / ego_speed
    else:
        time_to_merge = math.inf

    before_car = None
    after_car = None
    before_x = -math.inf
    after_x = math.inf

    for car in positions:
        if car == "ego":
            continue
        car_position = positions[car]
        if distance(car_position, ego_position) > Settings.SENSOR_RADIUS:
            continue
        car_x = car_position[0]
        if use_speeds:
            car_speed = speeds.get(car, 0)
            if car_x + (car_speed - ego_speed) * time_to_merge <= ego_x:
                if car_x > before_x:
                    before_car = car
                    before_x = car_x
            else:
                if car_x + (car_speed - ego_speed) * time_to_merge < after_x:
                    after_car = car
                    after_x = car_x
        else:
            if car_x <= ego_x:
                if car_x > before_x:
                    before_car = car
                    before_x = car_x
            else:
                if car_x < after_x:
                    after_car = car
                    after_x = car_x
    return before_car, after_car


def get_closest_x_vehicles(positions, speeds, num_vehicles):
    ego_position = positions.get("ego", (-200, 0))
    ego_x, ego_y = ego_position
    vehicle_tuples = []
    for car_id in positions:
        if car_id != "ego":
            car_x, car_y = positions[car_id]
            car_speed = speeds[car_id]
            vehicle_tuples.append((car_speed, car_x - ego_x, car_y - ego_y, 1))
    vehicle_tuples.sort(key=lambda x: abs(x[1]))
    while len(vehicle_tuples) < num_vehicles:
        vehicle_tuples.append((0, 0, 0, 0))
    return vehicle_tuples[:num_vehicles]


def get_closest_x_vehicles_back_front(positions, speeds, num_back, num_front):
    ego_position = positions.get("ego", (-200, 0))
    ego_x, ego_y = ego_position
    ego_speed = speeds.get("ego", 0)
    vehicle_tuples = []
    front_vehicles = []
    back_vehicles = []
    for car_id in positions:
        if car_id == "ego":
            continue
        car_x, car_y = positions[car_id]
        car_speed = speeds[car_id]
        if car_x > ego_x:
            if Settings.USE_SPEED_DIFFERENCE:
                front_vehicles.append((car_speed - ego_speed, car_x - ego_x, 1))
            else:
                front_vehicles.append((car_speed, car_x - ego_x, 1))
        else:
            if Settings.USE_SPEED_DIFFERENCE:
                back_vehicles.append((car_speed - ego_speed, car_x - ego_x, 1))
            else:
                back_vehicles.append((car_speed, car_x - ego_x, 1))
    front_vehicles.sort(key=lambda x: abs(x[1]))
    back_vehicles.sort(key=lambda x: abs(x[1]))
    while len(front_vehicles) < num_front:
        front_vehicles.append((0, 0, 0))
    while len(back_vehicles) < num_back:
        back_vehicles.append((0, 0, 0))
    vehicle_tuples.extend(front_vehicles[:num_front])
    vehicle_tuples.extend(back_vehicles[:num_back])
    return vehicle_tuples


def get_ego_speed_from_jerk(current_speed, current_acceleration, jerk):
    new_acceleration = current_acceleration + jerk * Settings.TICK_LENGTH
    if new_acceleration > Settings.MAX_POSITIVE_ACCELERATION:
        new_acceleration = Settings.MAX_POSITIVE_ACCELERATION
    if new_acceleration < Settings.MAX_NEGATIVE_ACCELERATION:
        new_acceleration = Settings.MAX_NEGATIVE_ACCELERATION
    new_speed = current_speed + new_acceleration * Settings.TICK_LENGTH
    if new_speed > Settings.MAX_SPEED:
        new_speed = Settings.MAX_SPEED
    if new_speed < 0:
        new_speed = 0
    return new_speed


def set_ego_jerk(jerk):
    current_acceleration = traci.vehicle.getAcceleration("ego")
    current_speed = traci.vehicle.getSpeed("ego")
    new_speed = get_ego_speed_from_jerk(current_speed, current_acceleration, jerk)
    traci.vehicle.setSpeed("ego", new_speed)
    return new_speed


def set_ego_speed(speed):
    traci.vehicle.setSpeed("ego", speed)


def get_ego_position():
    return traci.vehicle.getPosition("ego")


def get_ego_acceleration():
    return traci.vehicle.getAcceleration("ego")


def get_ego_speed():
    return traci.vehicle.getSpeed("ego")


def get_ego_start_speed():
    if Settings.RANDOMIZE_START_SPEED:
        start_speed = np.random.normal(Settings.START_SPEED, Settings.START_SPEED_VARIANCE)
        start_speed = np.clip(start_speed, Settings.MIN_START_SPEED, Settings.MAX_START_SPEED)
    else:
        start_speed = Settings.START_SPEED
    return start_speed


def ego_just_arrived():
    return "ego" in traci.simulation.getArrivedIDList()


def just_had_collision():
    return traci.simulation.getCollidingVehiclesNumber() > 0 and 'ego' not in traci.vehicle.getIDList()


def step():
    global delay
    traci.simulationStep()
    if Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
        current_time = traci.simulation.getCurrentTime()
        if delay <= 0:
            traci.vehicle.add("traffic_{}".format(current_time), "highwayRoute", "normal", departSpeed=Settings.OTHER_CAR_SPEED)
            if Settings.VARY_TRAFFIC_START_TIMES:
                delay = random.random() + Settings.BASE_TRAFFIC_INTERVAL
            else:
                delay = Settings.BASE_TRAFFIC_INTERVAL
        delay -= Settings.TICK_LENGTH


def run_episode(control_function: Callable[[Any], Any], state_function: Callable[[], Any] = get_ego_position,
                max_episode_length=100, start_velocity=None, end_episode_callback: Optional[Callable[[Any], Any]] = None,
                wait_before_start=20, limit_metrics=False):
    """
    Run a lane merging episode, using the given function to control the ego vehicle

    :param control_function: A function accepting the current state (from state_function), performing some control
    action, and returning the control signal, for logging purposes
    :param state_function: A function that returns an object representing the current state, which is passed to control_function
    :param max_episode_length: The maximum length of the episode, in seconds
    :param start_velocity: The starting velocity of the ego car, set to None for the default from settings
    :param end_episode_callback: A function that is called at the end of an episode, given the last state as a parameter
    :param wait_before_start: The number of seconds to wait before the ego car is added to SUMO and the episode starts
    :param limit_metrics: Whether to prevent some more expensive metrics from being calculated
    :return: A dictionary with a number of metrics about the episode, after it is finished running
    """
    end_simulation = False
    episode_length = 0
    state_history = []
    control_history = []
    position_history = []
    speed_history = []
    acceleration_history = []
    jerk_history = []
    closest_vehicle_history = []
    after_decel_history = []
    max_episode_ticks = max_episode_length / Settings.TICK_LENGTH

    for i in range(int(wait_before_start / Settings.TICK_LENGTH)):
        step()

    if start_velocity is None:
        start_velocity = get_ego_start_speed()

    add_ego_car(start_velocity)
    step()
    crashed = False
    finished = True
    start_time = time.perf_counter()

    while not end_simulation:
        episode_length += 1
        if "ego" in traci.simulation.getArrivedIDList():
            end_simulation = True
        elif traci.simulation.getCollidingVehiclesNumber() > 0:
            remove_ego_car()
            step()
            end_simulation = True
            crashed = True
            finished = False
        else:
            current_state = state_function()
            state_history.append(current_state)
            ego_position = get_ego_position()
            position_history.append(ego_position)
            speed_history.append(get_ego_speed())
            acceleration_history.append(get_ego_acceleration())
            if len(acceleration_history) == 1:
                jerk_history.append(0)
            else:
                jerk_history.append((acceleration_history[-1] - acceleration_history[-2]) / Settings.TICK_LENGTH)
            if not limit_metrics:
                ego_s = get_ego_s(ego_position)
                if ego_s > Settings.MERGE_POINT_X:
                    rich_state = prediction.HighwayState.from_sumo()
                    car_front, car_behind = rich_state.get_closest_cars()
                    if car_front:
                        before_x = car_front[0]
                    else:
                        before_x = np.inf
                    if car_behind:
                        after_x = car_behind[0]
                        after_decel = -min(car_behind[2], 0)
                    else:
                        after_x = np.inf
                        after_decel = 0
                    min_distance = min(abs(before_x - ego_position[0]), abs(after_x - ego_position[0]), 100)
                    if ego_s > Settings.CRASH_MIN_S:
                        closest_vehicle_history.append(min_distance)
                    after_decel_history.append(after_decel)

            control_history.append(control_function(current_state))

        if episode_length > max_episode_ticks and not end_simulation:
            remove_ego_car()
            step()
            finished = False
            break
        else:
            step()

    if end_episode_callback is not None and len(state_history) >= 1:
        end_episode_callback(state_history[-1])
    end_time = time.perf_counter()

    episode_stats = {
        "crashed": crashed,
        "merged": finished,
        "state_history": state_history,
        "control_history": control_history,
        "position_history": position_history,
        "speed_history": speed_history,
        "acceleration_history": acceleration_history,
        "disruption_history": after_decel_history,
        "jerk_history": jerk_history,
        "closest_vehicle_history": closest_vehicle_history,
        "simulation_time_taken": len(state_history) * Settings.TICK_LENGTH,
        "end_time": end_time,
        "start_time": start_time
    }

    return episode_stats


def evaluate_control(control_function: Callable[[Any], Any], num_episodes=1000,
                     state_function: Callable[[], Any] = get_ego_position,
                     custom_stats_function: Optional[Callable[[Dict], Dict]] = None,
                     end_episode_callback: Optional[Callable[[Any], Any]] = None,
                     max_episode_length=100, start_velocity=None, wait_before_start=50, save_state_on_crash=False, verbose=False, crash_callback=None):

    aggregate_stats = stats.StatsAggregator(save_state_on_crash)
    if custom_stats_function is not None:
        aggregate_stats.add_custom_stat_callback(custom_stats_function)
    for i in range(num_episodes):
        if verbose:
            print(i)
        episode_stats = run_episode(control_function, state_function, max_episode_length, start_velocity,
                                    end_episode_callback, wait_before_start)
        aggregate_stats.add_episode_stats(episode_stats)
        if episode_stats["crashed"] and crash_callback is not None:
            crash_callback(episode_stats["state_history"])
        if verbose and episode_stats["crashed"]:
            print("crashed")

    return aggregate_stats


merge_point = (-50.9, 1.72)
merge_point2 = (1.5, -1.5)
merge_point3 = (-51, -1.5)
merge_distance = distance(merge_point, merge_point2)
common_s = merge_point2[0] - merge_point3[0]


def get_ego_s(ego_position):
    ego_x, ego_y = ego_position
    if ego_x < merge_point[0]:
        return - distance(ego_position, merge_point)
    elif ego_x < merge_point2[0]:
        return distance(ego_position, merge_point)
    else:
        return ego_x - merge_point2[0] + common_s


def get_obstacle_s(vehicle_position):
    vehicle_x = vehicle_position[0]
    return vehicle_x - merge_point3[0]


def get_obstacle_s_from_x(vehicle_x):
    return vehicle_x - merge_point3[0]
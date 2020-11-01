import heapq

from scipy import interpolate
from cvxopt import solvers, matrix
import matplotlib.pyplot as plt
import numpy as np

import control
from config import Settings
import prediction

if Settings.USE_CYTHON:
    import st_cy


solvers.options['show_progress'] = False
solvers.options['maxiters'] = 10


def get_range_index(min_s, delta_s, s):
    # Gets the last index before s in the list [min_s, min_s + delta_s, min_s + 2*delta_s, ...]
    return int((s - min_s) / delta_s)


def find_s_t_obstacles_from_state(current_state, future_s=150, delta_s=0.5, delta_t=0.2, time_limit=5, start_uncertainty=0.0, uncertainty_per_second=0.1):
    ego_position = current_state.ego_position
    ego_speed = current_state.ego_speed
    start_s = control.get_ego_s(ego_position)

    # We discretize the s and t space, and store the lookup for s and t values in these arrays
    s_values = np.arange(start_s, start_s + future_s + delta_s, delta_s)
    t_values = np.arange(0, time_limit + delta_t, delta_t)
    obstacles = np.zeros((t_values.size, s_values.size), dtype=np.bool)
    distances = np.zeros(obstacles.shape, dtype=np.float)
    distances += 1E10  # Big number but not NaN

    discrete_length = int(Settings.CAR_LENGTH / delta_s)
    predicted_state = current_state
    for (t_index, t) in enumerate(t_values):
        uncertainty = start_uncertainty + uncertainty_per_second * t
        discrete_uncertainty = int(uncertainty/delta_s)
        if t_index != 0:
            predicted_state, _ = predicted_state.predict_step_without_ego(delta_t)
        for vehicle_index, vehicle_x in enumerate(predicted_state.other_xs):
            current_obstacle_s = control.get_obstacle_s_from_x(vehicle_x)
            if current_obstacle_s < Settings.CRASH_MIN_S - Settings.MIN_ALLOWED_DISTANCE:
                break  # Cars do not obstruct path until the merge point
            elif current_obstacle_s > s_values[-1] + Settings.CAR_LENGTH:
                continue

            # calculate the distance from each point to this obstacle vehicle at time t
            obstacle_distances_front = np.abs(s_values - (current_obstacle_s - Settings.CAR_LENGTH - uncertainty))
            obstacle_distances_back = np.abs(s_values - (current_obstacle_s + Settings.CAR_LENGTH + uncertainty))
            # the distance is the minimum of the distance from the front of the ego vehicle to the back of the
            # obstacle vehicle, and from the front of the obstacle to the back of the ego vehicle
            distances[t_index] = np.minimum(obstacle_distances_front, distances[t_index, :])
            distances[t_index] = np.minimum(obstacle_distances_back, distances[t_index, :])

            # Within a vehicle length of the obstacle's s position, register the presence of an obstacle
            start_s_index = get_range_index(start_s, delta_s, current_obstacle_s)
            index_min = max(start_s_index - discrete_length - discrete_uncertainty, 0)
            index_max = min(start_s_index + discrete_length + discrete_uncertainty, s_values.size)
            if index_min < s_values.size and index_max > 0:
                obstacles[t_index, index_min:index_max] = True
                distances[t_index, index_min:index_max] = 0

    # plt.close()
    # plot_s_t_obstacles(obstacles, s_values, t_values)
    # plt.show()
    return obstacles, s_values, t_values, ego_speed, distances


def find_s_t_obstacles(future_s=150, delta_s=0.5, delta_t=0.2, time_limit=5, start_uncertainty=0.0, uncertainty_per_second=0.1):
    """
    For the current state of the system, predict and plot the positions of and distances to all obstacles in the future

    :param future_s: how far in the future to look in the s space
    :param delta_s: the discretization of the s space
    :param delta_t: the discretization of the t space
    :param time_limit: how far in the future to look in the t space
    :param start_uncertainty: makes other cars start out this much longer for collision detection
    :param uncertainty_per_second: makes cars this much longer for each second in the future
    """
    current_state = prediction.HighwayState.from_sumo()
    return find_s_t_obstacles_from_state(current_state)


def plot_s_t_obstacles(obstacles, s_values, t_values, color='blue'):
    nonzero_t, nonzero_s = np.nonzero(obstacles)
    ts = t_values[nonzero_t]
    ss = s_values[nonzero_s]
    plt.figure()
    plt.scatter(ts, ss, c=color)
    plt.ylim(s_values[0], s_values[-1])
    plt.xlim(t_values[0], t_values[-1])
    plt.xlabel('t')
    plt.ylabel('s')


def plot_s_path(obstacles, s_values, t_values, s_path):
    plot_s_t_obstacles(obstacles, s_values, t_values)
    plt.plot(t_values, s_path, c='red')


def get_feasible_next_s_range_no_jerk_limits(s, prev_s, delta_t):
    v = (s - prev_s) / delta_t
    min_v = max(v + Settings.MAX_NEGATIVE_ACCELERATION * delta_t, 0)
    max_v = min(v + Settings.MAX_POSITIVE_ACCELERATION * delta_t, Settings.MAX_SPEED)
    min_s = s + min_v * delta_t  # note: automatically greater than zero
    max_s = s + max_v * delta_t  # note: automatically capped wrt max speed
    return min_s, max_s


def get_feasible_next_s_range_with_jerk_limits(s, s_1, s_2, delta_t):
    prev_v = (s_1 - s_2) / delta_t
    v = (s - s_1) / delta_t
    a = (v - prev_v) / delta_t
    min_a = max(a + Settings.MINIMUM_NEGATIVE_JERK * delta_t, Settings.MAX_NEGATIVE_ACCELERATION)
    max_a = min(a + Settings.MAXIMUM_POSITIVE_JERK * delta_t, Settings.MAX_POSITIVE_ACCELERATION)
    min_v = max(v + min_a * delta_t, 0)
    max_v = min(v + max_a * delta_t, Settings.MAX_SPEED)
    min_s = s + min_v * delta_t  # note: automatically greater than zero
    max_s = s + max_v * delta_t  # note: automatically capped wrt max speed
    return min_s, max_s


def distance_penalty(min_distance):
    if min_distance < Settings.MIN_ALLOWED_DISTANCE:
        return 1000000.0 * Settings.D_WEIGHT / max(min_distance, 1.0)
    else:
        return Settings.D_WEIGHT / min_distance


def cost_no_jerk(s, s_1, s_2, t_discretization, min_distance):
    v = (s - s_1) / t_discretization
    a = (s - 2*s_1 + s_2) / (t_discretization**2)
    return Settings.V_WEIGHT * (v - Settings.DESIRED_SPEED)**2 + Settings.A_WEIGHT * a**2 + distance_penalty(min_distance)


def cost(s, s_1, s_2, s_3, t_discretization, min_distance):
    v = (s - s_1) / t_discretization
    a = (s - 2*s_1 + s_2) / (t_discretization**2)
    j = (s - 3*s_1 + 3*s_2 - s_3) / (t_discretization**3)
    return Settings.V_WEIGHT * (v - Settings.DESIRED_SPEED)**2 + Settings.A_WEIGHT * a**2 + Settings.J_WEIGHT * j**2 + distance_penalty(min_distance)


def get_all_range_indices(start_s, delta_s, range_min, range_max):
    """
    Gets the indices in [start_s, start_s + delta_s, start_s + 2*delta_s, ...] between range_min and range_max (inclusive)

    :param start_s: The start of the list
    :param delta_s: The discretization of the list
    :param range_min: The minimum bound of the desired range
    :param range_max: The maximum bound of the desired range
    :return: A list of the desired indices
    """
    min_index = (range_min - start_s) / delta_s
    if int(min_index) < min_index:
        min_index = int(min_index) + 1
    else:
        min_index = int(min_index)
    max_index = int((range_max - start_s) / delta_s)
    return list(range(min_index, max_index + 1))


def readable_solve_s_t_path_no_jerk(obstacles, s_values, t_values, ego_start_speed, distances):
    """
    Finds the optimal path through the discretized s-t space

    Adheres to velocity, acceleration, and monotonicity contraints (but not jerk constraints). If we assume that the
    discretization of the s and t space have maximum horizons of s_max and t_max respectively, with quantization sizes
    delta_s and delta_t, then we are given information about the discretized space as follows (where num_s = s_max/delta_s
    and num_t = t_max / delta_t)

    For this version we will use Djikstra's algorithm. For later improvements, consider making a better heuristic for A*

    :param obstacles: a boolean ndarray of size (num_t x num_s), encoding the projected positions of obstacles
    :param s_values: an ndarray of size num_s, encoding the actual s values
    :param t_values: an ndarray of size num_t, encoding the actual t values
    :param ego_start_speed: the starting speed of the ego car
    :param distances: an ndarray of size (num_t x num_s), encoding the distances to the nearest obstacle
    :return:
    """

    delta_s = s_values[1] - s_values[0]
    delta_t = t_values[1] - t_values[0]
    num_s = s_values.size
    num_t = t_values.size
    start_s = s_values[0]

    # For this version we will work forwards instead of backwards for the DP, as it is more readable that way

    best_previous_s = np.zeros((num_t, num_s, num_s), dtype=np.int32)
    encountered = np.zeros((num_t, num_s, num_s), dtype=bool)

    estimated_previous_s = start_s - delta_t * ego_start_speed

    entry_order = 0

    min_first_s, max_first_s = get_feasible_next_s_range_no_jerk_limits(start_s, estimated_previous_s, delta_t)

    # We transform the raw s value range to a list of possible s indices

    possible_first_s_indices = get_all_range_indices(start_s, delta_s, min_first_s, max_first_s)

    node_priority_queue = []
    for s_index in possible_first_s_indices:
        s_value = s_values[s_index]
        if not obstacles[1, s_index]:
            s_cost = cost_no_jerk(s_value, start_s, estimated_previous_s, delta_t, distances[1, s_index])
            node_priority_queue.append((s_cost, entry_order, 1, s_index, 0, 0))
            entry_order -= 1  # We want the queue to be LIFO, as this tends to be faster for shortest path problems

    heapq.heapify(node_priority_queue)

    best_last_s_tuple = (-1, -1)
    best_t_index = 0

    while len(node_priority_queue) > 0:
        # We get the (t, s, prev_s) tuple with the lowest cost so far
        total_cost, _, t_index, s_index, prev_s_index, second_s_index = heapq.heappop(node_priority_queue)
        s_value = s_values[s_index]
        prev_s_value = s_values[prev_s_index]

        if encountered[t_index, s_index, prev_s_index]:
            continue
        else:
            encountered[t_index, s_index, prev_s_index] = True
            best_previous_s[t_index, s_index, prev_s_index] = second_s_index

        # We keep track of the furthest point in time we can safely reach in case we cannot reach the end
        if t_index > best_t_index:
            best_t_index = t_index
            best_last_s_tuple = (s_index, prev_s_index)
        if t_index == num_t - 1:
            break

        # Again, calculate the possible next values of s
        min_next_s, max_next_s = get_feasible_next_s_range_no_jerk_limits(s_value, prev_s_value, delta_t)
        possible_next_s_indices = get_all_range_indices(start_s, delta_s, min_next_s, max_next_s)
        next_t = t_index + 1

        for next_s_index in possible_next_s_indices:

            # We can't exceed the bounds of our simulation, but if this is happening it may be a good idea to increase the planning horizon
            if next_s_index >= num_s:
                break

            # If we have not yet encountered the next (s, prev_s) tuple at the specified time, we have found the optimal path to reach it
            if not encountered[next_t, next_s_index, s_index]:

                if obstacles[next_t, next_s_index]:
                    continue  # No colliding with obstacles
                next_s_value = s_values[next_s_index]
                s_cost = cost_no_jerk(next_s_value, s_value, prev_s_value, delta_t, distances[next_t, next_s_index])
                heapq.heappush(node_priority_queue, (total_cost + s_cost, entry_order, next_t, next_s_index, s_index, prev_s_index))
                entry_order -= 1

    # Reconstruct the best sequence of s values, using the saved values from best_previous_s
    best_s_index, best_prev_s_index = best_last_s_tuple
    s_sequence = np.zeros(num_t)
    for t_index in range(best_t_index, 1, -1):
        s_sequence[t_index] = s_values[best_s_index]
        second_s_index = best_previous_s[t_index, best_s_index, best_prev_s_index]
        best_s_index = best_prev_s_index
        best_prev_s_index = second_s_index

    s_sequence[0] = s_values[best_prev_s_index]
    s_sequence[1] = s_values[best_s_index]

    return s_sequence


def get_path_mean_abs_jerk(s_sequence, ego_start_speed, ego_start_acceleration, delta_t):
    prev_a = ego_start_acceleration
    prev_v = ego_start_speed
    path_cost = 0
    for i, s in enumerate(s_sequence):
        if i == 0:
            continue
        s_1 = s_sequence[i - 1]
        v = (s - s_1)/delta_t
        a = (v - prev_v)/delta_t
        j = (a - prev_a)/delta_t
        prev_v = v
        prev_a = a
        path_cost += abs(j)
    return path_cost / (len(s_sequence) - 1)


def get_path_cost(s_sequence, ego_start_speed, ego_start_acceleration, delta_t, s_values, distances):
    """
    Get the cost of a path produced by an s-t path planning algorithm

    :param s_sequence: The path as an ndarray of s coordinates
    :param ego_start_speed: The starting speed of the ego car
    :param delta_t: The time between points on the path
    :param s_values: An ndarray of possible s coordinates, as given as input to the s-t solver
    :param distances: An ndarray of distances to the nearest obstacle, as given as input to the s-t solver
    :return: The total cost of the given path
    """
    path_cost = 0
    prev_a = ego_start_acceleration
    prev_v = ego_start_speed
    for i in range(1, len(s_sequence)):
        s = s_sequence[i]
        s_1 = s_sequence[i - 1]
        if i == 1:
            s_2 = s_1 - ego_start_speed * delta_t
            s_3 = s_2 - (ego_start_speed - ego_start_acceleration * delta_t) * delta_t
        elif i == 2:
            s_2 = s_sequence[i - 2]
            s_3 = s_1 - ego_start_speed * delta_t
        else:
            s_2 = s_sequence[i - 2]
            s_3 = s_sequence[i - 3]
        matches = np.where(s_values == s)[0]
        v = (s - s_1)/delta_t
        a = (v - prev_v)/delta_t
        j = (a - prev_a)/delta_t
        if v > Settings.MAX_SPEED:
            print("Exceeded speed limit")
        if a > Settings.MAX_POSITIVE_ACCELERATION or a < Settings.MAX_NEGATIVE_ACCELERATION:
            print("Exceeded acceleration limit")
        if j > Settings.MAXIMUM_POSITIVE_JERK or j < Settings.MINIMUM_NEGATIVE_JERK:
            print("Exceeded jerk limit")
        prev_v = v
        prev_a = a
        if len(matches) > 0:
            s_index = np.where(s_values == s)[0][0]
            path_cost += cost(s, s_1, s_2, s_3, delta_t, distances[i, s_index])
        else:
            # No valid path
            path_cost = np.infty
            break
    return path_cost


def valid_descendant_s_indices_no_jerk(t_index, start_s, s, s_1, delta_s, delta_t, obstacles):
    min_s, max_s = get_feasible_next_s_range_no_jerk_limits(s, s_1, delta_t)
    descendant_s_indices = get_all_range_indices(start_s, delta_s, min_s, max_s)
    descendants = []
    for s_index in descendant_s_indices:
        if not obstacles[t_index, s_index]:
            descendants.append(s_index)
    return descendants


def valid_descendant_s_indices_with_jerk(t_index, start_s, s, s_1, s_2, delta_s, delta_t, obstacles):
    min_s, max_s = get_feasible_next_s_range_with_jerk_limits(s, s_1, s_2, delta_t)
    descendant_s_indices = get_all_range_indices(start_s, delta_s, min_s, max_s)
    descendants = []
    for s_index in descendant_s_indices:
        if s_index >= obstacles.shape[1]:
            break
        if not obstacles[t_index + 1, s_index]:
            descendants.append(s_index)
    return descendants


def solve_st_fast_v2(obstacles, s_values, t_values, ego_start_speed, ego_start_acceleration, distances):
    """
    A much faster st solver that still attempts not to crash, but produces suboptimal solutions

    :param obstacles: a boolean ndarray of size (num_t x num_s), encoding the projected positions of obstacles
    :param s_values: an ndarray of size num_s, encoding the actual s values
    :param t_values: an ndarray of size num_t, encoding the actual t values
    :param ego_start_speed: the starting speed of the ego car
    :param ego_start_acceleration: the starting acceleration of the ego car
    :param distances: an ndarray of size (num_t x num_s), encoding the distances to the nearest obstacle
    :return: an ndarray of size num_t, giving the planned trajectory in the s space
    """

    # Extracting some relevant quantities from the input arrays
    delta_s = s_values[1] - s_values[0]
    delta_t = t_values[1] - t_values[0]
    num_s = s_values.size
    num_t = t_values.size
    start_s = s_values[0]
    estimated_previous_s = start_s - delta_t * ego_start_speed
    estimated_second_s = estimated_previous_s - delta_t * (ego_start_speed - ego_start_acceleration * delta_t)

    # Initialize arrays to backtrack after the search is done and avoid visiting a node twice
    best_previous_s = np.zeros((num_t, num_s), dtype=np.int32)
    encountered = np.zeros((num_t, num_s), dtype=bool)

    # We need a priority queue for a more efficient implementation of Dijkstra's algorithm
    node_priority_queue = []
    heapq.heapify(node_priority_queue)

    # The priority queue is sorted by tuples of the form:
    # Total cost (0), t index (1), s index (2), s value (3), previous s index (4), previous s value (5), index for s_{t-2} (6), value for s_{t-2} (7)
    first_heap_item = (0, 0, 0, start_s, 0, estimated_previous_s, 0, estimated_second_s)

    heapq.heappush(node_priority_queue, first_heap_item)

    # Our nodes in our graph are in the form t_index, s_index. This keeps track of the best node we have reached so far
    best_node = (0, 0)

    while len(node_priority_queue) > 0:
        # We get the (t, s, prev_s) tuple with the lowest cost so far
        total_cost, t_index, s_index, s_value, prev_s_index, prev_s_value, second_s_index, second_s_value = heapq.heappop(node_priority_queue)
        node = t_index, s_index

        # We may add the same node to the priority queue multiple times (cost depending on the path taken to get there)
        # However, only the first, and therefore lowest cost, instance has its neighbors expanded.
        if encountered[node]:
            continue
        else:
            encountered[node] = True
            best_previous_s[node] = prev_s_index

        # We keep track of the furthest ("best") point in time we can safely reach in case we cannot reach the end
        if t_index > best_node[0]:
            best_node = (t_index, s_index)
        if t_index == num_t - 1:
            break  # We have found the best path to the end of our planning period

        # Calculate the possible next values of s given the velocity and acceleration constraints
        possible_next_s_indices = valid_descendant_s_indices_with_jerk(t_index, start_s, s_value, prev_s_value, second_s_value, delta_s, delta_t, obstacles)

        next_t = t_index + 1

        for next_s_index in possible_next_s_indices:
            # We can't exceed the bounds of our simulation, but if this is happening it may be a good idea
            # to increase the planning horizon in the s dimension
            if next_s_index >= num_s:
                break

            next_node = (next_t, next_s_index)

            if not encountered[next_node]:
                if obstacles[next_node]:
                    continue  # No colliding with obstacles

                next_s_value = s_values[next_s_index]
                s_cost = cost(next_s_value, s_value, prev_s_value, second_s_value, delta_t, distances[next_node])

                # Total cost (0), t index (1), s index (2), s value (3), previous s index (4),
                # previous s value (5), index for s_{t-2} (6), value for s_{t-2} (7)
                heapq.heappush(node_priority_queue, (total_cost + s_cost, next_t, next_s_index, next_s_value, s_index, s_value, prev_s_index, prev_s_value))

    # Reconstruct the best sequence of s values, using the saved values from best_previous_s
    best_t_index, best_s_index = best_node
    s_sequence = np.zeros(num_t)
    for t_index in range(best_t_index, 0, -1):
        s_sequence[t_index] = s_values[best_s_index]
        node = t_index, best_s_index
        best_s_index = best_previous_s[node]

    s_sequence[0] = s_values[best_s_index]
    return s_sequence


def solve_st_fast(obstacles, s_values, t_values, ego_start_speed, distances):
    """
    A much faster st solver that still attempts not to crash, but produces suboptimal solutions

    :param obstacles: a boolean ndarray of size (num_t x num_s), encoding the projected positions of obstacles
    :param s_values: an ndarray of size num_s, encoding the actual s values
    :param t_values: an ndarray of size num_t, encoding the actual t values
    :param ego_start_speed: the starting speed of the ego car
    :param distances: an ndarray of size (num_t x num_s), encoding the distances to the nearest obstacle
    :return: an ndarray of size num_t, giving the planned trajectory in the s space
    """

    delta_s = s_values[1] - s_values[0]
    delta_t = t_values[1] - t_values[0]
    num_s = s_values.size
    num_t = t_values.size
    start_s = s_values[0]

    # For this version we will work forwards instead of backwards for the DP, as it is more readable that way

    best_previous_s = np.zeros((num_t, num_s), dtype=np.int32)
    encountered = np.zeros((num_t, num_s), dtype=bool)

    estimated_previous_s = start_s - delta_t * ego_start_speed

    entry_order = 0

    min_first_s, max_first_s = get_feasible_next_s_range_no_jerk_limits(start_s, estimated_previous_s, delta_t)

    # We transform the raw s value range to a list of possible s indices

    possible_first_s_values = get_all_range_indices(start_s, delta_s, min_first_s, max_first_s)

    node_priority_queue = []
    for s_index in possible_first_s_values:
        s_value = s_values[s_index]
        if not obstacles[1, s_index]:
            s_cost = cost_no_jerk(s_value, start_s, estimated_previous_s, delta_t, distances[1, s_index])
            node_priority_queue.append((s_cost, entry_order, 1, s_index, 0))
            entry_order -= 1  # We want the queue to be LIFO, as this tends to be faster for shortest path problems

    heapq.heapify(node_priority_queue)

    best_last_s = -1
    best_t_index = 0

    while len(node_priority_queue) > 0:
        # We get the (t, s, prev_s) tuple with the lowest cost so far
        total_cost, entry_order, t_index, s_index, prev_s_index = heapq.heappop(node_priority_queue)
        s_value = s_values[s_index]
        prev_s_value = s_values[prev_s_index]

        if encountered[t_index, s_index]:
            continue
        else:
            encountered[t_index, s_index] = True
            best_previous_s[t_index, s_index] = prev_s_index

        # We keep track of the furthest point in time we can safely reach in case we cannot reach the end
        if t_index > best_t_index:
            best_t_index = t_index
            best_last_s = s_index
        if t_index == num_t - 1:
            break

        # Again, calculate the possible next values of s
        min_next_s, max_next_s = get_feasible_next_s_range_no_jerk_limits(s_value, prev_s_value, delta_t)
        possible_next_s_values = get_all_range_indices(start_s, delta_s, min_next_s, max_next_s)
        next_t = t_index + 1

        for next_s_index in possible_next_s_values:
            # We can't exceed the bounds of our simulation, but if this is happening it may be a good idea to increase the planning horizon
            if next_s_index >= num_s:
                break

            if not encountered[next_t, next_s_index]:

                if obstacles[next_t, next_s_index]:
                    continue  # No colliding with obstacles

                next_s_value = s_values[next_s_index]
                s_cost = cost_no_jerk(next_s_value, s_value, prev_s_value, delta_t, distances[next_t, next_s_index])
                heapq.heappush(node_priority_queue, (total_cost + s_cost, entry_order, next_t, next_s_index, s_index))
                entry_order -= 1

    # Reconstruct the best sequence of s values, using the saved values from best_previous_s
    best_s_index = best_last_s
    s_sequence = np.zeros(num_t)
    for t_index in range(best_t_index, 0, -1):
        s_sequence[t_index] = s_values[best_s_index]
        best_s_index = best_previous_s[t_index, best_s_index]

    s_sequence[0] = s_values[best_s_index]
    return s_sequence


def get_before_after_constraints(s_sequence, t_values):
    last_ego_position = s_sequence[-1]
    last_future_time = t_values[-1]
    before_car_start_pos = np.inf
    before_car_speed = 0
    after_car_start_pos = np.inf
    after_car_speed = 0
    before_car_end_pos = -np.inf
    after_car_end_pos = np.inf
    vehicle_ids = control.get_vehicle_ids()
    positions = control.get_vehicle_positions(vehicle_ids)
    speeds = control.get_vehicle_speeds(vehicle_ids)
    ego_position = positions["ego"]

    for vehicle_id in vehicle_ids:
        if vehicle_id != "ego":
            obstacle_s = control.get_obstacle_s(positions[vehicle_id])
            obstacle_speed = speeds[vehicle_id]
            end_obstacle_s = obstacle_s + obstacle_speed * last_future_time
            if end_obstacle_s < -Settings.CAR_LENGTH:
                continue
            if end_obstacle_s > last_ego_position and end_obstacle_s < after_car_end_pos:
                after_car_end_pos = end_obstacle_s
                after_car_start_pos = obstacle_s
                after_car_speed = obstacle_speed
            elif end_obstacle_s < last_ego_position and end_obstacle_s > before_car_end_pos:
                before_car_end_pos = end_obstacle_s
                before_car_start_pos = obstacle_s
                before_car_speed = obstacle_speed

    return before_car_start_pos, before_car_speed, after_car_start_pos, after_car_speed


def finer_fit(s_sequence, delta_t, coarse_delta_t, start_speed, start_acceleration, before_after_cars=None):

    s_length = len(s_sequence)
    if s_length == 1:
        return s_sequence

    t = np.arange(s_length) * coarse_delta_t
    sub_length = int(np.round(t[-1] / delta_t + 1))
    if (sub_length - 1)*delta_t > t[-1]:
        sub_length -= 1
    finer_t = np.arange(sub_length) * delta_t

    # Calculate a linear interpolation of our original sequence
    interpolation = interpolate.interp1d(t, s_sequence)
    interpolated = interpolation(finer_t)

    # QP objective ||Ax - b||^2
    A = np.identity(sub_length)
    b = interpolated

    # In the form (1/2)x^TPx + q^Tx
    P = 2 * np.dot(A.T, A)
    q = -2 * np.dot(A.T, b)

    # Velocity min constraints: velocity \geq 0 in the form V_1 x \leq h
    V_1 = np.zeros((sub_length - 1, sub_length))
    h_1 = np.zeros(sub_length - 1)
    for i in range(sub_length - 1):
        V_1[i, i] = 1 / delta_t
        V_1[i, i+1] = -1 / delta_t
        # (s_{i+1} - s_i)/delta_t \geq 0

    # Velocity max constraints: velocity \leq v_max
    V_2 = -V_1
    h_2 = np.zeros(sub_length - 1)
    for i in range(sub_length - 1):
        h_2[i] = Settings.MAX_SPEED

    # Acceleration max constraints: (s_t - 2*s_{t-1} + s_{t-2})/(delta_t ** 2) \leq a_max
    delta_t_2 = delta_t ** 2
    A_3 = np.zeros((sub_length - 1, sub_length))
    h_3 = np.zeros(sub_length - 1)
    A_3[0, 0] = -1 / delta_t_2
    A_3[0, 1] = 1 / delta_t_2
    h_3[0] = Settings.MAX_POSITIVE_ACCELERATION + start_speed / delta_t
    for i in range(1, sub_length - 1):
        A_3[i, i-1] = 1 / delta_t_2
        A_3[i, i] = -2 / delta_t_2
        A_3[i, i+1] = 1 / delta_t_2
        h_3[i] = Settings.MAX_POSITIVE_ACCELERATION

    # Acceleration min constraints: (s_t - 2*s_{t-1} + s_{t-2})/(delta_t ** 2) \geq a_min
    h_4 = np.zeros(sub_length - 1)
    A_4 = -A_3
    h_4[0] = -Settings.MAX_NEGATIVE_ACCELERATION - start_speed / delta_t
    for i in range(1, sub_length - 1):
        h_4[i] = -Settings.MAX_NEGATIVE_ACCELERATION

    # Jerk max constraints: (s_t - 3*s_{t-1} + 3*s_{t-2} - s_{t-3})/(delta_t**3) \leq j_max
    delta_t_3 = delta_t ** 3
    J_5 = np.zeros((sub_length - 1, sub_length))
    h_5 = np.zeros(sub_length - 1)
    J_5[0, 0] = -1 / delta_t_3
    J_5[0, 1] = 1 / delta_t_3
    h_5[0] = Settings.MAXIMUM_POSITIVE_JERK + start_acceleration / delta_t + start_speed / delta_t_2
    if sub_length > 2:
        J_5[1, 0] = 2 / delta_t_3
        J_5[1, 1] = -3 / delta_t_3
        J_5[1, 2] = 1 / delta_t_3
        h_5[1] = Settings.MAXIMUM_POSITIVE_JERK - start_speed / delta_t_2
    for i in range(2, sub_length - 1):
        J_5[i, i-2] = -1 / delta_t_3
        J_5[i, i-1] = 3 / delta_t_3
        J_5[i, i] = -3 / delta_t_3
        J_5[i, i+1] = 1 / delta_t_3
        h_5[i] = Settings.MAXIMUM_POSITIVE_JERK

    # Jerk min constraints: (s_t - 3*s_{t-1} + 3*s_{t-2} - s_{t-3})/(delta_t**3) \geq j_min
    J_6 = -J_5
    h_6 = np.zeros(sub_length - 1)
    h_6[0] = -Settings.MINIMUM_NEGATIVE_JERK - start_acceleration / delta_t - start_speed / delta_t_2
    if sub_length > 2:
        h_6[1] = -Settings.MINIMUM_NEGATIVE_JERK + start_speed / delta_t_2
    for i in range(2, sub_length - 1):
        h_6[i] = -Settings.MINIMUM_NEGATIVE_JERK

    C_7 = None
    h_7 = None
    if before_after_cars is not None:
        before_s, before_speed, after_s, after_speed = before_after_cars
        before_ts = []
        before_ss = []
        if not np.isinf(before_s):
            for i, t in enumerate(finer_t):
                before_s_projected = before_s + t * before_speed
                if before_s_projected < -Settings.CAR_LENGTH:
                    continue
                else:
                    before_ts.append(i)
                    before_ss.append(before_s_projected)
        after_ts = []
        after_ss = []
        if not np.isinf(after_s):
            for i, t in enumerate(finer_t):
                after_s_projected = after_s + t * after_speed
                if after_s_projected < -Settings.CAR_LENGTH:
                    continue
                else:
                    after_ts.append(i)
                    after_ss.append(after_s_projected)

        C_7 = np.zeros((len(before_ts) + len(after_ts), sub_length))
        h_7 = np.zeros(len(before_ts) + len(after_ts))
        index = 0
        for i, t_index in enumerate(before_ts):
            C_7[index, t_index] = -1
            h_7[index] = -before_ss[i] - Settings.CAR_LENGTH
            index += 1
        for i, t_index in enumerate(after_ts):
            C_7[index, t_index] = 1
            h_7[index] = after_ss[i] - Settings.CAR_LENGTH
            index += 1

    # Equality constraints, start at the correct point please
    A = np.zeros((1, sub_length))
    A[0, 0] = 1
    b = np.zeros(1)
    b[0] = s_sequence[0]

    # Put together in the form Gx \leq h
    if C_7 is not None:
        G = np.vstack((V_1, V_2, A_3, A_4, J_5, J_6, C_7))
        h = np.concatenate((h_1, h_2, h_3, h_4, h_5, h_6, h_7))
    else:
        G = np.vstack((V_1, V_2, A_3, A_4, J_5, J_6))
        h = np.concatenate((h_1, h_2, h_3, h_4, h_5, h_6))

    # Solve the QP
    sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    return np.array(sol['x']).flatten()


def get_appropriate_base_st_path_and_obstacles(state):
    obstacles, s_values, t_values, ego_speed, distances = find_s_t_obstacles_from_state(
        state,
        Settings.FUTURE_S,
        Settings.S_DISCRETIZATION,
        Settings.T_DISCRETIZATION,
        Settings.FUTURE_T,
        Settings.START_UNCERTAINTY,
        Settings.UNCERTAINTY_PER_SECOND)
    ego_acceleration = state.ego_acceleration

    # Do the ST path planning
    if Settings.USE_FAST_ST_SOLVER:
        if Settings.USE_CYTHON:
            s_sequence = st_cy.solve_s_t_path_fast(obstacles, s_values, t_values, ego_speed, ego_acceleration,
                                                   distances, Settings.D_WEIGHT, Settings.V_WEIGHT,
                                                   Settings.A_WEIGHT, Settings.J_WEIGHT, Settings.DESIRED_SPEED,
                                                   Settings.MAX_SPEED, Settings.MAX_NEGATIVE_ACCELERATION,
                                                   Settings.MAX_POSITIVE_ACCELERATION,
                                                   Settings.MINIMUM_NEGATIVE_JERK,
                                                   Settings.MAXIMUM_POSITIVE_JERK, Settings.MIN_ALLOWED_DISTANCE)
        else:
            s_sequence = solve_st_fast_v2(obstacles, s_values, t_values, ego_speed, ego_acceleration, distances)
    else:
        if Settings.USE_CYTHON:
            s_sequence = st_cy.solve_s_t_path_no_jerk_djikstra(obstacles, s_values, t_values, ego_speed, distances)
        else:
            s_sequence = readable_solve_s_t_path_no_jerk(obstacles, s_values, t_values, ego_speed, distances)
    return s_sequence, obstacles, s_values, t_values, distances


def do_st_control(state):
    ego_acceleration = state.ego_acceleration
    ego_speed = state.ego_speed
    s_sequence, obstacles, s_values, t_values, distances = get_appropriate_base_st_path_and_obstacles(state)

    # Trim the zeros from the end of the planned sequence (in the case where pathfinding failed)
    end_point = len(s_sequence)
    while s_sequence[end_point - 1] == 0:
        end_point -= 1
    if end_point != len(s_sequence):
        print("ST Solver finds crash inevitable")
    s_sequence = s_sequence[:end_point]

    # If the planning was done at a t discretization different from the tick length, smooth with a QP
    if Settings.TICK_LENGTH < Settings.T_DISCRETIZATION:
        s_sequence = finer_fit(s_sequence, Settings.TICK_LENGTH, Settings.T_DISCRETIZATION, ego_speed, ego_acceleration)

    # If the st solver predicts an immediate crash, nothing we can do
    if len(s_sequence) <= 1:
        control.set_ego_speed(ego_speed)
        return ego_speed

    # Plan using Euler updates
    planned_distance_first_step = s_sequence[1] - s_sequence[0]
    end_speed_first_step = planned_distance_first_step / (Settings.TICK_LENGTH)
    control.set_ego_speed(end_speed_first_step)
    return end_speed_first_step


def get_s_state():
    return control.get_ego_s(control.get_ego_position())


def test_guaranteed_crash_from_state(state):
    s_sequence, obstacles, s_values, t_values, distances = get_appropriate_base_st_path_and_obstacles(state)
    end_point = len(s_sequence)
    while s_sequence[end_point - 1] == 0:
        end_point -= 1
    if end_point != len(s_sequence):
        return True
    for i, s in enumerate(s_sequence):
        s_index = get_range_index(s_values[0], s_values[1] - s_values[0], s)
        distance = distances[i, s_index]
        if distance < Settings.COMBINATION_MIN_DISTANCE - Settings.CAR_LENGTH:
            return True
    return False


def do_conditional_st_based_on_first_step(state, start_speed):
    next_state, crashed = state.predict_step_with_ego(start_speed, delta_t=Settings.TICK_LENGTH)
    crash_guaranteed = test_guaranteed_crash_from_state(next_state)
    if crashed or crash_guaranteed:
        print("ST solver taking over")
        # Then the ST solver can't find a valid path after the predicted first step
        return do_st_control(state)
    else:
        control.set_ego_speed(start_speed)
        return start_speed


def evaluate_st(num_episodes=1000):
    aggregate_stats = control.evaluate_control(do_st_control, num_episodes=num_episodes, state_function=prediction.HighwayState.from_sumo, verbose=True)
    aggregate_stats.print_stats()


def evaluate_st_and_dump_crash(num_episodes=1000):
    aggregate_stats = control.evaluate_control(do_st_control, num_episodes, state_function=prediction.HighwayState.from_sumo, crash_callback=plot_crash, verbose=True, save_state_on_crash=True)
    aggregate_stats.print_stats()


def replay_crash():
    import pickle
    saved_data = pickle.load(open("crashed_state_history.pkl", 'rb'))
    for i, item in enumerate(saved_data):
        obstacles, s_values, t_values, ego_speed, ego_acceleration, distances = item
        s_sequence = st_cy.solve_s_t_path_fast(obstacles, s_values, t_values, ego_speed, ego_acceleration,
                                               distances, Settings.D_WEIGHT, Settings.V_WEIGHT,
                                               Settings.A_WEIGHT, Settings.J_WEIGHT, Settings.DESIRED_SPEED,
                                               Settings.MAX_SPEED, Settings.MAX_NEGATIVE_ACCELERATION,
                                               Settings.MAX_POSITIVE_ACCELERATION,
                                               Settings.MINIMUM_NEGATIVE_JERK,
                                               Settings.MAXIMUM_POSITIVE_JERK, Settings.MIN_ALLOWED_DISTANCE)
        end_point = len(s_sequence)
        while s_sequence[end_point-1] == 0:
            end_point -= 1
        s_sequence2 = finer_fit(s_sequence[:end_point], Settings.TICK_LENGTH, Settings.T_DISCRETIZATION, ego_speed, ego_acceleration)
        print(s_sequence2)
        plot_s_path(obstacles, s_values, t_values, s_sequence)
        plt.plot(np.linspace(t_values[0], Settings.TICK_LENGTH*(len(s_sequence2) - 1), len(s_sequence2)), s_sequence2, c='green')
        plt.savefig("plots/{}.png".format(i))
        plt.close()


def plot_crash(states):
    import os
    plotdir = os.path.join(Settings.FULL_LOG_DIR, "plots")
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    for j, start_state in enumerate(states):
        s_sequence, obstacles, s_values, t_values, distances = get_appropriate_base_st_path_and_obstacles(start_state)
        plot_s_path(obstacles, s_values, t_values, s_sequence)
        plt.savefig(os.path.join(plotdir, "st_{}".format(j)))
        plt.close()

import numpy as np
import heapq
cimport numpy as np

cdef double MAX_FLOAT = np.finfo(np.float64).max

cdef struct Range:
    double min
    double max

cdef struct IntRange:
    int min
    int max

cdef struct Point:
    double x
    double y

ctypedef np.uint8_t uint8

cdef double MAX_SPEED = 40
cdef double MAX_POSITIVE_ACCELERATION = 4.5
cdef double MAX_NEGATIVE_ACCELERATION = -6.0

cdef double DESIRED_SPEED = 30
cdef double V_WEIGHT = 0.5
cdef double A_WEIGHT = 1.0
cdef double D_WEIGHT = 1000.0

cdef double CAR_LENGTH = 5.0
cdef double MIN_ALLOWED_DISTANCE = 7.5


cdef double distance_penalty(double min_distance, double min_allowed_distance):
    if min_distance < min_allowed_distance:
        return 1000000.0 / max(min_distance, 1.0)
    else:
        return 1 / min_distance


cdef double cost(double s, double s_1, double s_2, double t_discretization, double min_distance):
    cdef double v = (s - s_1) / t_discretization
    cdef double a = (s - 2*s_1 + s_2) / (t_discretization**2)
    return V_WEIGHT * (v - DESIRED_SPEED)**2 + A_WEIGHT * a**2 + D_WEIGHT * distance_penalty(min_distance, MIN_ALLOWED_DISTANCE)

cdef double cost_with_jerk(double s, double s_1, double s_2, double s_3, double delta_t, double min_distance, double min_allowed_distance, double v_weight, double desired_speed, double a_weight, double j_weight, double d_weight):
    cdef double v = (s - s_1) / delta_t
    cdef double a = (s - 2*s_1 + s_2) / (delta_t**2)
    cdef double j = (s - 3*s_1 + 3*s_2 - s_3) / (delta_t**3)
    return v_weight * (v - desired_speed)**2 + a_weight * a**2 + j_weight * j**2 + d_weight * distance_penalty(min_distance, min_allowed_distance)

cdef int get_range_index(double min_s, double s_discretization, double s):
    return int((s - min_s) / s_discretization)


cdef Range get_feasible_next_s_range_no_jerk_limits(double s, double prev_s, double delta_t):
    cdef double v = (s - prev_s) / delta_t
    cdef double min_v = max(v + MAX_NEGATIVE_ACCELERATION * delta_t, 0)
    cdef double max_v = min(v + MAX_POSITIVE_ACCELERATION * delta_t, MAX_SPEED)
    cdef double min_s = s + min_v * delta_t  # Note: automatically greater than zero
    cdef double max_s = s + max_v * delta_t  # Note: automatically capped wrt max speed
    return Range(min_s, max_s)


cdef Range get_feasible_next_s_range_with_jerk_limits(double s, double s_1, double s_2, double delta_t, double negative_jerk_limit, double positive_jerk_limit, double negative_acceleration_limit, double positive_acceleration_limit, double max_speed):
    cdef double prev_v = (s_1 - s_2) / delta_t
    cdef double v = (s - s_1) / delta_t
    cdef double a = (v - prev_v) / delta_t
    cdef double min_a = max(a + negative_jerk_limit * delta_t, negative_acceleration_limit)
    cdef double max_a = min(a + positive_jerk_limit * delta_t, positive_acceleration_limit)
    cdef double min_v = max(v + min_a * delta_t, 0)
    cdef double max_v = min(v + max_a * delta_t, max_speed)
    cdef double min_s = s + min_v * delta_t  # note: automatically greater than zero
    cdef double max_s = s + max_v * delta_t  # note: automatically capped wrt max speed
    return Range(min_s, max_s)


cdef IntRange get_all_range_indices(double start_s, double delta_s, double range_min, double range_max):
    """
    Gets the indices in [start_s, start_s + delta_s, start_s + 2*delta_s, ...] between range_min and range_max (inclusive)

    :param start_s: The start of the list
    :param delta_s: The discretization of the list
    :param range_min: The minimum bound of the desired range
    :param range_max: The maximum bound of the desired range
    :return: A list of the desired indices
    """
    cdef double min_index_exact = (range_min - start_s) / delta_s
    cdef int min_index = int(min_index_exact)
    cdef int max_index = int((range_max - start_s) / delta_s)
    if min_index < min_index_exact:
        min_index += 1
    return IntRange(min_index, max_index + 1)


def solve_s_t_path_no_jerk_djikstra(obstacles_bool, double[:] s_indices, double[:] t_indices, double ego_start_speed, double[:,:] distances):
    cdef Py_ssize_t num_s = s_indices.shape[0]
    cdef Py_ssize_t num_t = t_indices.shape[0]
    cdef double delta_s = s_indices[1] - s_indices[0]
    cdef double delta_t = t_indices[1] - t_indices[0]
    cdef double start_s = s_indices[0]
    cdef uint8[:,:] obstacles = np.frombuffer(obstacles_bool, dtype=np.uint8).reshape(obstacles_bool.shape)

    encountered_array = np.zeros((num_t, num_s, num_s), dtype=np.uint8)
    previous_array = np.zeros((num_t, num_s, num_s), dtype=np.int32)

    cdef uint8[:,:,:] encountered = encountered_array
    cdef int[:,:,:] previous = previous_array

    cdef double estimated_previous_s = start_s - ego_start_speed * delta_t

    cdef int entry_order = 0

    cdef Range first_s_range = get_feasible_next_s_range_no_jerk_limits(start_s, estimated_previous_s, delta_t)

    # We transform the raw s value range to a list of possible s indices

    cdef IntRange possible_first_s_indices = get_all_range_indices(start_s, delta_s, first_s_range.min, first_s_range.max)
    cdef int s_index
    cdef double s_value
    cdef double s_cost

    node_priority_queue = []
    for s_index in range(possible_first_s_indices.min, possible_first_s_indices.max):
        if obstacles[1, s_index]:
            continue
        s_value = s_indices[s_index]
        s_cost = cost(s_value, start_s, estimated_previous_s, delta_t, distances[1, s_index])
        node_priority_queue.append((s_cost, entry_order, 1, s_index, 0, 0))
        entry_order -= 1  # We want the queue to be LIFO, as this tends to be faster for shortest path problems

    heapq.heapify(node_priority_queue)

    cdef int best_last_s = 0
    cdef int best_second_last_s = 0
    cdef double total_cost
    cdef int t_index
    cdef int prev_s_index
    cdef int second_s_index
    cdef double prev_s_value

    cdef Range next_s_range
    cdef IntRange next_s_indices
    cdef int next_t
    cdef int next_s_index
    cdef double next_s_value

    cdef int best_t_index = 0
    cdef int _ # This value isn't used

    while len(node_priority_queue) > 0:
        # We get the (t, s, prev_s) tuple with the lowest cost so far
        total_cost, _, t_index, s_index, prev_s_index, second_s_index = heapq.heappop(node_priority_queue)
        s_value = s_indices[s_index]
        prev_s_value = s_indices[prev_s_index]

        if encountered[t_index, s_index, prev_s_index]:
            continue
        else:
            encountered[t_index, s_index, prev_s_index] = True
            previous[t_index, s_index, prev_s_index] = second_s_index

        if t_index == num_t - 1:
            best_t_index = num_t - 1
            best_last_s = s_index
            best_second_last_s = prev_s_index
            break
        elif t_index > best_t_index:
            best_t_index = t_index
            best_last_s = s_index
            best_second_last_s = prev_s_index

        # Again, calculate the possible next values of s
        next_s_range = get_feasible_next_s_range_no_jerk_limits(s_value, prev_s_value, delta_t)
        next_s_indices = get_all_range_indices(start_s, delta_s, next_s_range.min, next_s_range.max)
        next_t = t_index + 1

        for next_s_index in range(next_s_indices.min, next_s_indices.max):
            # We can't exceed the bounds of our simulation, but if this is happening it may be a good idea to increase the planning horizon
            if next_s_index >= num_s:
                break

            if not encountered[next_t, next_s_index, s_index]:

                if obstacles[next_t, next_s_index]:
                    continue  # No colliding with obstacles
                next_s_value = s_indices[next_s_index]
                s_cost = cost(next_s_value, s_value, prev_s_value, delta_t, distances[next_t, next_s_index])
                heapq.heappush(node_priority_queue, (total_cost + s_cost, entry_order, next_t, next_s_index, s_index, prev_s_index))
                entry_order -= 1

    # Reconstruct the best sequence of s values, using the saved values from best_previous_s
    cdef int best_s_index = best_last_s
    cdef int best_prev_s_index = best_second_last_s

    s_sequence = np.zeros(num_t)
    for t_index in range(best_t_index, 1, -1):
        s_sequence[t_index] = s_indices[best_s_index]
        second_s_index = previous[t_index, best_s_index, best_prev_s_index]
        best_s_index = best_prev_s_index
        best_prev_s_index = second_s_index

    s_sequence[0] = s_indices[best_prev_s_index]
    s_sequence[1] = s_indices[best_s_index]

    return s_sequence


def solve_s_t_path_no_jerk_fast(obstacles_bool, double[:] s_indices, double[:] t_indices, double ego_start_speed, double[:,:] distances):
    cdef Py_ssize_t num_s = s_indices.shape[0]
    cdef Py_ssize_t num_t = t_indices.shape[0]
    cdef double delta_s = s_indices[1] - s_indices[0]
    cdef double delta_t = t_indices[1] - t_indices[0]
    cdef double start_s = s_indices[0]
    cdef uint8[:,:] obstacles = np.frombuffer(obstacles_bool, dtype=np.uint8).reshape(obstacles_bool.shape)

    encountered_array = np.zeros((num_t, num_s), dtype=np.uint8)
    previous_array = np.zeros((num_t, num_s), dtype=np.int32)

    cdef uint8[:,:] encountered = encountered_array
    cdef int[:,:] previous = previous_array

    cdef double estimated_previous_s = start_s - ego_start_speed * delta_t

    cdef int entry_order = 0

    cdef Range first_s_range = get_feasible_next_s_range_no_jerk_limits(start_s, estimated_previous_s, delta_t)

    # We transform the raw s value range to a list of possible s indices

    cdef IntRange possible_first_s_indices = get_all_range_indices(start_s, delta_s, first_s_range.min, first_s_range.max)
    cdef int s_index
    cdef double s_value
    cdef double s_cost

    node_priority_queue = []
    for s_index in range(possible_first_s_indices.min, possible_first_s_indices.max):
        if obstacles[1, s_index]:
            continue
        s_value = s_indices[s_index]
        s_cost = cost(s_value, start_s, estimated_previous_s, delta_t, distances[1, s_index])
        node_priority_queue.append((s_cost, entry_order, 1, s_index, 0))
        entry_order -= 1  # We want the queue to be LIFO, as this tends to be faster for shortest path problems

    heapq.heapify(node_priority_queue)

    cdef double total_cost
    cdef int t_index
    cdef int prev_s_index
    cdef double prev_s_value

    cdef Range next_s_range
    cdef IntRange next_s_indices
    cdef int next_t
    cdef int next_s_index
    cdef double next_s_value

    cdef int best_last_s = 0
    cdef int best_t_index = 0
    cdef int _ # This value isn't used

    while len(node_priority_queue) > 0:
        # We get the (t, s, prev_s) tuple with the lowest cost so far
        total_cost, _, t_index, s_index, prev_s_index = heapq.heappop(node_priority_queue)
        s_value = s_indices[s_index]
        prev_s_value = s_indices[prev_s_index]

        if encountered[t_index, s_index]:
            continue
        else:
            encountered[t_index, s_index] = True
            previous[t_index, s_index] = prev_s_index

        if t_index == num_t - 1:
            best_t_index = num_t - 1
            best_last_s = s_index
            break
        elif t_index > best_t_index:
            best_t_index = t_index
            best_last_s = s_index

        # Again, calculate the possible next values of s
        next_s_range = get_feasible_next_s_range_no_jerk_limits(s_value, prev_s_value, delta_t)
        next_s_indices = get_all_range_indices(start_s, delta_s, next_s_range.min, next_s_range.max)
        next_t = t_index + 1

        for next_s_index in range(next_s_indices.min, next_s_indices.max):
            # We can't exceed the bounds of our simulation, but if this is happening it may be a good idea to increase the planning horizon
            if next_s_index >= num_s:
                break

            if not encountered[next_t, next_s_index]:

                if obstacles[next_t, next_s_index]:
                    continue  # No colliding with obstacles

                next_s_value = s_indices[next_s_index]
                s_cost = cost(next_s_value, s_value, prev_s_value, delta_t, distances[next_t, next_s_index])
                heapq.heappush(node_priority_queue, (total_cost + s_cost, entry_order, next_t, next_s_index, s_index))
                entry_order -= 1

    # Reconstruct the best sequence of s values, using the saved values from best_previous_s
    cdef int best_s_index = best_last_s

    s_sequence = np.zeros(num_t)
    for t_index in range(best_t_index, 0, -1):
        s_sequence[t_index] = s_indices[best_s_index]
        best_s_index = previous[t_index, best_s_index]

    s_sequence[0] = s_indices[best_s_index]

    return s_sequence


def solve_s_t_path_fast(obstacles_bool, double[:] s_values, double[:] t_indices, double ego_start_speed, double ego_start_acceleration, double[:,:] distances, double d_weight, double v_weight, double a_weight, double j_weight, double desired_speed, double max_speed, double negative_acceleration_limit, double positive_acceleration_limit, double negative_jerk_limit, double positive_jerk_limit, double min_allowed_distance):
    cdef Py_ssize_t num_s = s_values.shape[0]
    cdef Py_ssize_t num_t = t_indices.shape[0]
    cdef double delta_s = s_values[1] - s_values[0]
    cdef double delta_t = t_indices[1] - t_indices[0]
    cdef double start_s = s_values[0]
    cdef uint8[:,:] obstacles = np.frombuffer(obstacles_bool, dtype=np.uint8).reshape(obstacles_bool.shape)

    encountered_array = np.zeros((num_t, num_s), dtype=np.uint8)
    previous_array = np.zeros((num_t, num_s), dtype=np.int32)

    cdef uint8[:,:] encountered = encountered_array
    cdef int[:,:] previous = previous_array

    cdef double estimated_previous_s = start_s - ego_start_speed * delta_t
    cdef double estimated_second_s = estimated_previous_s - delta_t * (ego_start_speed - ego_start_acceleration * delta_t)

    cdef double total_cost
    cdef int t_index
    cdef int s_index
    cdef double s_value
    cdef int prev_s_index
    cdef double prev_s_value
    cdef int second_s_index
    cdef double second_s_value
    cdef double s_cost

    node_priority_queue = [(0, 0, 0, start_s, 0, estimated_previous_s, 0, estimated_second_s)]
    heapq.heapify(node_priority_queue)


    cdef Range next_s_range
    cdef IntRange next_s_indices
    cdef int next_t
    cdef int next_s_index
    cdef double next_s_value

    cdef int best_last_s = 0
    cdef int best_t_index = 0

    while len(node_priority_queue) > 0:
        # We get the (t, s, prev_s) tuple with the lowest cost so far
        total_cost, t_index, s_index, s_value, prev_s_index, prev_s_value, second_s_index, second_s_value = heapq.heappop(node_priority_queue)

        if encountered[t_index, s_index]:
            continue
        else:
            encountered[t_index, s_index] = True
            previous[t_index, s_index] = prev_s_index

        if t_index > best_t_index:
            best_t_index = t_index
            best_last_s = s_index
        if t_index == num_t - 1:
            break

        # Calculate the possible next values of s
        next_s_range = get_feasible_next_s_range_with_jerk_limits(s_value, prev_s_value, second_s_value, delta_t, negative_jerk_limit, positive_jerk_limit, negative_acceleration_limit, positive_acceleration_limit, max_speed)
        next_s_indices = get_all_range_indices(start_s, delta_s, next_s_range.min, next_s_range.max)
        next_t = t_index + 1

        for next_s_index in range(next_s_indices.min, next_s_indices.max):
            # We can't exceed the bounds of our simulation, but if this is happening it may be a good idea to increase the planning horizon

            if next_s_index >= num_s:
                break

            if not encountered[next_t, next_s_index]:
                if obstacles[next_t, next_s_index]:
                    continue  # No colliding with obstacles

                next_s_value = s_values[next_s_index]
                s_cost = cost_with_jerk(next_s_value, s_value, prev_s_value, second_s_value, delta_t, distances[next_t, next_s_index], min_allowed_distance, v_weight, desired_speed, a_weight, j_weight, d_weight)
                heapq.heappush(node_priority_queue, (total_cost + s_cost, next_t, next_s_index, next_s_value, s_index, s_value, prev_s_index, prev_s_value))

    # Reconstruct the best sequence of s values, using the saved values from best_previous_s
    cdef int best_s_index = best_last_s

    s_sequence = np.zeros(num_t)
    for t_index in range(best_t_index, 0, -1):
        s_sequence[t_index] = s_values[best_s_index]
        best_s_index = previous[t_index, best_s_index]

    s_sequence[0] = s_values[best_s_index]
    return s_sequence

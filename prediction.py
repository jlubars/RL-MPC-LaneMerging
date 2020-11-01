import numpy as np
import matplotlib.pyplot as plt

import control
from config import Settings
import st


class HighwayState:

    ego_reaction_threshold = 8
    ego_crash_threshold = 11

    def __init__(self, ego_position, ego_speed, ego_acceleration, other_xs, other_speeds, other_accelerations):
        self.ego_position = ego_position
        self.ego_speed = ego_speed
        self.ego_acceleration = ego_acceleration
        self.other_xs = other_xs
        self.other_speeds = other_speeds
        self.other_accelerations = other_accelerations

    def predict_step_without_ego(self, delta_t, min_crash_distance=5):
        # Assume ego will just follow the car in front as closely as possible if ego is merge, or ego just stays put otherwise
        ego_s = control.get_ego_s(self.ego_position)
        ego_x = self.ego_position[0]
        if ego_s < self.ego_reaction_threshold or len(self.other_xs) == 0:
            return self.predict_step_with_ego(0, delta_t, min_crash_distance)
        elif self.other_xs[0] < ego_x:
            # If ego car is in front and on highway, pretend it's not there
            modified_state = HighwayState((-20, -10), 0, 0, self.other_xs, self.other_speeds, self.other_accelerations)
            return modified_state.predict_step_with_ego(0, delta_t, min_crash_distance)
        else:
            last_speed = 0
            last_x = 0
            for i, car_x in enumerate(self.other_xs):
                if car_x < ego_x:
                    # Assume the ego car directly follows the car in front of it for the purposes of allocating
                    # space to other vehicles when predicting motion
                    modified_state = HighwayState((last_x - Settings.CAR_LENGTH - 5, self.ego_position[1]), last_speed, 0, self.other_xs, self.other_speeds, self.other_accelerations)
                    return modified_state.predict_step_with_ego(last_speed, delta_t, min_crash_distance)
                else:
                    last_speed = self.other_speeds[i]
                    last_x = car_x
            return self.predict_step_with_ego(last_speed, delta_t, min_crash_distance)

    def predict_step_with_ego(self, selected_speed, delta_t, min_crash_distance=5):
        current_x, current_y = self.ego_position
        if current_x < control.merge_point2[0]:
            # Approximate the chane in position as traveling straight towards the merge point
            direction = np.array([control.merge_point2[0] - current_x, control.merge_point2[1] - current_y])
            direction /= np.linalg.norm(direction)
            direction *= selected_speed * delta_t
            predicted_x = current_x + direction[0]
            predicted_y = current_y + direction[1]
            if predicted_y < -1.6:
                predicted_y = -1.6  # The lane to merge into is at y=-1.6
        else:
            predicted_y = current_y
            predicted_x = current_x + selected_speed * delta_t

        next_acceleration = (selected_speed - self.ego_speed) / delta_t

        # The ego can't crash into other cars until this point. Not 100% sure if accurate
        ego_can_crash = control.get_ego_s((predicted_x, predicted_y)) > self.ego_crash_threshold
        # Also don't let the other cars really react to the ego car until a threshold
        ego_has_merged = control.get_ego_s((predicted_x, predicted_y)) > self.ego_reaction_threshold

        # The other cars are in the order from front to back
        new_other_xs = []
        new_other_speeds = []
        new_other_accelerations = []
        last_x = np.inf
        last_speed = 0
        ego_encountered = False
        for other_car_index in range(len(self.other_xs)):
            other_speed = self.other_speeds[other_car_index]
            other_x = self.other_xs[other_car_index]
            if other_x < predicted_x and not ego_encountered:
                ego_encountered = True
                if ego_has_merged:
                    last_x = predicted_x
                    last_speed = selected_speed
            speed_diff = last_speed - other_speed
            x_diff = last_x - other_x
            if speed_diff < 0 and x_diff < 30:
                new_other_acceleration = max(speed_diff, Settings.MAX_PREDICTED_DECELERATION)
                new_other_speed = other_speed + new_other_acceleration * delta_t
            else:
                new_other_acceleration = 0
                new_other_speed = other_speed
            predicted_next_position = other_x + new_other_speed * delta_t

            last_x = predicted_next_position
            last_speed = new_other_speed
            new_other_xs.append(predicted_next_position)
            new_other_speeds.append(new_other_speed)
            new_other_accelerations.append(new_other_acceleration)

        crashed = False
        crash_detection_distance = max(Settings.CAR_LENGTH, min_crash_distance)
        for x in new_other_xs:
            if abs(x - predicted_x) < crash_detection_distance and ego_can_crash:
                crashed = True

        return HighwayState((predicted_x, predicted_y), selected_speed, next_acceleration, new_other_xs, new_other_speeds, new_other_accelerations), crashed

    @classmethod
    def empty_state(cls):
        return cls(0, 0, 0, [], [], [])

    @classmethod
    def from_sumo(cls):
        ids = control.get_vehicle_ids()
        positions = control.get_vehicle_positions(ids)
        speeds = control.get_vehicle_speeds(ids)
        accelerations = control.get_vehicle_accelerations(ids)
        if "ego" in ids:
            ego_position = control.get_ego_position()
            ego_speed = control.get_ego_speed()
            ego_acceleration = control.get_ego_acceleration()
        else:
            ego_position = (-200, 0)
            ego_speed = 0
            ego_acceleration = 0
        other_xs = []
        other_speeds = []
        other_accelerations = []
        for vehicle in ids:
            # vehicles are already in order
            if vehicle == "ego":
                continue
            else:
                other_position = positions[vehicle]
                if control.distance(other_position, ego_position) < Settings.SENSOR_RADIUS:
                    other_xs.append(other_position[0])
                    other_speeds.append(speeds[vehicle])
                    other_accelerations.append(accelerations[vehicle])
        index_order = np.argsort(other_xs)
        other_xs = [other_xs[i] for i in reversed(index_order)]
        other_speeds = [other_speeds[i] for i in reversed(index_order)]
        other_accelerations = [other_accelerations[i] for i in reversed(index_order)]
        return cls(ego_position, ego_speed, ego_acceleration, other_xs, other_speeds, other_accelerations)

    def plot_state(self, i):
        ego_x, ego_y = self.ego_position
        plt.scatter(i, ego_x, color='r')
        last_x = np.inf
        had_car = False
        encountered_ego = False
        for other_index, x in enumerate(self.other_xs):
            if not encountered_ego and x < ego_x:
                encountered_ego = True
                if had_car:
                    plt.scatter(i, last_x, color='g')
                plt.scatter(i, x, color='g')
                break
            last_x = x
            had_car = True
        if had_car and not encountered_ego:
            plt.scatter(i, last_x, color='g')

    def get_closest_cars(self):
        ego_x, ego_y = self.ego_position
        index_behind = -1
        index_front = -1
        last_index = -1
        for other_index, x in enumerate(self.other_xs):
            if x < ego_x:
                index_behind = other_index
                break
            last_index = other_index
        if last_index != -1:
            index_front = last_index
        if index_front != -1:
            car_front = (self.other_xs[index_front], self.other_speeds[index_front], self.other_accelerations[index_front])
        else:
            car_front = None
        if index_behind != -1:
            car_behind = (self.other_xs[index_behind], self.other_speeds[index_behind], self.other_accelerations[index_behind])
        else:
            car_behind = None
        return car_front, car_behind

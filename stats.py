import numpy as np
from scipy.stats import sem
import logging
import matplotlib.pyplot as plt
import os
from csv import DictWriter
import pandas as pd

from config import Settings


class StatsAggregator:

    def __init__(self, save_state_on_crash=False):
        self.crashed = []
        self.merged = []
        self.mean_speed = []
        self.max_speed = []
        self.mean_abs_jerk = []
        self.closest_distance = []
        self.mean_closest_distance = []
        self.mean_abs_jerk_merged = []
        self.mean_disruption = []
        self.max_disruption = []
        self.total_disruption = []
        self.disruption_time = []
        self.closest_distance_merged = []
        self.mean_closest_distance_merged = []
        self.time_taken = []
        self.time_to_merge = []
        self.clock_time_taken = []
        self.clock_time_per_step = []
        self.bins = np.arange(-220, 61, 20)
        self.custom_stat_lists = {}
        self.episodes = 0
        self.last_episode_stats = None
        self.save_state_on_crash = save_state_on_crash
        self.custom_stats_function = None
        self.counts = np.zeros(len(self.bins) - 1)
        self.jerks = np.zeros(len(self.bins) - 1)
        self.speeds = np.zeros(len(self.bins) - 1)

    def add_episode_stats(self, episode_stats):
        xs = [pos[0] for pos in episode_stats["position_history"]]
        hist, bins = np.histogram(xs, self.bins)
        self.counts += hist
        current_bin = 0
        for i, x in enumerate(xs):
            while x > self.bins[current_bin+1]:
                current_bin += 1
            self.jerks[current_bin] += abs(episode_stats["jerk_history"][i])
            self.speeds[current_bin] += abs(episode_stats["speed_history"][i])
        self.last_episode_stats = episode_stats
        self.crashed.append(episode_stats["crashed"])
        self.merged.append(episode_stats["merged"])
        self.time_taken.append(episode_stats["simulation_time_taken"])
        self.mean_speed.append(np.mean(episode_stats["speed_history"]))
        self.max_speed.append(np.max(episode_stats["speed_history"]))
        self.mean_abs_jerk.append(np.mean(np.abs(episode_stats["jerk_history"])))
        if len(episode_stats["closest_vehicle_history"]) > 0:
            self.closest_distance.append(min(episode_stats["closest_vehicle_history"]))
            self.mean_closest_distance.append(np.mean(episode_stats["closest_vehicle_history"]))
        self.clock_time_taken.append(episode_stats["end_time"] - episode_stats["start_time"])
        self.clock_time_per_step.append(self.clock_time_taken[-1]/len(episode_stats["speed_history"]))
        if len(episode_stats["disruption_history"]) > 0:
            self.mean_disruption.append(np.mean(episode_stats["disruption_history"]))
            self.max_disruption.append(np.max(episode_stats["disruption_history"]))
            self.total_disruption.append(np.sum(episode_stats["disruption_history"]) * Settings.TICK_LENGTH)
            self.disruption_time.append(np.count_nonzero(episode_stats["disruption_history"]) * Settings.TICK_LENGTH)
        if episode_stats["merged"]:
            self.time_to_merge.append(episode_stats["simulation_time_taken"])
            self.closest_distance_merged.append(min(episode_stats["closest_vehicle_history"]))
            self.mean_closest_distance_merged.append(np.mean(episode_stats["closest_vehicle_history"]))
            self.mean_abs_jerk_merged.append(np.mean(np.abs(episode_stats["jerk_history"])))
        if episode_stats["crashed"] and self.save_state_on_crash:
            import pickle
            pickle.dump(episode_stats["state_history"], open("crashed_state_history.pkl", 'wb'))

        if self.custom_stats_function is not None:
            custom_stats = self.custom_stats_function(episode_stats)
            for key, value in custom_stats.items():
                if key not in self.custom_stat_lists:
                    self.custom_stat_lists[key] = [value]
                else:
                    self.custom_stat_lists[key].append(value)

    def add_custom_stat_callback(self, callback):
        self.custom_stats_function = callback

    def get_stats(self):
        aggregate_stats = {
            "crashed": self.crashed,
            "merged": self.merged,
            "mean_speed": self.mean_speed,
            "max_speed": self.max_speed,
            "mean_abs_jerk": self.mean_abs_jerk,
            "closest_distance": self.closest_distance,
            "mean_closest_distance": self.mean_closest_distance,
            "mean_abs_jerk_merged": self.mean_abs_jerk_merged,
            "closest_distance_merged": self.closest_distance_merged,
            "mean_closest_distance_merged": self.mean_closest_distance_merged,
            "mean_disruption": self.mean_disruption,
            "max_disruption": self.max_disruption,
            "total_disruption": self.total_disruption,
            "disruption_time": self.disruption_time,
            "time_taken": self.time_taken,
            "time_to_merge": self.time_to_merge,
            "clock_time_per_episode": self.clock_time_taken,
            "clock_time_per_step": self.clock_time_per_step,
        }
        for key, value in self.custom_stat_lists.items():
            aggregate_stats[key] = value
        return aggregate_stats

    def print_stats(self):
        Settings.ensure_run_plot_directory()
        plot_dir = Settings.get_run_plot_directory()
        aggregate_stats = self.get_stats()
        avg_jerks = self.jerks / self.counts
        avg_speeds = self.speeds / self.counts
        print("Average jerks per segment:")
        for i in range(len(self.counts)):
            print("{} to {}: {}".format(self.bins[i], self.bins[i+1], avg_jerks[i]))
        plt.figure()
        plt.bar(self.bins[:-1], avg_jerks, width=np.diff(self.bins), edgecolor="black", align="edge")
        plt.ylim(0, 5.0)
        plt.savefig(os.path.join(plot_dir, "avg_jerks"))
        plt.close()
        plt.figure()
        plt.bar(self.bins[:-1], avg_speeds, width=np.diff(self.bins), edgecolor="black", align="edge")
        plt.ylim(0, 30.0)
        plt.savefig(os.path.join(plot_dir, "avg_speeds"))
        plt.close()
        for stat_name in aggregate_stats:
            stat = aggregate_stats[stat_name]
            mean = np.mean(stat)
            err = sem(stat)
            message = "{}: {} {} {}".format(
                stat_name, mean, '\u00B1', err
            )
            logging.log(logging.INFO, message)
            print(message)
        self.add_csv_data()

    def get_stat_averages(self, report_stds=False):
        averages = {}
        stds = {}
        aggregate_stats = self.get_stats()
        for stat_name in aggregate_stats:
            stat = aggregate_stats[stat_name]
            mean = np.mean(stat)
            err = sem(stat)
            averages[stat_name] = mean
            stds[stat_name] = err
        if report_stds:
            return averages, stds
        else:
            return averages

    def get_stat_report_row_dict(self):
        averages, stds = self.get_stat_averages(report_stds=True)
        columns = {}
        for name in averages:
            columns[name] = averages[name]
            columns[name + "_std"] = stds[name]
        settings_dict = Settings.export_settings()
        for key, value in settings_dict.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, float):
                columns[key] = value
        if Settings.USE_ALTERNATE_TRAFFIC_DISTRIBUTION:
            traffic = "joseph_{}".format(Settings.TRAFFIC_DENSITY)
        elif Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
            static = "varying" if Settings.VARY_TRAFFIC_START_TIMES else "constant"
            traffic = "uniform-{}-{}-{}".format(Settings.OTHER_CAR_SPEED, Settings.BASE_TRAFFIC_INTERVAL, static)
        else:
            traffic = "harsh"
        st_signature = "st-{}-{}-{}-{}-{}-{}-{}-{}".format(
            Settings.V_WEIGHT,
            Settings.A_WEIGHT,
            Settings.J_WEIGHT,
            Settings.A_WEIGHT,
            Settings.MIN_ALLOWED_DISTANCE,
            Settings.CRASH_MIN_S,
            Settings.START_UNCERTAINTY,
            Settings.UNCERTAINTY_PER_SECOND
        )
        columns["ST_DESCRIPTION"] = st_signature
        columns["TRAFFIC_DESCRIPTION"] = traffic
        columns["TIME"] = pd.Timestamp.now(tz="US/Pacific")
        return columns

    def add_csv_data(self):
        if os.path.exists("run_data.csv"):
            dataframe = pd.read_csv("run_data.csv")
        else:
            dataframe = pd.DataFrame()
        new_row = pd.DataFrame([self.get_stat_report_row_dict()])
        combined = dataframe.append(new_row)
        combined.to_csv("run_data.csv", index=False)


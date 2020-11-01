import os
import sys
from config import Settings
import logging


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')

    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Must be done after SUMO_HOME is set
import traci


def get_sumo_binary():
    sumo_binary = "sumo"
    if Settings.USE_GUI:
        if Settings.SYSTEM == "Windows":
            sumo_binary = "sumo-gui.exe"
        elif Settings.SYSTEM == "Linux":
            sumo_binary = "sumo-gui"
    else:
        if Settings.SYSTEM == "Windows":
            sumo_binary = "sumo.exe"
        elif Settings.SYSTEM == "Linux":
            sumo_binary = "sumo"
    return sumo_binary


def start_sumo():
    sumo_binary = get_sumo_binary()
    sumo_cmd = [sumo_binary, "-c", "ramp.sumocfg", "--step-length", str(Settings.TICK_LENGTH)]
    if Settings.USE_ALTERNATE_TRAFFIC_DISTRIBUTION:
        if Settings.TRAFFIC_DENSITY == "low":
            sumo_cmd.extend(["--route-files", "merge2.rou.xml"])
        elif Settings.TRAFFIC_DENSITY == "medium":
            sumo_cmd.extend(["--route-files", "merge2b.rou.xml"])
        elif Settings.TRAFFIC_DENSITY == "high":
            sumo_cmd.extend(["--route-files", "merge2c.rou.xml"])
        else:
            raise ValueError("Unkown TRAFFIC_DENSITY: {}".format(Settings.TRAFFIC_DENSITY))
    elif Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
        sumo_cmd.extend(["--route-files", "merge_impossible.rou.xml"])
    if Settings.SEED != "Random":
        sumo_cmd.extend(["--seed", str(Settings.SEED)])
    else:
        sumo_cmd.extend(["--random"])
    traci.start(sumo_cmd)
    if Settings.USE_SIMPLE_TRAFFIC_DISTRIBUTION:
        traci.vehicletype.setMaxSpeed("normal", Settings.OTHER_CAR_SPEED)


def exit_sumo():
    traci.close()


def change_step_size(new_step_size):
    exit_sumo()
    Settings.TICK_LENGTH = new_step_size
    start_sumo()



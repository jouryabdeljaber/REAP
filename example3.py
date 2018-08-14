from simulate_manipulation import *
import numpy as np


def traj(t):
    spd = 0.4
    return np.array([spd, spd, -spd, spd, 0., 0., 0., 0.])


B = 100.*np.eye(8)
time = 0.5

(object_positions, manipulator_positions, object_velocities, manipulator_velocities, times) = simulateRobot(time, B, traj)


raw_input("press Enter to continue")

playbackMotion(object_positions, manipulator_positions, object_velocities, manipulator_velocities, times)


raw_input("press Enter to continue")

(pred_object_positions, pred_manipulator_positions, pred_object_velocities, pred_manipulator_velocities, pred_times) = getPredictedMotion(B, traj, object_positions, manipulator_positions, object_velocities, manipulator_velocities, times)

raw_input("press Enter to continue")

playbackMotion(pred_object_positions, pred_manipulator_positions, pred_object_velocities, pred_manipulator_velocities, pred_times)

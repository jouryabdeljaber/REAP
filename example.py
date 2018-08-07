from simulate_manipulation import *
import numpy as np


def traj(t):
    spd = 0.4
    return np.array([spd, spd, -spd, spd, 0., 0., 0., 0.])


B = 10*np.eye(8)
time = 3

(positions, times) = simulateRobot(time, B, traj)

import numpy as np
import math
raceline = np.loadtxt('carla_center.txt', delimiter=',').T

raceline_new = np.zeros((raceline.shape[0], 4))
raceline_new[:, :2] = raceline
raceline_new[:, 2] = 1.5
raceline_new[:, 3] = 1.5

np.savetxt('../../global_racetrajectory_optimization/inputs/tracks/carla.csv', raceline_new, delimiter=',', fmt='%.3f')
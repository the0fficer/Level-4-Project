import numpy as np
import matplotlib.pyplot as plt
import GPy
import TrajectoryPlotter as TP
from time import time
import matplotlib.gridspec as gridspec


total_time = time()

# trajectories = TP.array_unpacker(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_as_arrays\2000birds101timesteps20201114-154317")
# TP.trajectory_plotter(trajectories)
# def nearest_neighbour_sq_dist(trajectories):
#     n_birds = np.shape(trajectories)[0]
#     n_timesteps = np.shape(trajectories)[-1]
#     NN_sq_dist_array = np.zeros((n_birds, n_timesteps))
#     for t in range(n_timesteps):
#         for b in range(n_birds):
#             intermediate1_array = (trajectories[b,:,t] - trajectories[:b,:,t])**2
#             intermediate2_array = (trajectories[b, :, t] - trajectories[b+1:, :, t]) ** 2
#             intermediate_array = np.concatenate((intermediate1_array,intermediate2_array), axis=0)
#             NN_sq_dist_array[b,t] = min(np.sum(intermediate_array,axis=1)) # Counting Nearest Neighbour distance squared including itself and double counting.
#     return NN_sq_dist_array
NN_sq_dist_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\NearestNeighbourSqDistances\NN_sq_dist_300b_501t")
NN_dist_array = NN_sq_dist_array**0.5
velocities = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\NearestNeighbourSqDistances\velocities300b_501t")
speeds = np.sum(velocities**2, axis=1)**0.5

fig = plt.figure(figsize=(9.,4.))
outer_grid = gridspec.GridSpec(2, 1, figure=fig, left=0.1, right=0.975,top=0.975, wspace=0.3)

left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
right_cell = outer_grid[1].subgridspec(1, 1)

ax0 = fig.add_subplot(left_cell[:, :])
ax1 = fig.add_subplot(right_cell[:, :])

n_bins = 100
ax0.hist(NN_dist_array[:,0], n_bins, histtype='step', fill=False, density=True, label="In the first frame")
ax0.hist(NN_dist_array.flatten(), n_bins, histtype='step', fill=False, density=True, label= "Over all the frames")
ax0.set_xlabel("Distance to Nearest Neighbour")
ax1.hist(speeds[:,0], n_bins, histtype='step', fill=False, density=True, label="In the first frame")
ax1.hist(speeds.flatten(), n_bins, histtype='step', fill=False, density=True, label="Over all the frames")
ax1.set_xlabel("Speed")
fig.legend()
plt.show()
print("--- %s ---" % TP.seconds_to_str((time() - total_time)))
import numpy as np
import matplotlib.pyplot as plt
import GPy
import TrajectoryPlotter as TP
from time import time
import gp_reconstruction as gpr
import matplotlib.gridspec as gridspec

total_time = time()


# trajectories = TP.array_unpacker("trajectories_as_arrays2/2birds1000timesteps20201126-111657")
# TP.trajectory_plotter(trajectories)
# X, gp1 = TP.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=3, fraction=0.1)
# X, gp2 = TP.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=6, fraction=0.1)
# X, gp3 = TP.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=9, fraction=0.1)
# # trajectories = TP.array_fractional_reducer(trajectories, 0.1, 2)
# # X, gp = TP.three_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=.1, length=10.)
# newX = np.linspace(0.,11.,100)
# TP.prediction_plotter(newX,gp1)
# TP.prediction_plotter(newX,gp2)
# TP.prediction_plotter(newX,gp3)
#########################################

# t = np.linspace(0,10,100)
# xyz = np.zeros((3, np.shape(t)[0]))
# xyz[0, :] = np.cos(t)
# xyz[1, :] = np.sin(t)
# TP.trajectory_plotter(xyz[None, :, :])
# # T, gp2 = TP.three_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10.)
# # newT = np.linspace(0.,25.,100)
# # TP.prediction_plotter(newT,gp2)
# newT = np.linspace(0.,25.,100)
# T, gp4 = TP.multi_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10., n_dimensions=3, fraction=1.)
# TP.prediction_plotter(newT,gp4)
# T, gp5 = TP.multi_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10., n_dimensions=6, fraction=1.)
# TP.prediction_plotter(newT,gp5)
# T, gp6 = TP.multi_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10., n_dimensions=9, fraction=1.)
# TP.prediction_plotter(newT,gp6)


######################################
# print("CHECK!! Using absolute path in line 43 so check that the path hasn't changed since last use.")
# trajectories = TP.array_unpacker(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_as_arrays2\2birds1000timesteps20201126-111657")
# # TP.trajectory_plotter(trajectories)
# #X, gp1 = TP.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=3, fraction=0.1)
# #newX = np.linspace(0.,11.,100)
# #TP.prediction_plotter(newX,gp1)
#
# fraction = 0.1
# last_index = 700
# trajectories = trajectories[:, :, :last_index]
# trajectories = TP.array_fractional_reducer(trajectories, fraction, 2)
# # trajectories = np.flip(trajectories, axis=2)
# # TP.report_graph_plotter(trajectories[0, :, :],trajectories[1, :, :], int(fraction*500), int(fraction*50), fraction=fraction, length=last_index/100)
# TP.presentation_graph_plotter(trajectories[0, :, :],trajectories[1, :, :], int(fraction*500), int(fraction*50), fraction=fraction, length=last_index/100)
#
# # X = np.arange(1,5)
# # X = X[:,None]
# # X_list = [X,X]
# # Y = np.arange(1,5)
# # Y1 = Y[:,None]
# # Y2 = Y[:+5,None]
# # Y_list = [Y1,Y2]
# # print(TP.build_XY(X_list,Y_list))
#
#
# # times = np.linspace(0.,9.99,1000)
# # assert np.shape(trajectories)[-1] == np.shape(times)[-1]
# # fraction = 0.1
# # trajectories = TP.array_fractional_reducer(trajectories, fraction, 2)
# # times = TP.array_fractional_reducer(times, fraction, 0)
# # input_trajectory, output_trajectory = TP.trajectory_masker(trajectories[0,:,:],  int(650*fraction), int(150*fraction))
#
# # TP.train_GPs_on_position([input_trajectory],[output_trajectory],times)

#################
# trajectories = TP.array_unpacker(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_from_250b_8001t\trajectory_250b_500t_7")
# gpr.trajectory_plotter(trajectories)
# fraction = 0.1
# threshold_distance = 1.3
# trajectories = gpr.array_fractional_reducer(trajectories, fraction, 2)
# broken_trajectories, trajectories_altered, trajectories_altered_between_times = gpr.break_trajectories(trajectories, threshold_distance)
# list_of_input_lists, list_of_output_lists, list_of_occlusion_lists, trajectories = gpr.masker(broken_trajectories,trajectories_altered,trajectories_altered_between_times)
#
# TP.array_save(trajectories, r"masked_trajectories", r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Presentation_data")
# TP.array_save(trajectories_altered, r"indices_of_altered_trajectories", r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Presentation_data")
####
trajectories = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Presentation_data\masked_trajectories")
trajectories_altered = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Presentation_data\indices_of_altered_trajectories")

n_birds, n_axis, n_timesteps = np.shape(trajectories)
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig = plt.figure(figsize=(9.,4.))
outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975,top=0.975, wspace=0.23)

left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
ax2 = fig.add_subplot(left_cell[:, :])
right_cell = outer_grid[1].subgridspec(2, 1, hspace=0.3)
ax1 = fig.add_subplot(right_cell[1], sharex=ax2)
ax3 = fig.add_subplot(right_cell[0], sharey=ax1)
ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

ax1.set_ylabel("Y (m)")
ax1.set_xlabel("X (m)")
ax2.set_ylabel("Z (m)")
ax2.set_xlabel("X (m)")
ax3.set_ylabel("Y (m)")
ax3.set_xlabel("Z (m)")
alpha=0.1
alpha2 = 1
for i in range(n_birds):
    ax1.plot(trajectories[i, 0, :], trajectories[i, 1, :], alpha=alpha)
    ax2.plot(trajectories[i, 0, :], trajectories[i, 2, :], alpha=alpha)
    ax3.plot(trajectories[i, 2, :], trajectories[i, 1, :], alpha=alpha)
for k in [0,5]:
    for j in range(len(trajectories_altered[k])):
        ax1.plot(trajectories[trajectories_altered[k][j], 0, :], trajectories[trajectories_altered[k][j], 1, :], alpha=alpha2)
        ax2.plot(trajectories[trajectories_altered[k][j], 0, :], trajectories[trajectories_altered[k][j], 2, :], alpha=alpha2)
        ax3.plot(trajectories[trajectories_altered[k][j], 2, :], trajectories[trajectories_altered[k][j], 1, :], alpha=alpha2)
fig.show()
##################

print("--- %s ---" % TP.seconds_to_str((time() - total_time)))



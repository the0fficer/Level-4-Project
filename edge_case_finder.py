import numpy as np
import matplotlib.pyplot as plt
import GPy
import gp_reconstruction as gpr
from time import time
import os
import matplotlib.gridspec as gridspec
import TrajectoryPlotter as TP
np.random.seed(0)

total_time = time()

# plot_cost=False
# from_scratch = False
# adjusting_output_len = False
# coregionalised = False
# acceleration=True
# swapping_IO=False
# if acceleration:
#     assert(swapping_IO==False)
# folder_with_trajectories = f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\trajectories_for_analysing"
# # number_of_trajectories = len(os.listdir(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_for_analysing"))
# number_of_trajectories = len(os.listdir(folder_with_trajectories))
#
# if from_scratch:
#     frame_rates = np.array([24, 23, 22, 21, 20, 19, 18])
#     fraction, threshold_distances = gpr.framerate_to_fraction_and_threshold_distance(frame_rates, delta_t=0.01, max_speed=13.5, )
#     number_of_threshold_distances = len(threshold_distances)
#     independent_variable = frame_rates
#     independent_variable_axis_title = "Frame rate (fps)"
# elif adjusting_output_len:
#     output_max_length = np.array([1,3,5,10, 15, 25, 50])
#     threshold_distances = 1.32*np.ones_like(output_max_length)
#     fraction = 0.1*np.ones_like(threshold_distances)
#     independent_variable = output_max_length
#     independent_variable_axis_title = "Number of positions in output"
# else:
#     # number_of_threshold_distances = 1
#     # threshold_distances = np.flip(np.array([1.225,1.25,1.275,1.3, 1.325,1.35,1.375,1.4,1.45,1.5]))
#     threshold_distances = np.flip(np.array([1.25]))
#     # threshold_distances = np.flip(np.linspace(1.175, 1.375, number_of_threshold_distances))
#     output_max_length = 3*np.ones_like(threshold_distances, dtype=int)
#     fraction = 0.1*np.ones_like(threshold_distances)
#     independent_variable = threshold_distances
#     independent_variable_axis_title = "Threshold Distance (m)"
#     adjusting_output_len = True
#
#
# for index, local_path in enumerate(os.listdir(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_for_analysing")):
# # for index, local_path in enumerate(os.listdir(folder_with_trajectories)):
#     print("CHECK!! Using absolute path in line 43 so check that the path hasn't changed since last use.")
#     absolute_path = f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\trajectories_for_analysing\\{local_path}"
#     trajectories = gpr.array_unpacker(absolute_path)
#     times_array = np.linspace(0, 5., np.shape(trajectories)[2])
#
#     for index3 in range(len(independent_variable)):
#         print(f"\nFolder: {local_path}")
#         print(f"Independent Variable: {independent_variable[index3]}")
#         intermediate_time = time()
#         gpr.edge_case_reconstructor(
#             trajectories, threshold_distances[index3], from_scratch=False, fraction=fraction[index3], length=5., n_restarts=3, verbose=False,
#             lengthscales=False, print_cost_matrix=False, adjusting_output_len=adjusting_output_len,
#             output_max_length=output_max_length[index3], swapping_IO=swapping_IO, acceleration=acceleration, trajectory_title=local_path)
#
#         print("--- %s ---" % gpr.seconds_to_str((time() - intermediate_time)))
#     print("--- %s ---" % gpr.seconds_to_str((time() - total_time)))


##################

# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CF_NS\1.25TD_trajectory_250b_500t_14_25_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CF_NS\1.25TD_trajectory_250b_500t_14_25_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NF\Times_array")
# # gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# # gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=True, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)

####
# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NF\1.25TD_trajectory_250b_500t_13_9_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NF\1.25TD_trajectory_250b_500t_13_9_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NF\Times_array")
# gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)

# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_12_2_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_12_2_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\Times_array")
# gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
#
# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_13_10_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_13_10_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\Times_array")
# gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
#######

# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_14_6_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_14_6_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\Times_array")
# # gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# # gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=True, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)

#######
# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_14_17_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_14_17_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\Times_array")
# gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)

# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_14_21_input")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\1.25TD_trajectory_250b_500t_14_21_output")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AF_CS_NS\Times_array")
# gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
# gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
#
input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AS_CF_NS\1.25TD_trajectory_250b_500t_14_29_input")
output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AS_CF_NS\1.25TD_trajectory_250b_500t_14_29_output")
times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\Edge Cases\AS_CF_NS\Times_array")


# fig = plt.figure(figsize=(6.25, 2.8))
# outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3, bottom=0.15)
#
# left_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[0])
#
# axx = fig.add_subplot(left_cell[0])
# axy = fig.add_subplot(left_cell[1])
# axz = fig.add_subplot(left_cell[2])
# right_cell = outer_grid[1].subgridspec(5, 3, hspace=0.05)
# upper_right_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_cell[:5, :], hspace=0.0)
# # lower_right_cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_cell[3:, :], hspace=0.0)
# # upper_right_cell = right_Cell[:3, :].subgridspec(3, 1)
# # lower_right_cell = right_Cell[3:, :].subgridspec(2, 1)
#
# # axx = fig.add_subplot(right_cell[0, :])
# # axy = fig.add_subplot(right_cell[1, :])
# # axz = fig.add_subplot(right_cell[2, :])
# # ax2 = fig.add_subplot(right_cell[3, :])
# # ax3 = fig.add_subplot(right_cell[4, :])
# ax1 = fig.add_subplot(upper_right_cell[0])
# ax2 = fig.add_subplot(upper_right_cell[1], sharex=axx)
# ax3 = fig.add_subplot(upper_right_cell[2], sharex=axx)
# # ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
# # ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)
#
# # ax.set_xlabel('Z (m)')
# # ax.set_ylabel('X (m)')
#
# # axins = ax.inset_axes([0.3, 0.6, 0.375, 0.35])   #Bottom left: [0.175, 0.15, 0.375, 0.35]; Top Left: [0.175, 0.6, 0.375, 0.35]
# # axins.set_xlabel('Z (m)')
# # axins.set_ylabel('Y (m)')
#
# # ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# # axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#
# axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# ax1.tick_params(axis="both", labelbottom=False)
# ax2.tick_params(axis="both", labelbottom=False)
# axx.tick_params(axis="both", labelbottom=False)
# axy.tick_params(axis="both", labelbottom=False)
# # axz.tick_params(axis="both", labelbottom=False)
#
# axx.set_ylabel('X (m)')
# axy.set_ylabel('Y (m)')
# axz.set_ylabel('Z (m)')
# ax1.set_ylabel('Norm. Res. for X')
# ax2.set_ylabel('Norm. Res. for X')
# ax3.set_ylabel('Norm. Res. for X')
# axz.set_xlabel('Time (s)')
# # ax3.set_ylabel('Incorrect\nMatching')
# # ax2.set_ylabel('Correct\nMatching')
#
# cmap = plt.get_cmap('rainbow')
# colors = [cmap(i) for i in np.linspace(0, 1, len(input_list))]
# alpha = 0.1
# alpha2 = 1
#
#
#
#
#
trajectory1 = np.array(input_list[0])
trajectory2 = np.array(input_list[1])
n_length=2
n_split = gpr.latest_unmasked_index(input_list[0])+1

# TP.report_graph_plotter(trajectory1, trajectory2, n_split=n_split, n_length=n_length,fraction=0.1, length=5.)
# gpr.assignment_problem_plotter(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
gpr.summative_report_graph_plotter_coreg(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# masked_trajectories, col_ind, cost, row_ind = gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=True, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=False)
# print(f"Coregionalised assignment: {row_ind} -> {col_ind}")
# print(f"Cost: {cost}")
# # gpr.summative_report_graph_plotter_ind(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# masked_trajectories, col_ind, cost, row_ind = gpr.train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=True, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=False)
# print(f"Non-coregionalised assignment: {row_ind} -> {col_ind}")
# print(f"Cost: {cost}")
input_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
gpr.summative_report_graph_plotter_acc(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# masked_trajectories, col_ind, cost, row_ind = gpr.train_GPs_on_position_acceleration(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=True, print_cost_matrix=True, adjusting_output_len=False, output_max_length=5, plot_cost=False, deltat=0.1)
# print(f"Acceleration assignment: {row_ind} -> {col_ind}")
# print(f"Cost: {cost}")
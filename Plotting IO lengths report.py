import numpy as np
import matplotlib.pyplot as plt
import TrajectoryPlotter as TP
from time import time
import os
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# n = "MaxOutputLength2" ###CHECK LINE 19-21 ARE CONSISTENT WITH gp_3d.py!!!
# n = "Results9traj_9_TD"
# n = "Results16traj_20_TD"
# n = "Results16traj_1.32_TD - Max Output Length"
n = "Results16traj_20_TD"
local_folder = "not_coregionalised_no_switch"
not_coregionalised_list_of_successful_assignments = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\not_coregionalised_no_switch\\list_of_successful_assignments")
coregionalised_list_of_successful_assignments = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\coregionalised_no_switch\\list_of_successful_assignments")
acceleration_list_of_successful_assignments = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\list_of_successful_assignments")
acceleration_list_of_number_of_breakages = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\list_of_number_of_breakages")


independent_variable = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\independent_variable")
independent_variable_axis_title = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\independent_variable_axis_title")
list_of_approx_number_of_occlusions = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_approx_number_of_occlusions")
list_of_assignment_problems = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_assignment_problems")
list_of_number_of_breakages = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_number_of_breakages")
list_of_obscured_assignment_problems = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_obscured_assignment_problems")
list_of_successful_recombinations = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_successful_recombinations")
list_of_times = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_times")
list_of_total_occlusions = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_total_occlusions")
list_of_total_occlusions_assigned_correctly = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_total_occlusions_assigned_correctly")
list_of_total_partial_assignments = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_total_partial_assignments")
# independent_variable= TP.array_fractional_reducer(independent_variable,0.5, 0)

number_of_trajectories = np.shape(list_of_times)[0]
number_of_independent_variable = np.shape(list_of_times)[1]
list_of_acceleration_data_too_short = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\Number_of_times_not_long_enough_input")





trajectories = range(np.shape(list_of_total_occlusions_assigned_correctly)[0])
cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, number_of_trajectories)]
alpha = 0.2
alpha2 = 0.8
fontsize = 12
labelsize = 11
summing = False
Meaning = True
Alling = True


#DURATIONS OF INPUTS, OUTPUTS, INTERUPTIONS AND HISTOGRAMS OF NNDISTANCE
average_length_of_occlusion = list_of_total_occlusions/list_of_number_of_breakages
list_of_average_output_length = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_average_output_length")
list_of_average_input_length = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_average_input_length")
speeds = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\speeds")
NN_dist_array_not_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\NN_dist_lst_of_arrays")
NN_dist_array_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\coregionalised_no_switch\\NN_dist_lst_of_arrays")
NN_dist_array_acceleration = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\NN_dist_lst_of_arrays")

# average_length_of_occlusion= TP.array_fractional_reducer(average_length_of_occlusion,0.5, 1)
# list_of_average_output_length= TP.array_fractional_reducer(list_of_average_output_length,0.5, 1)
# list_of_average_input_length= TP.array_fractional_reducer(list_of_average_input_length,0.5, 1)
# list_of_number_of_breakages= TP.array_fractional_reducer(list_of_number_of_breakages,0.5, 1)
# acceleration_list_of_number_of_breakages= TP.array_fractional_reducer(acceleration_list_of_number_of_breakages,0.5, 1)


multiplier = 1.
fig = plt.figure(figsize=(6.25, 3.))
# outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.45, bottom=0.23)
outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.25, bottom=0.23)
left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
# fig1 = plt.figure(figsize=(5.5, 1.5))
# right_right_cell = gridspec.GridSpec(1, 1, figure=fig1, left=0.2, right=0.975, top=0.975, wspace=0.45, bottom=0.3)
fig1 = plt.figure(figsize=(3.1, 2.75))
right_right_cell = gridspec.GridSpec(1, 1, figure=fig1, left=0.2, right=0.975, top=0.975, wspace=0.45, bottom=0.23)
# gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[2])
right_cell = outer_grid[1].subgridspec(1, 1)
ax0 = fig.add_subplot(right_cell[:, :])
ax1 = fig1.add_subplot(right_right_cell[:, :])
ax2 = fig.add_subplot(left_cell[:,:])

n_bins = 50
bins = np.linspace(1.1, 1.8, n_bins)
hist = np.zeros((number_of_trajectories, n_bins-1))
for index4 in range(number_of_trajectories):
    ax1.hist(NN_dist_array_not_coregionalised[index4], bins=bins, histtype='step', fill=False, density=True, color=colors[index4], alpha=alpha)
ax1.hist(np.concatenate(NN_dist_array_not_coregionalised), bins=bins, histtype='step', fill=False, density=True, color='k', alpha=alpha2)
ax1.set_xlabel("Nearest Neighbour\nDistance (m)", fontsize=fontsize)
ax1.set_ylabel("Frequency\nDensity", fontsize=fontsize)

x = np.concatenate((list_of_average_output_length[:,:,None], list_of_average_input_length[:,:,None]), axis=2)
y = np.nanmean(np.concatenate((list_of_average_output_length[:,:,None], list_of_average_input_length[:,:,None]), axis=2), axis=2)
average_length_of_trajectory = np.nanmean(np.concatenate((list_of_average_output_length[:,:,None], list_of_average_input_length[:,:,None]), axis=2), axis=2)

alpha = 0.2


linewidth=0.5
elinewidth=1.
markersize = 5 * (72. / fig.dpi) ** 2
marker='.'
occ_marker = 'x'
acc_marker = '+'

ax0.errorbar(independent_variable-0.004, np.nanmean(average_length_of_trajectory, axis=0)/10, yerr=np.nanstd(x,axis=(0,2), ddof=1)/np.sqrt(len(trajectories))/10, linestyle='', color='k', marker=marker, alpha=alpha2, label="Output", linewidth=linewidth, elinewidth=elinewidth, markersize=markersize)
ax0.errorbar(independent_variable, np.nanmean(average_length_of_occlusion, axis=0)/10, yerr=np.nanstd(average_length_of_occlusion, axis=0, ddof=1)/np.sqrt(len(trajectories))/10, linestyle='', color='grey', marker=occ_marker, alpha=alpha2, label="Interruption", linewidth=linewidth, elinewidth=elinewidth, markersize=markersize)
for index4 in [1,12]:#range(number_of_trajectories):##### [12],[5],[6];[1,6,5,9,12];[1,6,12]
    ax0.errorbar(independent_variable, average_length_of_trajectory[index4]/10,
                linestyle='-', color=colors[index4], marker=marker,
                 alpha=alpha, linewidth=linewidth, elinewidth=elinewidth, markersize=markersize)
    ax0.errorbar(independent_variable, average_length_of_occlusion[index4]/10,
                linestyle='--', color=colors[index4], marker=occ_marker,
                 alpha=alpha, linewidth=linewidth, elinewidth=elinewidth, markersize=markersize)
ax2.errorbar(independent_variable-0.004, np.mean(list_of_number_of_breakages, axis=0), yerr=np.std(list_of_number_of_breakages,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker=marker, label="Disconnections", color='k', alpha=alpha2, linewidth=linewidth, elinewidth=elinewidth, markersize=markersize)
ax2.errorbar(independent_variable, np.mean(acceleration_list_of_number_of_breakages, axis=0), yerr=np.std(acceleration_list_of_number_of_breakages,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker=acc_marker, label="Disconnections", color='grey', alpha=alpha2, linewidth=linewidth, elinewidth=elinewidth, markersize=markersize)
for index4 in [1,12]:#trajectories:
    ax2.plot(independent_variable,
             list_of_number_of_breakages[index4], marker=marker,
             color=colors[index4], alpha=alpha, ls='-', linewidth=linewidth, markersize=markersize)
    ax2.plot(independent_variable,
             acceleration_list_of_number_of_breakages[index4], marker=acc_marker,
             color=colors[index4], alpha=alpha, ls='--', linewidth=linewidth, markersize=markersize)

ax0.set_xlabel("Threshold\nDistance (m)", fontsize=fontsize)
ax0.set_ylabel("Average Duration (s)", fontsize=fontsize)
ax0.xaxis.set_major_locator(MultipleLocator(0.1))
ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
ax2.xaxis.set_major_locator(MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
ax1.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax0.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax2.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)


# ax0.legend(loc="upper left")
ax2.set_xlabel("Threshold\nDistance (m)", fontsize=fontsize)
ax2.set_ylabel("Valid Disconnections", fontsize=fontsize)
plt.show()






plt.tight_layout()
# ax0.legend(loc=[0.52, 0.8])#, frameon=False)

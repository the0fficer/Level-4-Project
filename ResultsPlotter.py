import numpy as np
import matplotlib.pyplot as plt
import TrajectoryPlotter as TP
from time import time
import os
import matplotlib.gridspec as gridspec

n = "MaxOutputLength2" ###CHECK LINE 19-21 ARE CONSISTENT WITH gp_3d.py!!!
n = "16traj_12_TD - Max Output Length"
# n = "Results9traj_9_TD"
# local_folder = "not_coregionalised"
local_folder = "acceleration"
independent_variable = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\independent_variable")
independent_variable_axis_title = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\independent_variable_axis_title")
n = f"16traj_12_TD - Max Output Length\\{local_folder}"
# independent_variable = np.flip(np.linspace(1.2, 1.45, number_of_independent_variable))
# independent_variable = np.array([1,3,5,10,50])
# independent_variable = np.array([24,23,22,21,20,19,18])

# n = "Nice" ###CHECK LINE 19-21 ARE CONSISTENT WITH gp_3d.py!!!
# independent_variable_axis_title = "Threshold Distance (m)"
# independent_variable_axis_title = "Max Output Length"
# independent_variable_axis_title = "Frame rate (fps)"



list_of_approx_number_of_occlusions = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_approx_number_of_occlusions")
list_of_assignment_problems = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_assignment_problems")
list_of_number_of_breakages = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_number_of_breakages")
list_of_obscured_assignment_problems = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_obscured_assignment_problems")
list_of_successful_assignments = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_successful_assignments")
list_of_successful_recombinations = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_successful_recombinations")
list_of_times = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_times")
list_of_total_occlusions = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_total_occlusions")
list_of_total_occlusions_assigned_correctly = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_total_occlusions_assigned_correctly")
list_of_total_partial_assignments = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_total_partial_assignments")



number_of_trajectories = np.shape(list_of_times)[0]
number_of_independent_variable = np.shape(list_of_times)[1]

cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, number_of_trajectories)]
alpha = 0.1
alpha2 = 0.5
fontsize = 13
labelsize = 12

# fig = plt.figure()
# # for index4 in range(number_of_trajectories):
# #     plt.plot(independent_variable, 100*list_of_successful_assignments[index4]/list_of_number_of_breakages[index4], color=colors[index4], alpha=alpha)
# # plt.plot(independent_variable, 100*np.mean(list_of_successful_assignments, axis=0)/np.mean(list_of_number_of_breakages, axis=0), color='k', alpha=alpha2)
# plt.errorbar(independent_variable, np.mean(100*list_of_successful_assignments/list_of_number_of_breakages, axis=0), yerr=np.std(100*list_of_successful_assignments/list_of_number_of_breakages,axis=0), linestyle='', marker='x', label="Correct Assignments", color='g', alpha=alpha2)
# plt.xlabel(independent_variable_axis_title)
# plt.ylabel("Success Percentage")
# plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# plt.show()
#
# fig = plt.figure()
# for index4 in range(number_of_trajectories):
#     plt.plot(independent_variable, average_length_of_occlusion[index4], color=colors[index4], alpha=alpha)
# plt.plot(independent_variable, np.mean(average_length_of_occlusion, axis=0),  color='k', alpha=alpha2)
# plt.xlabel(independent_variable_axis_title)
# plt.ylabel("Average Length of Occlusion")
# plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# plt.show()
#

#

#
# fig = plt.figure()
# for index4 in range(number_of_trajectories):
#     # plt.plot(independent_variable, list_of_assignment_problems[index4], label="Solvable assignment problems", color=colors[index4], ls='-', alpha=alpha)
#     # plt.plot(independent_variable, list_of_total_partial_assignments[index4]+list_of_assignment_problems[index4], label="Assignment problems (inc. partially solvable)", color=colors[index4], ls='-.', alpha=alpha)
#     # plt.plot(independent_variable, list_of_successful_recombinations[index4], label="Correct Recombinations", color=colors[index4], ls=':', alpha=alpha)
#     # plt.plot(independent_variable, list_of_obscured_assignment_problems[index4], label="Obscured assignment problems", color=colors[index4], ls='--', alpha=alpha)
#     plt.plot(independent_variable, list_of_assignment_problems[index4], color=colors[index4], ls='-', alpha=alpha)
#     plt.plot(independent_variable, list_of_total_partial_assignments[index4] + list_of_assignment_problems[index4], color=colors[index4], ls='-.', alpha=alpha)
#     plt.plot(independent_variable, list_of_successful_recombinations[index4], color=colors[index4], ls=':', alpha=alpha)
#     plt.plot(independent_variable, list_of_obscured_assignment_problems[index4], color=colors[index4], ls='--', alpha=alpha)
#
# plt.plot(independent_variable, np.mean(list_of_assignment_problems, axis=0), label="Solvable assignment problems", color='k', ls='-', alpha=alpha2)
# plt.plot(independent_variable, np.mean(list_of_total_partial_assignments + list_of_assignment_problems, axis=0), label="Assignment problems (inc. partially solvable)", color='k', ls='-.', alpha=alpha2)
# plt.plot(independent_variable, np.mean(list_of_successful_recombinations, axis=0), label="Correct Recombinations", color='k', ls=':', alpha=alpha2)
# plt.plot(independent_variable, np.mean(list_of_obscured_assignment_problems, axis=0), label="Obscured assignment problems", color='k', ls='--', alpha=alpha2)
# plt.legend()
# plt.xlabel(independent_variable_axis_title)
# plt.ylabel("N")
# plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# plt.show()
#
# fig = plt.figure()
# for index4 in range(number_of_trajectories):
#     # plt.plot(independent_variable, list_of_approx_number_of_occlusions[index4], label="Approx Number of Occlusions", color=colors[index4], ls='-', alpha=alpha)
#     #plt.plot(independent_variable, list_of_total_occlusions[index4], label="Total Occlusions", color=colors[index4], ls='--', alpha=alpha)
#     #plt.plot(independent_variable, list_of_total_occlusions_assigned_correctly[index4], label="Correctly assigned occlusions", color=colors[index4], ls='-.', alpha=alpha)
#     plt.plot(independent_variable, list_of_total_occlusions[index4], color=colors[index4], ls='--', alpha=alpha)
#     plt.plot(independent_variable, list_of_total_occlusions_assigned_correctly[index4], color=colors[index4], ls='-.', alpha=alpha)
# plt.plot(independent_variable, np.mean(list_of_total_occlusions, axis=0), label="Total Occlusions", color='k',
#          ls='--', alpha=alpha2)
# plt.plot(independent_variable, np.mean(list_of_total_occlusions_assigned_correctly, axis=0),
#          label="Correctly assigned occlusions", color='k', ls='-.', alpha=alpha2)
# plt.legend()
# plt.xlabel(independent_variable_axis_title)
# plt.ylabel("N")
# plt.show()
#
#
#
# fig = plt.figure()
# plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#
# # for index4 in range(number_of_trajectories):
# #     plt.plot(independent_variable, list_of_average_output_length[index4], color=colors[index4], ls='--', alpha=alpha)
# #     plt.plot(independent_variable, list_of_average_input_length[index4], color=colors[index4], ls='-.', alpha=alpha)
# #     plt.plot(independent_variable, average_length_of_occlusion[index4], color=colors[index4], ls=':', alpha=alpha)
# # plt.plot(independent_variable, np.mean(list_of_average_output_length,axis=0), color='k', ls='--', alpha=alpha2, label="Output")
# # plt.plot(independent_variable, np.mean(list_of_average_input_length, axis=0), color='k', ls='-.', alpha=alpha2, label ="Input")
# # plt.plot(independent_variable, np.mean(average_length_of_occlusion, axis=0),  color='k', alpha=alpha2, label="Interruption Length")
# # for index4 in range(number_of_trajectories):
# #     plt.scatter(independent_variable, list_of_average_output_length[index4], color=colors[index4], marker='x', s=50*(72./fig.dpi)**2, alpha=alpha)
# #     plt.scatter(independent_variable, list_of_average_input_length[index4], color=colors[index4], marker='x', s=50*(72./fig.dpi)**2, alpha=alpha)
# #     plt.scatter(independent_variable, average_length_of_occlusion[index4], color=colors[index4],  marker='x', s=50*(72./fig.dpi)**2, alpha=alpha)
# plt.errorbar(independent_variable-0.002, np.mean(list_of_average_output_length,axis=0)/10, yerr=np.std(list_of_average_output_length,axis=0)/10, linestyle='', color='r', marker='x', alpha=alpha2, label="Output")
# plt.errorbar(independent_variable-0.002, np.mean(list_of_average_input_length, axis=0)/10, yerr=np.std(list_of_average_input_length, axis=0)/10, linestyle='', color='b', marker='x', alpha=alpha2, label ="Input")
# plt.errorbar(independent_variable, np.mean(average_length_of_occlusion, axis=0)/10, yerr=np.std(average_length_of_occlusion, axis=0)/10, linestyle='', color='g', marker='x', alpha=alpha2, label="Interruption")
# plt.xlabel(independent_variable_axis_title, fontsize=18)
# plt.ylabel("Average Duration (s)", fontsize=18)
# plt.legend(loc=[0.52,0.8])#, frameon=False)
# plt.show()
#
#
#

#TIMES OF COMPUTATION
fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(independent_variable, list_of_times[index4], color=colors[index4], alpha=alpha)
plt.plot(independent_variable, np.mean(list_of_times, axis=0), color='k', alpha=alpha2)
plt.xlabel(independent_variable_axis_title)
plt.ylabel("Time of Computation (s)")
plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
plt.show()



####################################################






# CONFIDENCE RATINGS FIGURE
list_of_success_cost_lists = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_success_costs")
list_of_success_cost_lists = [item for sublist in list_of_success_cost_lists for item in sublist]
list_of_failure_cost_lists = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_failure_costs")
list_of_failure_cost_lists = [item for sublist in list_of_failure_cost_lists for item in sublist]
fig = plt.figure()
plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# for index4 in range(number_of_trajectories):
#     n_bins = 10
#     s_n, s_bins, _ = plt.hist(list_of_success_cost_lists[index4], n_bins, histtype='step', fill=False, density=False, label="Correct Recombinations", color='g', alpha=alpha)
#     f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists[index4], n_bins, histtype='step', fill=False, density=False, label="Incorrect Recombinations", color='r', alpha=alpha)
n_bins = 15
s_n, s_bins, _ = plt.hist(list_of_success_cost_lists, n_bins, histtype='step', fill=False, density=True, label="Correct Recombinations", color='g', alpha=alpha2)
f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists, n_bins, histtype='step', fill=False, density=True, label="Incorrect Recombinations", color='r', alpha=alpha2)

C_over_9_s = [confidence for confidence in list_of_success_cost_lists if confidence>=0.9]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists if confidence>=0.9]
p_c_given_s = len(C_over_9_s)/len(list_of_success_cost_lists)
p_s = len(list_of_success_cost_lists)/(len(list_of_success_cost_lists)+len(list_of_failure_cost_lists))
p_c = (len(C_over_9_s)+len(C_over_9_sf))/(len(list_of_success_cost_lists)+len(list_of_failure_cost_lists))
print(f"P(S|C>=0.9) = {p_c_given_s*p_s/p_c}")
plt.legend(loc="upper left")
plt.ylabel("Frequency Density")
plt.xlabel("Confidence Rating")
plt.show()







#DURATIONS OF INPUTS, OUTPUTS, INTERUPTIONS AND HISTOGRAMS OF NNDISTANCE
average_length_of_occlusion = list_of_total_occlusions/list_of_number_of_breakages
list_of_average_output_length = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_average_output_length")
list_of_average_input_length = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\list_of_average_input_length")
speeds = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\speeds")
NN_dist_array = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results{n}\\NN_dist_lst_of_arrays")

fig = plt.figure(figsize=(12., 5.5))
outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3)
left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
right_cell = outer_grid[1].subgridspec(1, 1)
ax0 = fig.add_subplot(left_cell[:, :])
ax1 = fig.add_subplot(right_cell[:, :])

n_bins = 50
bins = np.linspace(1.1, 1.8, n_bins)
hist = np.zeros((number_of_trajectories, n_bins-1))
for index4 in range(number_of_trajectories):
    ax1.hist(NN_dist_array[index4], bins=bins, histtype='step', fill=False, density=True, color=colors[index4], alpha=alpha)
ax1.hist(np.concatenate(NN_dist_array), bins=bins, histtype='step', fill=False, density=True, color='k', alpha=alpha2)
ax1.set_xlabel("Nearest Neighbour Distance (m)", fontsize=fontsize)
ax1.set_ylabel("Frequency Density", fontsize=fontsize)
ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax0.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax0.errorbar(independent_variable-0.002, np.mean(list_of_average_output_length,axis=0)/10, yerr=np.std(list_of_average_output_length,axis=0)/10, linestyle='', color='r', marker='x', alpha=alpha2, label="Output")
ax0.errorbar(independent_variable-0.002, np.mean(list_of_average_input_length, axis=0)/10, yerr=np.std(list_of_average_input_length, axis=0)/10, linestyle='', color='b', marker='x', alpha=alpha2, label ="Input")
ax0.errorbar(independent_variable, np.mean(average_length_of_occlusion, axis=0)/10, yerr=np.std(average_length_of_occlusion, axis=0)/10, linestyle='', color='g', marker='x', alpha=alpha2, label="Interruption")
ax0.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
ax0.set_ylabel("Average Duration (s)", fontsize=fontsize)
ax0.legend(loc=[0.52, 0.8])#, frameon=False)







#TRAJECTORIES AND SUCCESS PERCENTAGE
plt.show()

fig = plt.figure(figsize=(10., 4.5))
outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3)

left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
right_cell = outer_grid[1].subgridspec(1, 1)

ax0 = fig.add_subplot(left_cell[:, :])
ax1 = fig.add_subplot(right_cell[:, :])

ax1.errorbar(independent_variable, np.mean(100*list_of_successful_assignments/list_of_number_of_breakages, axis=0), yerr=np.std(100*list_of_successful_assignments/list_of_number_of_breakages,axis=0), linestyle='', marker='x', label="Correct Assignments", color='k', alpha=alpha2)
ax1.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
ax1.set_ylabel("Success Percentage", fontsize=fontsize)

ax0.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)

ax0.errorbar(independent_variable-0.002, np.mean(list_of_number_of_breakages, axis=0), yerr=np.std(list_of_number_of_breakages,axis=0), linestyle='', marker='x', label="Disconnections", color='r', alpha=alpha2)
ax0.errorbar(independent_variable, np.mean(list_of_successful_assignments, axis=0), yerr=np.std(list_of_successful_assignments,axis=0), linestyle='', marker='x', label="Correct Assignments", color='g', alpha=alpha2)
ax0.legend(loc="upper left")
ax0.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
ax0.set_ylabel("N", fontsize=fontsize)
plt.show()





# NUMBER OF TRAJECTORIES REPAIRED.
# fig = plt.figure()
# # for index4 in range(number_of_trajectories):
# #     # plt.plot(independent_variable, list_of_number_of_breakages[index4], label="Broken Trajectories", color=colors[index4], ls='-', alpha=alpha)
# #     # plt.plot(independent_variable, list_of_successful_assignments[index4], label="Correct Reassignments", color=colors[index4], ls='-.', alpha=alpha)
# #     plt.plot(independent_variable, list_of_number_of_breakages[index4], # label="Broken Trajectories",
# #              color=colors[index4], ls='-', alpha=alpha)
# #     plt.plot(independent_variable, list_of_successful_assignments[index4], # label="Correct Reassignments",
# #              color=colors[index4], ls='-.', alpha=alpha)
# #
# # plt.plot(independent_variable, np.mean(list_of_number_of_breakages, axis=0), label="Broken Trajectories", color='k', ls='-', alpha=alpha2)
# # plt.plot(independent_variable, np.mean(list_of_successful_assignments, axis=0), label="Correct Reassignments", color='k', ls='-.', alpha=alpha2)
# plt.errorbar(independent_variable-0.002, np.mean(list_of_number_of_breakages, axis=0), yerr=np.std(list_of_number_of_breakages,axis=0), linestyle='', marker='x', label="Disconnections", color='r', alpha=alpha2)
# plt.errorbar(independent_variable, np.mean(list_of_successful_assignments, axis=0), yerr=np.std(list_of_successful_assignments,axis=0), linestyle='', marker='x', label="Correct Assignments", color='g', alpha=alpha2)
# plt.legend(loc="upper left")
# plt.xlabel(independent_variable_axis_title)
# plt.ylabel("N")
# plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# plt.show()
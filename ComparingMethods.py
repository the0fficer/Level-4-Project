import numpy as np
import matplotlib.pyplot as plt
import TrajectoryPlotter as TP
from time import time
import os
import matplotlib.gridspec as gridspec

# n = "MaxOutputLength2" ###CHECK LINE 19-21 ARE CONSISTENT WITH gp_3d.py!!!
# n = "Results9traj_9_TD"
# n = "Results16traj_12_TD"
# n = "Results16traj_1.32_TD - Max Output Length"
n = "Results16traj_20_TD - Max Output 3"
# n = "Results16traj_20_TD"
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
number_of_trajectories = np.shape(list_of_times)[0]
number_of_independent_variable = np.shape(list_of_times)[1]
list_of_acceleration_data_too_short = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\Number_of_times_not_long_enough_input")

cmap = plt.get_cmap('gist_rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, number_of_trajectories)]
alpha = 0.3
alpha1 = 0.1
alpha2 = 0.8
alpha3 = 0.5
fontsize = 12
labelsize = 10
summing = False
Meaning = False
Alling = True


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
list_of_success_cost_lists_not_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_success_costs")
list_of_success_cost_lists_not_coregionalised = [item for sublist in list_of_success_cost_lists_not_coregionalised for item in sublist]
list_of_failure_cost_lists_not_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_failure_costs")
list_of_failure_cost_lists_not_coregionalised = [item for sublist in list_of_failure_cost_lists_not_coregionalised for item in sublist]
list_of_success_cost_lists_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\coregionalised_no_switch\\list_of_success_costs")
list_of_success_cost_lists_coregionalised = [item for sublist in list_of_success_cost_lists_coregionalised for item in sublist]
list_of_failure_cost_lists_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\coregionalised_no_switch\\list_of_failure_costs")
list_of_failure_cost_lists_coregionalised = [item for sublist in list_of_failure_cost_lists_coregionalised for item in sublist]
list_of_success_cost_lists_acceleration = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\list_of_success_costs")
list_of_success_cost_lists_acceleration = [item for sublist in list_of_success_cost_lists_acceleration for item in sublist]
list_of_failure_cost_lists_acceleration = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\list_of_failure_costs")
list_of_failure_cost_lists_acceleration = [item for sublist in list_of_failure_cost_lists_acceleration for item in sublist]
fig = plt.figure()

plt.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
# for index4 in range(number_of_trajectories):
#     n_bins = 10
#     s_n, s_bins, _ = plt.hist(list_of_success_cost_lists_not_coregionalised[index4], n_bins, histtype='step', fill=False, density=False, label="Correct Recombinations", color='g', alpha=alpha)
#     f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists_not_coregionalised[index4], n_bins, histtype='step', fill=False, density=False, label="Incorrect Recombinations", color='r', alpha=alpha)
n_bins = 20
bins = np.linspace(0,1,n_bins)
s_n, s_bins, _ = plt.hist(list_of_success_cost_lists_not_coregionalised, bins=bins, histtype='step', fill=False, density=False, label="Correct - not coregionalised", color='g', alpha=alpha2, linestyle="--")
f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists_not_coregionalised, bins=bins, histtype='step', fill=False, density=False, label="Incorrect - not coregionalised", color='r', alpha=alpha2, linestyle="--")
s_n, s_bins, _ = plt.hist(list_of_success_cost_lists_coregionalised, bins=bins, histtype='step', fill=False, density=False, label="Correct - coregionalised", color='g', alpha=alpha2, linestyle=":")
f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists_coregionalised, bins=bins, histtype='step', fill=False, density=False, label="Incorrect - coregionalised", color='r', alpha=alpha2, linestyle=":")
s_n, s_bins, _ = plt.hist(list_of_success_cost_lists_acceleration, bins=bins, histtype='step', fill=False, density=False, label="Correct - acceleration", color='g', alpha=alpha2, linestyle="-.")
f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists_acceleration, bins=bins, histtype='step', fill=False, density=False, label="Incorrect - acceleration", color='r', alpha=alpha2, linestyle="-.")
plt.yscale('log')
confidence_threshold=0.95
C_over_9_s = [confidence for confidence in list_of_success_cost_lists_not_coregionalised if confidence>=confidence_threshold]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists_not_coregionalised if confidence>=confidence_threshold]
p_c_given_s = len(C_over_9_s)/len(list_of_success_cost_lists_not_coregionalised)
p_s = len(list_of_success_cost_lists_not_coregionalised)/(len(list_of_success_cost_lists_not_coregionalised)+len(list_of_failure_cost_lists_not_coregionalised))
p_c = (len(C_over_9_s)+len(C_over_9_sf))/(len(list_of_success_cost_lists_not_coregionalised)+len(list_of_failure_cost_lists_not_coregionalised))
print(f"Not coregionalised - P(S|C>{str(confidence_threshold)}) = {p_c_given_s*p_s/p_c}")
C_less_than_9_s = [confidence for confidence in list_of_success_cost_lists_not_coregionalised if confidence<=confidence_threshold]
C_less_than_9_sf = [confidence for confidence in list_of_failure_cost_lists_not_coregionalised if confidence<=confidence_threshold]
p_c_given_s = len(C_less_than_9_s)/len(list_of_success_cost_lists_not_coregionalised)
p_s = len(list_of_success_cost_lists_not_coregionalised)/(len(list_of_success_cost_lists_not_coregionalised)+len(list_of_failure_cost_lists_not_coregionalised))
p_c = (len(C_less_than_9_s)+len(C_less_than_9_sf))/(len(list_of_success_cost_lists_not_coregionalised)+len(list_of_failure_cost_lists_not_coregionalised))
print(f"Not coregionalised - P(S|C<{str(confidence_threshold)}) = {p_c_given_s*p_s/p_c}")
plt.legend(loc="upper left")
plt.ylabel("Frequency Density")
plt.xlabel("Confidence Rating")
plt.show()
C_over_9_s = [confidence for confidence in list_of_success_cost_lists_coregionalised if confidence>=confidence_threshold]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists_coregionalised if confidence>=confidence_threshold]
p_c_given_s = len(C_over_9_s)/len(list_of_success_cost_lists_coregionalised)
p_s = len(list_of_success_cost_lists_coregionalised)/(len(list_of_success_cost_lists_coregionalised)+len(list_of_failure_cost_lists_coregionalised))
p_c = (len(C_over_9_s)+len(C_over_9_sf))/(len(list_of_success_cost_lists_coregionalised)+len(list_of_failure_cost_lists_coregionalised))
print(f"Coregionalised - P(S|C>{str(confidence_threshold)}) = {p_c_given_s*p_s/p_c}")
C_less_than_9_s = [confidence for confidence in list_of_success_cost_lists_coregionalised if confidence<=confidence_threshold]
C_less_than_9_sf = [confidence for confidence in list_of_failure_cost_lists_coregionalised if confidence<=confidence_threshold]
p_c_given_s = len(C_less_than_9_s)/len(list_of_success_cost_lists_coregionalised)
p_s = len(list_of_success_cost_lists_coregionalised)/(len(list_of_success_cost_lists_coregionalised)+len(list_of_failure_cost_lists_coregionalised))
p_c = (len(C_less_than_9_s)+len(C_less_than_9_sf))/(len(list_of_success_cost_lists_coregionalised)+len(list_of_failure_cost_lists_coregionalised))
print(f"Coregionalised - P(S|C<{str(confidence_threshold)}) = {p_c_given_s*p_s/p_c}")
C_over_9_s = [confidence for confidence in list_of_success_cost_lists_acceleration if confidence>=confidence_threshold]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists_acceleration if confidence>=confidence_threshold]
p_c_given_s = len(C_over_9_s)/len(list_of_success_cost_lists_acceleration)
p_s = len(list_of_success_cost_lists_acceleration)/(len(list_of_success_cost_lists_acceleration)+len(list_of_failure_cost_lists_acceleration))
p_c = (len(C_over_9_s)+len(C_over_9_sf))/(len(list_of_success_cost_lists_acceleration)+len(list_of_failure_cost_lists_acceleration))
print(f"Acceleration - P(S|C>{str(confidence_threshold)}) = {p_c_given_s*p_s/p_c}")
C_less_than_9_s = [confidence for confidence in list_of_success_cost_lists_acceleration if confidence<=confidence_threshold]
C_less_than_9_sf = [confidence for confidence in list_of_failure_cost_lists_acceleration if confidence<=confidence_threshold]
p_c_given_s = len(C_less_than_9_s)/len(list_of_success_cost_lists_acceleration)
p_s = len(list_of_success_cost_lists_acceleration)/(len(list_of_success_cost_lists_acceleration)+len(list_of_failure_cost_lists_acceleration))
p_c = (len(C_less_than_9_s)+len(C_less_than_9_sf))/(len(list_of_success_cost_lists_acceleration)+len(list_of_failure_cost_lists_acceleration))
print(f"Acceleration - P(S|C<{str(confidence_threshold)}) = {p_c_given_s*p_s/p_c}")






#DURATIONS OF INPUTS, OUTPUTS, INTERUPTIONS AND HISTOGRAMS OF NNDISTANCE
average_length_of_occlusion = list_of_total_occlusions/list_of_number_of_breakages
list_of_average_output_length = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_average_output_length")
list_of_average_input_length = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\list_of_average_input_length")
speeds = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\speeds")
NN_dist_array_not_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\{local_folder}\\NN_dist_lst_of_arrays")
NN_dist_array_coregionalised = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\coregionalised_no_switch\\NN_dist_lst_of_arrays")
NN_dist_array_acceleration = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\NN_dist_lst_of_arrays")

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
    ax1.hist(NN_dist_array_not_coregionalised[index4], bins=bins, histtype='step', fill=False, density=True, color=colors[index4], alpha=alpha)
ax1.hist(np.concatenate(NN_dist_array_not_coregionalised), bins=bins, histtype='step', fill=False, density=True, color='k', alpha=alpha2)
ax1.set_xlabel("Nearest Neighbour Distance (m)", fontsize=fontsize)
ax1.set_ylabel("Frequency Density", fontsize=fontsize)
ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax0.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
if Meaning or summing:
    ax0.errorbar(independent_variable-0.002, np.mean(list_of_average_output_length,axis=0)/10, yerr=np.std(list_of_average_output_length,axis=0)/10, linestyle='', color='r', marker='x', alpha=alpha2, label="Output")
    ax0.errorbar(independent_variable-0.002, np.mean(list_of_average_input_length, axis=0)/10, yerr=np.std(list_of_average_input_length, axis=0)/10, linestyle='', color='b', marker='x', alpha=alpha2, label ="Input")
    ax0.errorbar(independent_variable, np.mean(average_length_of_occlusion, axis=0)/10, yerr=np.std(average_length_of_occlusion, axis=0)/10, linestyle='', color='g', marker='x', alpha=alpha2, label="Interruption")
if Alling:
    for index4 in range(number_of_trajectories):
        ax0.errorbar(independent_variable, list_of_average_output_length[index4] / 10,
                    linestyle='--', color=colors[index4], marker='1',
                     alpha=alpha)
        ax0.errorbar(independent_variable, list_of_average_input_length[index4] / 10,
                    linestyle='-', color=colors[index4], marker='+',
                     alpha=alpha)
        ax0.errorbar(independent_variable, average_length_of_occlusion[index4] / 10,
                    linestyle=':', color=colors[index4], marker='o',
                     alpha=alpha)
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

curved_trajectories = [0, 4, 5, 12]
both = [7, 13]
straight_trajectories = [1, 2, 3, 6, 8, 9, 10, 11, 14, 15]
trajectories = range(number_of_trajectories)
# trajectories = curved_trajectories


summing = False
Meaning = False
Alling = True


if Alling:
    for index4 in trajectories:
        plt.plot(independent_variable,
                 100*coregionalised_list_of_successful_assignments[index4]/list_of_number_of_breakages[index4],
                 color=colors[index4], alpha=alpha, ls=':')
        plt.plot(independent_variable,
                 100 * not_coregionalised_list_of_successful_assignments[index4] / list_of_number_of_breakages[index4],
                 color=colors[index4], alpha=alpha, ls='-.')
        plt.plot(independent_variable,
                 100 *acceleration_list_of_successful_assignments[index4]/acceleration_list_of_number_of_breakages[index4],
                 color=colors[index4], alpha=alpha, ls='--')
if Meaning:

    ax1.errorbar(independent_variable, np.nanmean(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories], axis=0), yerr=np.nanstd(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle=':', marker='x', label="Coregionalised Method", color='k', alpha=alpha3)
    ax1.errorbar(independent_variable-0.002, np.nanmean(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories], axis=0), yerr=np.nanstd(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='-.', marker='x', label="Non-coregionalised Method", color='b', alpha=alpha3)
    ax1.errorbar(independent_variable+0.002, np.nanmean(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories], axis=0), yerr=np.nanstd(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='--', marker='x', label="Acceleration Method", color='green', alpha=alpha3)

if summing:
    ax1.errorbar(independent_variable, 100*np.nansum(coregionalised_list_of_successful_assignments[trajectories], axis=0)/np.nansum(list_of_number_of_breakages[trajectories], axis=0), linestyle=':', marker='x', label="Coregionalised Method", color='k', alpha=alpha2)
    ax1.errorbar(independent_variable-0.002, 100*np.nansum(not_coregionalised_list_of_successful_assignments[trajectories], axis=0)/np.nansum(list_of_number_of_breakages[trajectories], axis=0), linestyle='-.', marker='x', label="Non-coregionalised Method", color='b', alpha=alpha2)
    ax1.errorbar(independent_variable+0.002, 100*np.nansum(acceleration_list_of_successful_assignments[trajectories], axis=0)/np.nansum(acceleration_list_of_number_of_breakages[trajectories], axis=0), linestyle='--', marker='x', label="Acceleration Method", color='green', alpha=alpha2)
ax1.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
ax1.set_ylabel("Success Percentage", fontsize=fontsize)

ax0.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)


summing = False
Meaning = False
Alling = True



ax0.errorbar(independent_variable-0.002, np.mean(list_of_number_of_breakages, axis=0), yerr=np.std(list_of_number_of_breakages,axis=0), linestyle='', marker='x', label="Disconnections", color='k', alpha=alpha2)
ax0.errorbar(independent_variable, np.mean(acceleration_list_of_number_of_breakages, axis=0), yerr=np.std(acceleration_list_of_number_of_breakages,axis=0), linestyle='', marker='x', label="Disconnections", color='grey', alpha=alpha2)
# ax0.errorbar(independent_variable+0.002, np.mean(list_of_acceleration_data_too_short, axis=0), yerr=np.std(list_of_acceleration_data_too_short,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Not long enough", color='pink', alpha=alpha2)
# ax0.errorbar(independent_variable-0.002, np.mean(coregionalised_list_of_successful_assignments, axis=0), yerr=np.std(coregionalised_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle=':', marker='x', label="Correct Assignments-Coregionalised", color='k', alpha=alpha2)
# ax0.errorbar(independent_variable, np.mean(not_coregionalised_list_of_successful_assignments, axis=0), yerr=np.std(not_coregionalised_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='-.', marker='x', label="Correct Assignments-Not coregionalised", color='b', alpha=alpha2)
# ax0.errorbar(independent_variable+0.002, np.mean(acceleration_list_of_successful_assignments, axis=0), yerr=np.std(acceleration_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='--', marker='x', label="Correct Assignments-Acceleration", color='green', alpha=alpha2)
#####
# if summing:
#     ax0.errorbar(independent_variable-0.002, np.sum(list_of_number_of_breakages, axis=0),  linestyle='-', marker='x', label="Disconnections", color='k', alpha=alpha2)
#     ax0.errorbar(independent_variable, np.sum(acceleration_list_of_number_of_breakages, axis=0),  linestyle='-', marker='x', label="Disconnections", color='grey', alpha=alpha2)
#     ax0.errorbar(independent_variable+0.002, np.sum(list_of_acceleration_data_too_short, axis=0), linestyle='-', marker='x', label="Not long enough", color='pink', alpha=alpha2)
#     ax0.errorbar(independent_variable-0.002, np.sum(coregionalised_list_of_successful_assignments, axis=0),  linestyle=':', marker='x', label="Correct Assignments-Coregionalised", color='k', alpha=alpha2)
#     ax0.errorbar(independent_variable, np.sum(not_coregionalised_list_of_successful_assignments, axis=0),  linestyle='-.', marker='x', label="Correct Assignments-Not coregionalised", color='b', alpha=alpha2)
#     ax0.errorbar(independent_variable+0.002, np.sum(acceleration_list_of_successful_assignments, axis=0),  linestyle='--', marker='x', label="Correct Assignments-Acceleration", color='green', alpha=alpha2)
# if Alling:
for index4 in [1,12]:#trajectories:
    # ax0.plot(independent_variable,
    #          coregionalised_list_of_successful_assignments[index4],
    #          color=colors[index4], alpha=alpha, ls=':')
    # ax0.plot(independent_variable,
    #          not_coregionalised_list_of_successful_assignments[index4],
    #          color=colors[index4], alpha=alpha, ls='-.')
    # ax0.plot(independent_variable,
    #          acceleration_list_of_successful_assignments[index4],
    #          color=colors[index4], alpha=alpha, ls='--')
    ax0.plot(independent_variable,
             list_of_number_of_breakages[index4],
             color=colors[index4], alpha=alpha, ls='-')
    ax0.plot(independent_variable,
             acceleration_list_of_number_of_breakages[index4],
             color=colors[index4], alpha=alpha, ls='-.')
# ax0.legend(loc="upper left")
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
def str_round(line, decimals=1):
    if np.isnan(line):
        return "-"
    else:
        return(str(np.around(line, decimals=decimals)))


print("Method A")
print(" \\\\\n".join([" & ".join(map(str_round,line)) for line in 100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories]]))
print('')
print('')
print("Method B")
print(" \\\\\n".join([" & ".join(map(str_round,line)) for line in 100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories]]))
print('')
print('')
print("Method C")
print(" \\\\\n".join([" & ".join(map(str_round,line)) for line in 100*acceleration_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories]]))
print('')
print('')

a = 100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories]
b = 100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories]
c = 100*acceleration_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories]
arrays = [a, b, c]
shape = (len(arrays)*a.shape[0], a.shape[1])
interleaved_array = np.hstack(arrays).reshape(shape)
# interleaved_array = np.where(interleaved_array == np.nan, "-", interleaved_array)

# print(interleaved_array)
d = " \\\\\n".join([" & ".join(map(str_round,line)) for line in interleaved_array])
# d.replace("nan", "-")
print(d)
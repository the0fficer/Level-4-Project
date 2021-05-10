import numpy as np
import matplotlib.pyplot as plt
import TrajectoryPlotter as TP
from time import time
import os
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# n = "MaxOutputLength2" ###CHECK LINE 19-21 ARE CONSISTENT WITH gp_3d.py!!!
# n = "Results9traj_9_TD"
# n = "Results16traj_12_TD"
# n = "Results16traj_1.32_TD - Max Output Length"
n = "Results16traj_12_TD - Max Output 3"
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
alpha = 0.1
alpha2 = 0.5
fontsize = 13
labelsize = 12
summing = False
Meaning = False
Alling = True


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

fig = plt.figure(figsize=(6.25, 2.3))
outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=0.08, right=0.99, top=0.99, bottom=0.2, hspace=0.0)
outer_grid = outer_grid[0, :18].subgridspec(1, 3, hspace=0.0, wspace=0.03)#3)
ax1 = fig.add_subplot(outer_grid[0, 0])
plt.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
ax2 = fig.add_subplot(outer_grid[0, 1], sharex=ax1, sharey=ax1)
plt.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
ax3 = fig.add_subplot(outer_grid[0, 2], sharex=ax1, sharey=ax1)
plt.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)

# for index4 in range(number_of_trajectories):
#     n_bins = 10
#     s_n, s_bins, _ = plt.hist(list_of_success_cost_lists_not_coregionalised[index4], n_bins, histtype='step', fill=False, density=False, label="Correct Recombinations", color='g', alpha=alpha)
#     f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists_not_coregionalised[index4], n_bins, histtype='step', fill=False, density=False, label="Incorrect Recombinations", color='r', alpha=alpha)
n_bins = 20
bins = np.linspace(0,1,n_bins+1)
# i = 17
# list_of_success_cost_lists_not_coregionalised = list_of_success_cost_lists_not_coregionalised[i]
# list_of_failure_cost_lists_not_coregionalised = list_of_failure_cost_lists_not_coregionalised[i]
# list_of_success_cost_lists_coregionalised = list_of_success_cost_lists_coregionalised[i]
# list_of_failure_cost_lists_coregionalised = list_of_failure_cost_lists_coregionalised[i]
# list_of_success_cost_lists_acceleration = list_of_success_cost_lists_acceleration[i]
# list_of_failure_cost_lists_acceleration = list_of_failure_cost_lists_acceleration[i]

# s_n, s_bins, _ = ax1.hist(list_of_success_cost_lists_not_coregionalised[i], bins=bins, histtype='step', fill=True,
#                           density=False, label="Correct - not coregionalised", color='g',
#                           alpha=alpha2)  # , linestyle="--")
# f_n, f_bins, _ = ax1.hist(list_of_failure_cost_lists_not_coregionalised[i], bins=bins, histtype='step', fill=True,
#                           density=False, label="Incorrect - not coregionalised", color='r',
#                           alpha=alpha2)  # , linestyle="--")
# s_n, s_bins, _ = ax2.hist(list_of_success_cost_lists_coregionalised[i], bins=bins, histtype='step', fill=True,
#                           density=False, label="Correct - coregionalised", color='g',
#                           alpha=alpha2)  # , linestyle=":")
# f_n, f_bins, _ = ax2.hist(list_of_failure_cost_lists_coregionalised[i], bins=bins, histtype='step', fill=True,
#                           density=False, label="Incorrect - coregionalised", color='r',
#                           alpha=alpha2)  # , linestyle=":")
# s_n, s_bins, _ = ax3.hist(list_of_success_cost_lists_acceleration[i], bins=bins, histtype='step', fill=True,
#                           density=False, label="Correct - acceleration", color='g',
#                           alpha=alpha2)  # , linestyle="-.")
# f_n, f_bins, _ = ax3.hist(list_of_failure_cost_lists_acceleration[i], bins=bins, histtype='step', fill=True,
#                           density=False, label="Incorrect - acceleration", color='r',
#                           alpha=alpha2)  # , linestyle="-.")

s_n, s_bins, _ = ax1.hist(list_of_success_cost_lists_not_coregionalised, bins=bins, histtype='step', fill=True, density=False, label="Correct - not coregionalised", color='g', alpha=alpha2)# , linestyle="--")
f_n, f_bins, _ = ax1.hist(list_of_failure_cost_lists_not_coregionalised, bins=bins, histtype='step', fill=True, density=False, label="Incorrect - not coregionalised", color='r', alpha=alpha2)# , linestyle="--")
s_n, s_bins, _ = ax2.hist(list_of_success_cost_lists_coregionalised, bins=bins, histtype='step', fill=True, density=False, label="Correct - coregionalised", color='g', alpha=alpha2)# , linestyle=":")
f_n, f_bins, _ = ax2.hist(list_of_failure_cost_lists_coregionalised, bins=bins, histtype='step', fill=True, density=False, label="Incorrect - coregionalised", color='r', alpha=alpha2)# , linestyle=":")
s_n, s_bins, _ = ax3.hist(list_of_success_cost_lists_acceleration, bins=bins, histtype='step', fill=True, density=False, label="Correct - acceleration", color='g', alpha=alpha2)# , linestyle="-.")
f_n, f_bins, _ = ax3.hist(list_of_failure_cost_lists_acceleration, bins=bins, histtype='step', fill=True, density=False, label="Incorrect - acceleration", color='r', alpha=alpha2)# , linestyle="-.")

plt.yscale('log')
# plt.legend(loc="upper left")
ax1.set_ylabel("Frequency")
ax1.set_xlabel("C")
ax2.set_xlabel("C")
ax3.set_xlabel("C")
ax2.tick_params(axis="y", direction="in", bottom=True, top=True, left=True, right=True, labelleft=False)
ax3.tick_params(axis="y", direction="in", bottom=True, top=True, left=True, right=True, labelleft=False)
# ax1.xaxis.set_major_locator(MultipleLocator(0.25))
# ax2.xaxis.set_major_locator(MultipleLocator(0.25))
# ax3.xaxis.set_major_locator(MultipleLocator(0.25))
ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
ax3.xaxis.set_minor_locator(AutoMinorLocator(5))
plt.show()



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

print('')
print('')
print('')

confidence_threshold=0.5
C_over_9_s = [confidence for confidence in list_of_success_cost_lists_not_coregionalised if confidence>=confidence_threshold and confidence<=0.95]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists_not_coregionalised if confidence>=confidence_threshold and confidence<=0.95]
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
C_over_9_s = [confidence for confidence in list_of_success_cost_lists_coregionalised if confidence>=confidence_threshold and confidence<=0.95]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists_coregionalised if confidence>=confidence_threshold and confidence<=0.95]
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
C_over_9_s = [confidence for confidence in list_of_success_cost_lists_acceleration if confidence>=confidence_threshold and confidence<=0.95]
C_over_9_sf = [confidence for confidence in list_of_failure_cost_lists_acceleration if confidence>=confidence_threshold and confidence<=0.95]
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




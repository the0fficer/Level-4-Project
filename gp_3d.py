import numpy as np
import matplotlib.pyplot as plt
import GPy
import gp_reconstruction as gpr
from time import time
import os
import TrajectoryPlotter as TP
np.random.seed(0)

total_time = time()


trajectories = gpr.array_unpacker("trajectories_as_arrays2/2birds1000timesteps20201126-111657")
# gpr.trajectory_plotter(trajectories)
# X, gp1 = gpr.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=3, fraction=0.1)
# X, gp2 = gpr.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=6, fraction=0.1)
# X, gp3 = gpr.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=9, fraction=0.1)
# # trajectories = gpr.array_fractional_reducer(trajectories, 0.1, 2)
# # X, gp = gpr.three_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=.1, length=10.)
# newX = np.linspace(0.,11.,100)
# gpr.prediction_plotter(newX,gp1)
# gpr.prediction_plotter(newX,gp2)
# gpr.prediction_plotter(newX,gp3)
#########################################

# t = np.linspace(0,10,100)
# xyz = np.zeros((3, np.shape(t)[0]))
# xyz[0, :] = np.cos(t)
# xyz[1, :] = np.sin(t)
# gpr.trajectory_plotter(xyz[None, :, :])
# # T, gp2 = gpr.three_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10.)
# # newT = np.linspace(0.,25.,100)
# # gpr.prediction_plotter(newT,gp2)
# newT = np.linspace(0.,25.,100)
# T, gp4 = gpr.multi_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10., n_dimensions=3, fraction=1.)
# gpr.prediction_plotter(newT,gp4)
# T, gp5 = gpr.multi_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10., n_dimensions=6, fraction=1.)
# gpr.prediction_plotter(newT,gp5)
# T, gp6 = gpr.multi_dimensional_gaussian_plotter(xyz, extension_ratio=1.5, length=10., n_dimensions=9, fraction=1.)
# gpr.prediction_plotter(newT,gp6)


######################################
#X, gp1 = gpr.multi_dimensional_gaussian_plotter(trajectories[0, :, :], extension_ratio=0.1, length=10., n_dimensions=3, fraction=0.1)
#newX = np.linspace(0.,11.,100)
#gpr.prediction_plotter(newX,gp1)

# fraction = 0.1
# trajectories = trajectories[:, :, :650]
# trajectories = gpr.array_fractional_reducer(trajectories, fraction, 2)
# # trajectories = np.flip(trajectories, axis=2)
# gpr.report_graph_plotter(trajectories[0, :, :],trajectories[1, :, :], int(fraction*500), int(fraction*50), fraction=fraction, length=6.5)


# X = np.arange(1,5)
# X = X[:,None]
# X_list = [X,X]
# Y = np.arange(1,5)
# Y1 = Y[:,None]
# Y2 = Y[:+5,None]
# Y_list = [Y1,Y2]
# print(gpr.build_XY(X_list,Y_list))

##################
# times = np.linspace(0.,10.,1000)
# assert np.shape(trajectories)[-1] == np.shape(times)[-1]
# fraction = 0.1
# trajectories = gpr.array_fractional_reducer(trajectories, fraction, 2)
# times = gpr.array_fractional_reducer(times, fraction, 0)
# input_trajectory0, output_trajectory0 = gpr.trajectory_masker(trajectories[0,:,:],  int(650*fraction), int(50*fraction))
# input_trajectory1, output_trajectory1 = gpr.trajectory_masker(trajectories[1,:,:],  int(650*fraction), int(50*fraction))
# # input_trajectory2, output_trajectory2 = gpr.trajectory_masker(trajectories[2,:,:],  int(750*fraction), int(50*fraction))
# # input_trajectory3, output_trajectory3 = gpr.trajectory_masker(trajectories[3,:,:],  int(750*fraction), int(50*fraction))
# # input_trajectory4, output_trajectory4 = gpr.trajectory_masker(trajectories[4,:,:],  int(750*fraction), int(50*fraction))
# input_trajectory_list = [input_trajectory0, input_trajectory1 ]# , input_trajectory2]# , input_trajectory3]#, input_trajectory4]
# output_trajectory_list = [output_trajectory0, output_trajectory1]# , output_trajectory2]# , output_trajectory3]#, output_trajectory4]
#
# # gpr.train_GPs_on_position(input_trajectory_list, output_trajectory_list, times, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# # gpr.train_GPs_on_position_1D(input_trajectory_list, output_trajectory_list, times, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# masked_trajectories, col_ind, cost = gpr.train_GPs_on_position_acceleration(input_trajectory_list, output_trajectory_list, times, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=True)
# print(f"Assignment:{col_ind}")
# print(f"Cost: {cost}")
# gpr.train_GPs_on_position(input_trajectory_list,output_trajectory_list,times)
#################
plot_cost=False
from_scratch = False
adjusting_output_len = True
coregionalised = False
acceleration=True
swapping_IO=False
if swapping_IO:
    swap = "_switch"
else:
    swap = "_no_switch"
if acceleration:
    assert(swapping_IO==False)
# n = "9traj_9_TD"
# local_folder = "not_coregionalised"
# coregionalised = False
# n = "Results9traj_9_TD"
# local_folder = "not_coregionalised"
# n="Results16traj_1.32_TD - Max Output Length"
# n="Results16traj_12_TD - Max Output 3"
# n = "Results16traj_20_TD - Max Output 3"
# n="Throwaway"
# n = "Results16traj_12_TD"
n="Results16traj_1.35_TD - Max Output Length"
if acceleration:
    local_folder = f"acceleration{swap}"
elif coregionalised:
    local_folder = f"coregionalised{swap}"
else:
    local_folder = f"not_coregionalised{swap}"
folder_with_trajectories = f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\trajectories_for_analysing"
# number_of_trajectories = len(os.listdir(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_for_analysing"))
number_of_trajectories = len(os.listdir(folder_with_trajectories))

if from_scratch:
    frame_rates = np.array([24, 23, 22, 21, 20, 19, 18])
    fraction, threshold_distances = gpr.framerate_to_fraction_and_threshold_distance(frame_rates, delta_t=0.01, max_speed=13.5, )
    number_of_threshold_distances = len(threshold_distances)
    independent_variable = frame_rates
    independent_variable_axis_title = "Frame rate (fps)"
elif adjusting_output_len:
    output_max_length = np.array([1,3,5,10, 15, 25, 50])
    threshold_distances = 1.32*np.ones_like(output_max_length)
    fraction = 0.1*np.ones_like(threshold_distances)
    independent_variable = output_max_length
    independent_variable_axis_title = "Number of positions in output"
else:
    number_of_threshold_distances = 20
    # threshold_distances = np.flip(np.array([1.225,1.25,1.275,1.3, 1.325,1.35,1.375,1.4,1.45,1.5]))
    # threshold_distances = np.flip(np.array([1.3]))
    threshold_distances = np.flip(np.linspace(1.175, 1.45, number_of_threshold_distances)) # was from 1.75 to 1.375
    output_max_length = 3*np.ones_like(threshold_distances, dtype=int)
    fraction = 0.1*np.ones_like(threshold_distances)
    independent_variable = threshold_distances
    independent_variable_axis_title = "Threshold Distance (m)"
    adjusting_output_len = True

# independent_variable_axis_title = "Number of positions in output"
TP.array_save(independent_variable, f"independent_variable", f"{n}")
TP.array_save(independent_variable_axis_title, f"independent_variable_axis_title", f"{n}")
list_of_total_occlusions = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_total_occlusions_assigned_correctly = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_approx_number_of_occlusions = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_obscured_assignment_problems = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_assignment_problems = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_successful_recombinations = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_successful_assignments = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_number_of_breakages = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_total_partial_assignments = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_times = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_number_of_broken_trajectories = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_average_input_length = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_average_output_length = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_acceleration_data_too_short = np.zeros((number_of_trajectories, len(independent_variable)))
list_of_list_of_assignments = []
list_of_success_cost_lists = []
list_of_failure_cost_lists = []
NN_dist_lst_of_arrays = []
speeds = []

# for index, local_path in enumerate(os.listdir(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_for_analysing")):
for index, local_path in enumerate(os.listdir(folder_with_trajectories)):
    print("CHECK!! Using absolute path in line 43 so check that the path hasn't changed since last use.")
    absolute_path = f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\trajectories_for_analysing\\{local_path}"
    trajectories = gpr.array_unpacker(absolute_path)
    # trajectories = gpr.array_unpacker(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\trajectories_as_arrays2\10birds1000timesteps20201126-111656")
    # gpr.trajectory_plotter(trajectories)
    #####
    NN_sq_dist_array = TP.nearest_neighbour_sq_dist(trajectories)
    NN_dist_lst_of_arrays.append(NN_sq_dist_array.flatten()**0.5)
    velocities = TP.differentiater(trajectories, time_interval=0.01)
    speeds.append(np.sum(velocities ** 2, axis=1).flatten() ** 0.5)
    #####
    # TP.NN_and_speeds_histogram_plotter(NN_dist_lst_of_arrays[index], speeds[index])
    times_array = np.linspace(0, 5., np.shape(trajectories)[2])

    for index3 in range(len(independent_variable)):
        # broken_trajectories, trajectories_altered, trajectories_altered_between_times = gpr.break_trajectories(trajectories,
        #                                                                                                        threshold_distance)
        # list_of_number_of_broken_trajectories[index, index3] = len(trajectories_altered_between_times))
        ################
        print(f"\nFolder: {local_path}")
        print(f"Independent Variable: {independent_variable[index3]}")
        intermediate_time = time()
        average_number_of_input_positions, average_number_of_output_positions, list_of_assignments, total_partial_assignments, total_assignments, successful_combination, successful_assignments, number_of_breakages, obscured_assignment_problems, approx_number_of_occlusions, total_obscurations, total_obscurations_assigned_correctly, duration, costs_for_successes, costs_for_failures, acceleration_data_too_short  = gpr.gp_reconstructor(
            trajectories, threshold_distances[index3], from_scratch=False, fraction=fraction[index3], length=5., n_restarts=3, verbose=False,
            lengthscales=False, print_cost_matrix=False, plot_trajectories=False, adjusting_output_len=adjusting_output_len,
            output_max_length=output_max_length[index3], swapping_IO=swapping_IO, plot_cost=plot_cost, index=[0], coregionalised=coregionalised, acceleration=acceleration)

        # average_input_length, average_output_length, list_of_assignments, total_partial_assignments, total_assignments, successful_combination, successful_assignments, number_of_breakages, obscured_assignment_problems, approx_number_of_occlusions, total_obscurations, total_obscurations_assigned_correctly, duration = gpr.gp_reconstructor(
        #     trajectories, threshold_distance, fraction=fraction, length=5., n_restarts=3, verbose=False,
        #     lengthscales=False, print_cost_matrix=False, plot_trajectories=False)

        print("--- %s ---" % gpr.seconds_to_str((time() - intermediate_time)))

        list_of_total_occlusions_assigned_correctly[index, index3] = total_obscurations_assigned_correctly
        list_of_total_occlusions[index, index3] = total_obscurations
        list_of_approx_number_of_occlusions[index, index3] = approx_number_of_occlusions
        list_of_obscured_assignment_problems[index, index3] = obscured_assignment_problems
        list_of_assignment_problems[index, index3] = total_assignments
        list_of_successful_recombinations[index, index3] = successful_combination
        list_of_successful_assignments[index, index3] = successful_assignments
        list_of_number_of_breakages[index, index3] = number_of_breakages
        # list_of_list_of_assignments[index, index3] = list_of_assignments
        list_of_total_partial_assignments[index, index3] = total_partial_assignments
        list_of_times[index, index3] = duration
        list_of_failure_cost_lists.append(costs_for_failures)
        list_of_success_cost_lists.append(costs_for_successes)
        list_of_average_input_length[index, index3] = average_number_of_input_positions
        list_of_average_output_length[index, index3] = average_number_of_output_positions
        list_of_acceleration_data_too_short[index, index3] = acceleration_data_too_short
    print("--- %s ---" % gpr.seconds_to_str((time() - total_time)))
    # print(f"Successes: {list_of_total_partial_assignments}")
    print(f"Times: {list_of_times}")
    # print(f"Number of Broken Trajectories: {list_of_number_of_broken_trajectories}")
    # fig = plt.figure()
    # plt.plot(threshold_distances, list_of_number_of_broken_trajectories)
    # plt.show()


TP.array_save(list_of_total_occlusions_assigned_correctly, "list_of_total_occlusions_assigned_correctly", f"{n}\\{local_folder}")
TP.array_save(list_of_total_occlusions, "list_of_total_occlusions", f"{n}\\{local_folder}")
TP.array_save(list_of_approx_number_of_occlusions, "list_of_approx_number_of_occlusions", f"{n}\\{local_folder}")
TP.array_save(list_of_obscured_assignment_problems, "list_of_obscured_assignment_problems", f"{n}\\{local_folder}")
TP.array_save(list_of_assignment_problems, "list_of_assignment_problems", f"{n}\\{local_folder}")
TP.array_save(list_of_successful_recombinations, "list_of_successful_recombinations", f"{n}\\{local_folder}")
TP.array_save(list_of_successful_assignments, "list_of_successful_assignments", f"{n}\\{local_folder}")
TP.array_save(list_of_number_of_breakages, "list_of_number_of_breakages", f"{n}\\{local_folder}")
TP.array_save(list_of_total_partial_assignments, "list_of_total_partial_assignments", f"{n}\\{local_folder}")
TP.array_save(list_of_times, "list_of_times", f"{n}\\{local_folder}")
TP.array_save(list_of_success_cost_lists, "list_of_success_costs", f"{n}\\{local_folder}")
TP.array_save(list_of_failure_cost_lists, "list_of_failure_costs", f"{n}\\{local_folder}")
TP.array_save(list_of_average_input_length, "list_of_average_input_length", f"{n}\\{local_folder}")
TP.array_save(list_of_average_output_length, "list_of_average_output_length", f"{n}\\{local_folder}")
TP.array_save(NN_dist_lst_of_arrays, f"NN_dist_lst_of_arrays", f"{n}\\{local_folder}")
TP.array_save(speeds, f"speeds", f"{n}\\{local_folder}")
TP.array_save(list_of_acceleration_data_too_short, f"Number_of_times_not_long_enough_input", f"{n}\\{local_folder}")

print(f"Success confidences: {list_of_success_cost_lists}")

fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(threshold_distances , 100*list_of_successful_assignments[index4]/list_of_number_of_breakages[index4])
plt.xlabel("Threshold Distances (m)")
plt.ylabel("Success Percentage")
plt.show()

fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(threshold_distances , list_of_times[index4])
plt.xlabel("Threshold Distances (m)")
plt.ylabel("Time of Computation (s)")
plt.show()

fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(threshold_distances , list_of_number_of_breakages[index4], label="Broken Trajectories")
    plt.plot(threshold_distances , list_of_successful_assignments[index4], label="Correct Reassignments")
plt.legend()
plt.xlabel("Threshold Distances (m)")
plt.ylabel("N")
plt.show()

fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(threshold_distances , list_of_assignment_problems[index4], label="Solvable assignment problems")
    plt.plot(threshold_distances , list_of_total_partial_assignments[index4]+list_of_assignment_problems[index4], label="Assignment problems (inc. partially solvable)")
    plt.plot(threshold_distances , list_of_successful_recombinations[index4], label="Correct Recombinations")
    plt.plot(threshold_distances , list_of_obscured_assignment_problems[index4], label="Obscured assignment problems")
plt.legend()
plt.xlabel("Threshold Distances (m)")
plt.ylabel("N")
plt.show()

fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(threshold_distances , list_of_approx_number_of_occlusions[index4], label="Approx Number of Occlusions")
    plt.plot(threshold_distances , list_of_total_occlusions[index4], label="Total Occlusions")
    plt.plot(threshold_distances , list_of_total_occlusions_assigned_correctly[index4], label="Correctly assigned occlusions")
plt.legend()
plt.xlabel("Threshold Distances (m)")
plt.ylabel("N")
plt.show()

fig = plt.figure()
for index4 in range(number_of_trajectories):
    n_bins = 20
    s_n, s_bins, _ = plt.hist(list_of_success_cost_lists[index4], n_bins, histtype='step', fill=False, density=False, label="Correct Recombinations", color='g', alpha=0.3)
    f_n, f_bins, _ = plt.hist(list_of_failure_cost_lists[index4], n_bins, histtype='step', fill=False, density=False, label="Incorrect Recombinations", color='r', alpha=0.3)
plt.show()

fig = plt.figure()
for index4 in range(number_of_trajectories):
    plt.plot(threshold_distances, list_of_average_output_length[index4])
    plt.plot(threshold_distances, list_of_average_input_length[index4])
plt.xlabel("Threshold Distances (m)")
plt.ylabel("Average Input and Output Length")
plt.show()






################
# input_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\code_testing_results\Input_list")
# output_list = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\code_testing_results\Output_list")
# times_array = TP.array_load(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\code_testing_results\times_array")
# gpr.train_GPs_on_position(input_list, output_list, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=True)

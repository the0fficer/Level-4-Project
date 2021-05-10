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
# n = "Results16traj_12_TD - Max Output 3"
# n = "Results16traj_20_TD - Max Output 3"
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
number_of_trajectories = np.shape(list_of_times)[0]
number_of_independent_variable = np.shape(list_of_times)[1]
list_of_acceleration_data_too_short = TP.array_load(f"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\{n}\\acceleration_no_switch\\Number_of_times_not_long_enough_input")

fraction = 1.
not_coregionalised_list_of_successful_assignments = TP.array_fractional_reducer(not_coregionalised_list_of_successful_assignments, fraction, 1)
coregionalised_list_of_successful_assignments = TP.array_fractional_reducer(coregionalised_list_of_successful_assignments, fraction, 1)
acceleration_list_of_successful_assignments = TP.array_fractional_reducer(acceleration_list_of_successful_assignments, fraction, 1)
acceleration_list_of_number_of_breakages = TP.array_fractional_reducer(acceleration_list_of_number_of_breakages, fraction, 1)
list_of_number_of_breakages = TP.array_fractional_reducer(list_of_number_of_breakages, fraction, 1)
list_of_acceleration_data_too_short = TP.array_fractional_reducer(list_of_acceleration_data_too_short, fraction, 1)
independent_variable = TP.array_fractional_reducer(independent_variable, fraction, 0)


cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, number_of_trajectories)]
alpha = 0.4
alpha1 = 0.1
alpha2 = 0.5
alpha3 = 0.5
fontsize = 13
labelsize = 12
linewidth=0.5
elinewidth=1.
summing = False
Meaning = True
Alling = False
medianing = False

fig = plt.figure(figsize=(6.25, 3.))
outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.4, bottom=0.18)

left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
right_cell = outer_grid[1].subgridspec(1, 1)

ax0 = fig.add_subplot(left_cell[:, :])
ax1 = fig.add_subplot(right_cell[:, :])

curved_trajectories = [0, 4, 5, 12]
both = [7, 13]
straight_trajectories = [1, 2, 3, 6, 8, 9, 10, 11, 14, 15]
trajectories = range(number_of_trajectories)
trajectories = range(10)
# trajectories = curved_trajectories

Delta_co_succ = (100*coregionalised_list_of_successful_assignments/list_of_number_of_breakages)-(100*not_coregionalised_list_of_successful_assignments / list_of_number_of_breakages)
Delta_acc_succ = (100*acceleration_list_of_successful_assignments/list_of_number_of_breakages)-(100*not_coregionalised_list_of_successful_assignments / list_of_number_of_breakages)
Delta_acc_succ_exc = (100*acceleration_list_of_successful_assignments/acceleration_list_of_number_of_breakages)-(100*not_coregionalised_list_of_successful_assignments / list_of_number_of_breakages)

if medianing:
    # if Alling:
    #     for index4 in trajectories:
    #         # plt.plot(independent_variable,
    #         #          100*coregionalised_list_of_successful_assignments[index4]/list_of_number_of_breakages[index4],
    #         #          color=colors[index4], alpha=alpha, ls=':')
    #         # plt.plot(independent_variable,
    #         #          100 * not_coregionalised_list_of_successful_assignments[index4] / list_of_number_of_breakages[index4],
    #         #          color=colors[index4], alpha=alpha, ls='-.')
    #         # plt.plot(independent_variable,
    #         #          100 *acceleration_list_of_successful_assignments[index4]/acceleration_list_of_number_of_breakages[index4],
    #         #          color=colors[index4], alpha=alpha, ls='--')
    #         plt.plot(independent_variable,
    #                  Delta_co_succ[index4],
    #                  color='b', alpha=alpha, ls='-')
    #         plt.plot(independent_variable,
    #                  Delta_acc_succ[index4],
    #                  color='r', alpha=alpha, ls='-')
    if Meaning:
        ax1.errorbar(independent_variable-0.003, np.nanmedian(Delta_co_succ[trajectories], axis=0), yerr=[np.nanpercentile(Delta_co_succ[trajectories],50, axis=0)-np.nanpercentile(Delta_co_succ[trajectories],25, axis=0), np.nanpercentile(Delta_co_succ[trajectories],75, axis=0)-np.nanpercentile(Delta_co_succ[trajectories],50, axis=0)], linestyle='-', marker='.', label="Coregionalised Method", color='k', alpha=alpha2, markersize=5 * (72. / fig.dpi) ** 2,lw=linewidth, elinewidth=elinewidth)
        # ax1.errorbar(independent_variable, np.nanmean(Delta_acc_succ[trajectories], axis=0), yerr=np.nanstd(Delta_acc_succ[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='.', label="Non-coregionalised Method", color='b', alpha=alpha2, markersize=5 * (72. / fig.dpi) ** 2)
        ax1.errorbar(independent_variable, np.nanmedian(Delta_acc_succ_exc[trajectories], axis=0), yerr=[np.nanpercentile(Delta_acc_succ_exc[trajectories],50, axis=0)-np.nanpercentile(Delta_acc_succ_exc[trajectories],25, axis=0), np.nanpercentile(Delta_acc_succ_exc[trajectories],75, axis=0)-np.nanpercentile(Delta_acc_succ_exc[trajectories],50, axis=0)], linestyle='-', marker='.', label="Non-coregionalised Method", color='blue', alpha=alpha2, markersize=5 * (72. / fig.dpi) ** 2, lw=linewidth, elinewidth=elinewidth)
        # ax1.errorbar(independent_variable+0.002, np.nanmean(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories], axis=0), yerr=np.nanstd(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='--', marker='x', label="Acceleration Method", color='green', alpha=alpha2)
        xlim = ax1.get_xlim()
        ax1.plot(xlim, [0, 0], color='green')
        ax1.set_xlim(xlim)
    # if summing:
    #     ax1.errorbar(independent_variable, 100*np.nansum(coregionalised_list_of_successful_assignments[trajectories], axis=0)/np.nansum(list_of_number_of_breakages[trajectories], axis=0), linestyle=':', marker='x', label="Coregionalised Method", color='k', alpha=alpha2)
    #     ax1.errorbar(independent_variable-0.002, 100*np.nansum(not_coregionalised_list_of_successful_assignments[trajectories], axis=0)/np.nansum(list_of_number_of_breakages[trajectories], axis=0), linestyle='-.', marker='x', label="Non-coregionalised Method", color='b', alpha=alpha2)
    #     ax1.errorbar(independent_variable+0.002, 100*np.nansum(acceleration_list_of_successful_assignments[trajectories], axis=0)/np.nansum(acceleration_list_of_number_of_breakages[trajectories], axis=0), linestyle='--', marker='x', label="Acceleration Method", color='green', alpha=alpha2)
    ax1.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
    ax1.set_ylabel("Comparison with\nMethod A (%)", fontsize=fontsize)

    ax0.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
    ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
    if Meaning:
        ax0.errorbar(independent_variable-0.003, np.nanmedian(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories], axis=0), yerr=[np.nanpercentile(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],50, axis=0)-np.nanpercentile(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],25, axis=0), np.nanpercentile(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],75, axis=0)-np.nanpercentile(100*coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],50, axis=0)], linestyle='-', marker='.', label="Coregionalised Method", color='k', alpha=alpha3, markersize=5 * (72. / fig.dpi) ** 2,lw=linewidth, elinewidth=elinewidth)
        ax0.errorbar(independent_variable, np.nanmedian(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories], axis=0), yerr=[np.nanpercentile(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],50, axis=0)-np.nanpercentile(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],25, axis=0), np.nanpercentile(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],75, axis=0)-np.nanpercentile(100*not_coregionalised_list_of_successful_assignments[trajectories]/list_of_number_of_breakages[trajectories],50, axis=0)], linestyle='-', marker='.', label="Non-coregionalised Method", color='green', alpha=alpha3, markersize=5 * (72. / fig.dpi) ** 2,lw=linewidth, elinewidth=elinewidth)
        ax0.errorbar(independent_variable+0.003, np.nanmedian(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories], axis=0), yerr=[np.nanpercentile(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],50, axis=0)-np.nanpercentile(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],25, axis=0), np.nanpercentile(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],75, axis=0)-np.nanpercentile(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],50, axis=0)], linestyle='-', marker='.', label="Acceleration Method", color='blue', alpha=alpha3, markersize=5 * (72. / fig.dpi) ** 2, lw=linewidth, elinewidth=elinewidth)
    #     ax0.errorbar(independent_variable-0.002, np.mean(list_of_number_of_breakages, axis=0), yerr=np.std(list_of_number_of_breakages,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Disconnections", color='r', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.mean(acceleration_list_of_number_of_breakages, axis=0), yerr=np.std(acceleration_list_of_number_of_breakages,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Disconnections", color='pink', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.mean(list_of_acceleration_data_too_short, axis=0), yerr=np.std(list_of_acceleration_data_too_short,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Not long enough", color='pink', alpha=alpha2)
    #     ax0.errorbar(independent_variable-0.002, np.mean(coregionalised_list_of_successful_assignments, axis=0), yerr=np.std(coregionalised_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle=':', marker='x', label="Correct Assignments-Coregionalised", color='k', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.mean(not_coregionalised_list_of_successful_assignments, axis=0), yerr=np.std(not_coregionalised_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='-.', marker='x', label="Correct Assignments-Not coregionalised", color='b', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.mean(acceleration_list_of_successful_assignments, axis=0), yerr=np.std(acceleration_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='--', marker='x', label="Correct Assignments-Acceleration", color='green', alpha=alpha2)
    # #####
    # if summing:
    #     ax0.errorbar(independent_variable-0.002, np.sum(list_of_number_of_breakages, axis=0),  linestyle='-', marker='x', label="Disconnections", color='k', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.sum(acceleration_list_of_number_of_breakages, axis=0),  linestyle='-', marker='x', label="Disconnections", color='grey', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.sum(list_of_acceleration_data_too_short, axis=0), linestyle='-', marker='x', label="Not long enough", color='pink', alpha=alpha2)
    #     ax0.errorbar(independent_variable-0.002, np.sum(coregionalised_list_of_successful_assignments, axis=0),  linestyle=':', marker='x', label="Correct Assignments-Coregionalised", color='k', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.sum(not_coregionalised_list_of_successful_assignments, axis=0),  linestyle='-.', marker='x', label="Correct Assignments-Not coregionalised", color='b', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.sum(acceleration_list_of_successful_assignments, axis=0),  linestyle='--', marker='x', label="Correct Assignments-Acceleration", color='green', alpha=alpha2)
    # if Alling:
    #     for index4 in trajectories:
    #         ax0.plot(independent_variable,
    #                  coregionalised_list_of_successful_assignments[index4],
    #                  color=colors[index4], alpha=alpha, ls=':')
    #         ax0.plot(independent_variable,
    #                  not_coregionalised_list_of_successful_assignments[index4],
    #                  color=colors[index4], alpha=alpha, ls='-.')
    #         ax0.plot(independent_variable,
    #                  acceleration_list_of_successful_assignments[index4],
    #                  color=colors[index4], alpha=alpha, ls='--')
    #         ax0.plot(independent_variable,
    #                  list_of_number_of_breakages[index4],
    #                  color=colors[index4], alpha=alpha, ls='-')
    #         ax0.plot(independent_variable,
    #                  list_of_number_of_breakages[index4],
    #                  color=colors[index4], alpha=alpha, ls='-')
    # ax0.legend(loc="upper left")
    ax0.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
    ax0.set_ylabel("Correct Recombinations (%)", fontsize=fontsize)
else:
    # if Alling:
    #     for index4 in trajectories:
    #         # plt.plot(independent_variable,
    #         #          100*coregionalised_list_of_successful_assignments[index4]/list_of_number_of_breakages[index4],
    #         #          color=colors[index4], alpha=alpha, ls=':')
    #         # plt.plot(independent_variable,
    #         #          100 * not_coregionalised_list_of_successful_assignments[index4] / list_of_number_of_breakages[index4],
    #         #          color=colors[index4], alpha=alpha, ls='-.')
    #         # plt.plot(independent_variable,
    #         #          100 *acceleration_list_of_successful_assignments[index4]/acceleration_list_of_number_of_breakages[index4],
    #         #          color=colors[index4], alpha=alpha, ls='--')
    #         plt.plot(independent_variable,
    #                  Delta_co_succ[index4],
    #                  color='b', alpha=alpha, ls='-')
    #         plt.plot(independent_variable,
    #                  Delta_acc_succ[index4],
    #                  color='r', alpha=alpha, ls='-')
    if Meaning:
        ax1.errorbar(independent_variable - 0.003, np.nanmean(Delta_co_succ[trajectories], axis=0),
                     yerr=np.nanstd(Delta_co_succ[trajectories], axis=0, ddof=1) / np.sqrt(len(trajectories)),
                     linestyle='-', marker='.', label="Coregionalised Method", color='k', alpha=alpha2,
                     markersize=5 * (72. / fig.dpi) ** 2, lw=linewidth, elinewidth=elinewidth)
        # ax1.errorbar(independent_variable, np.nanmean(Delta_acc_succ[trajectories], axis=0), yerr=np.nanstd(Delta_acc_succ[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='.', label="Non-coregionalised Method", color='b', alpha=alpha2, markersize=5 * (72. / fig.dpi) ** 2)
        ax1.errorbar(independent_variable, np.nanmean(Delta_acc_succ_exc[trajectories], axis=0),
                     yerr=np.nanstd(Delta_acc_succ_exc[trajectories], axis=0, ddof=1) / np.sqrt(len(trajectories)),
                     linestyle='-', marker='.', label="Non-coregionalised Method", color='blue', alpha=alpha2,
                     markersize=5 * (72. / fig.dpi) ** 2, lw=linewidth, elinewidth=elinewidth)
        # ax1.errorbar(independent_variable+0.002, np.nanmean(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories], axis=0), yerr=np.nanstd(100*acceleration_list_of_successful_assignments[trajectories]/acceleration_list_of_number_of_breakages[trajectories],axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='--', marker='x', label="Acceleration Method", color='green', alpha=alpha2)
        xlim = ax1.get_xlim()
        ax1.plot(xlim, [0, 0], color='green')
        ax1.set_xlim(xlim)
    # if summing:
    #     ax1.errorbar(independent_variable, 100*np.nansum(coregionalised_list_of_successful_assignments[trajectories], axis=0)/np.nansum(list_of_number_of_breakages[trajectories], axis=0), linestyle=':', marker='x', label="Coregionalised Method", color='k', alpha=alpha2)
    #     ax1.errorbar(independent_variable-0.002, 100*np.nansum(not_coregionalised_list_of_successful_assignments[trajectories], axis=0)/np.nansum(list_of_number_of_breakages[trajectories], axis=0), linestyle='-.', marker='x', label="Non-coregionalised Method", color='b', alpha=alpha2)
    #     ax1.errorbar(independent_variable+0.002, 100*np.nansum(acceleration_list_of_successful_assignments[trajectories], axis=0)/np.nansum(acceleration_list_of_number_of_breakages[trajectories], axis=0), linestyle='--', marker='x', label="Acceleration Method", color='green', alpha=alpha2)
    ax1.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
    ax1.set_ylabel("Comparison with\nMethod A (%)", fontsize=fontsize)
    ax0.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax0.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax0.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
    ax1.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
    if Meaning:
        ax0.errorbar(independent_variable - 0.003, np.nanmean(
            100 * coregionalised_list_of_successful_assignments[trajectories] / list_of_number_of_breakages[
                trajectories], axis=0), yerr=np.nanstd(
            100 * coregionalised_list_of_successful_assignments[trajectories] / list_of_number_of_breakages[
                trajectories], axis=0, ddof=1) / np.sqrt(len(trajectories)), linestyle='-', marker='.',
                     label="Coregionalised Method", color='k', alpha=alpha3, markersize=5 * (72. / fig.dpi) ** 2,
                     lw=linewidth, elinewidth=elinewidth)
        ax0.errorbar(independent_variable, np.nanmean(
            100 * not_coregionalised_list_of_successful_assignments[trajectories] / list_of_number_of_breakages[
                trajectories], axis=0), yerr=np.nanstd(
            100 * not_coregionalised_list_of_successful_assignments[trajectories] / list_of_number_of_breakages[
                trajectories], axis=0, ddof=1) / np.sqrt(len(trajectories)), linestyle='-', marker='.',
                     label="Non-coregionalised Method", color='green', alpha=alpha3,
                     markersize=5 * (72. / fig.dpi) ** 2, lw=linewidth, elinewidth=elinewidth)
        ax0.errorbar(independent_variable + 0.003, np.nanmean(
            100 * acceleration_list_of_successful_assignments[trajectories] / acceleration_list_of_number_of_breakages[
                trajectories], axis=0), yerr=np.nanstd(
            100 * acceleration_list_of_successful_assignments[trajectories] / acceleration_list_of_number_of_breakages[
                trajectories], axis=0, ddof=1) / np.sqrt(len(trajectories)), linestyle='-', marker='.',
                     label="Acceleration Method", color='blue', alpha=alpha3, markersize=5 * (72. / fig.dpi) ** 2,
                     lw=linewidth, elinewidth=elinewidth)
    #     ax0.errorbar(independent_variable-0.002, np.mean(list_of_number_of_breakages, axis=0), yerr=np.std(list_of_number_of_breakages,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Disconnections", color='r', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.mean(acceleration_list_of_number_of_breakages, axis=0), yerr=np.std(acceleration_list_of_number_of_breakages,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Disconnections", color='pink', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.mean(list_of_acceleration_data_too_short, axis=0), yerr=np.std(list_of_acceleration_data_too_short,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='', marker='x', label="Not long enough", color='pink', alpha=alpha2)
    #     ax0.errorbar(independent_variable-0.002, np.mean(coregionalised_list_of_successful_assignments, axis=0), yerr=np.std(coregionalised_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle=':', marker='x', label="Correct Assignments-Coregionalised", color='k', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.mean(not_coregionalised_list_of_successful_assignments, axis=0), yerr=np.std(not_coregionalised_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='-.', marker='x', label="Correct Assignments-Not coregionalised", color='b', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.mean(acceleration_list_of_successful_assignments, axis=0), yerr=np.std(acceleration_list_of_successful_assignments,axis=0, ddof=1)/np.sqrt(len(trajectories)), linestyle='--', marker='x', label="Correct Assignments-Acceleration", color='green', alpha=alpha2)
    # #####
    # if summing:
    #     ax0.errorbar(independent_variable-0.002, np.sum(list_of_number_of_breakages, axis=0),  linestyle='-', marker='x', label="Disconnections", color='k', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.sum(acceleration_list_of_number_of_breakages, axis=0),  linestyle='-', marker='x', label="Disconnections", color='grey', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.sum(list_of_acceleration_data_too_short, axis=0), linestyle='-', marker='x', label="Not long enough", color='pink', alpha=alpha2)
    #     ax0.errorbar(independent_variable-0.002, np.sum(coregionalised_list_of_successful_assignments, axis=0),  linestyle=':', marker='x', label="Correct Assignments-Coregionalised", color='k', alpha=alpha2)
    #     ax0.errorbar(independent_variable, np.sum(not_coregionalised_list_of_successful_assignments, axis=0),  linestyle='-.', marker='x', label="Correct Assignments-Not coregionalised", color='b', alpha=alpha2)
    #     ax0.errorbar(independent_variable+0.002, np.sum(acceleration_list_of_successful_assignments, axis=0),  linestyle='--', marker='x', label="Correct Assignments-Acceleration", color='green', alpha=alpha2)
    # if Alling:
    #     for index4 in trajectories:
    #         ax0.plot(independent_variable,
    #                  coregionalised_list_of_successful_assignments[index4],
    #                  color=colors[index4], alpha=alpha, ls=':')
    #         ax0.plot(independent_variable,
    #                  not_coregionalised_list_of_successful_assignments[index4],
    #                  color=colors[index4], alpha=alpha, ls='-.')
    #         ax0.plot(independent_variable,
    #                  acceleration_list_of_successful_assignments[index4],
    #                  color=colors[index4], alpha=alpha, ls='--')
    #         ax0.plot(independent_variable,
    #                  list_of_number_of_breakages[index4],
    #                  color=colors[index4], alpha=alpha, ls='-')
    #         ax0.plot(independent_variable,
    #                  list_of_number_of_breakages[index4],
    #                  color=colors[index4], alpha=alpha, ls='-')
    # ax0.legend(loc="upper left")
    ax0.set_xlabel(independent_variable_axis_title, fontsize=fontsize)
    ax0.set_ylabel("Correct Recombinations (%)", fontsize=fontsize)
    ax0.set_ylim([ax0.get_ylim()[0],100.])
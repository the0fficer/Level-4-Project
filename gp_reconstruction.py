import matplotlib.pyplot as plt
import numpy as np
from time import time, strftime, localtime
from datetime import timedelta, datetime
import pickle
import os
import GPy
from scipy.optimize import linear_sum_assignment
import statistics
import matplotlib.gridspec as gridspec
import TrajectoryPlotter as TP
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.special import erf
np.random.seed(0)

total_time = time()


def seconds_to_str(elapsed=None):
    """Takes a number of seconds and converts into the format Year-Month-Days Hours:Minutes:Seconds."""
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def read_trajectory(filename, n_parameters=3):
    """Produces an array of shape (n_birds, n_parameters, n_time_steps)
     from the trajectory .dat files and specifying the number of parameters (default=3)
     being measured (eg. position would be 3, position and velocity would be 6)."""

    with open(filename) as f:
        n_cols = len(f.readline().split(' '))

    trajectories = np.loadtxt(filename, dtype=float, delimiter=' ', usecols=range(1, n_cols))

    n_birds = int((n_cols - 1) / n_parameters)
    n_time_steps = np.shape(trajectories)[0]

    trajectories = trajectories.flatten(order='F')  # Need to be flattened in this way to format correctly
    trajectories = np.reshape(trajectories, (n_birds, n_parameters, n_time_steps))

    return trajectories


def trajectory_plotter(trajectories):
    """Creates a 3D plot of the trajectories."""

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectories')

    for i in range(np.ma.shape(trajectories)[0]):
        ax.plot(trajectories[i, 0, :], trajectories[i, 1, :], trajectories[i, 2, :], label=f"Bird {i}")

    # ax.legend()

    return plt.show()


def squared_distance_calculator(position1, position2):
    """Returns the square of the distance between two points in 3D space"""
    difference_vector = position2 - position1
    return np.dot(difference_vector, difference_vector)


def trajectory_switcher(trajectory1, trajectory2, index_of_cut):
    """Switches the latter parts of a 2-dimensional array at the index of the second axis."""

    trajectory1_a, trajectory2_b = trajectory1[:, :index_of_cut], trajectory1[:, index_of_cut:]
    trajectory2_a, trajectory1_b = trajectory2[:, :index_of_cut], trajectory2[:, index_of_cut:]

    trajectory1 = np.concatenate((trajectory1_a, trajectory1_b), axis=1)
    trajectory2 = np.concatenate((trajectory2_a, trajectory2_b), axis=1)

    return trajectory1, trajectory2


def splits_array_switcher(trajectory1, trajectory2, index_of_cut):
    """Switches the latter parts of a 2-dimensional array at the index of the second axis."""

    trajectory1_a, trajectory2_b = trajectory1[:index_of_cut], trajectory1[index_of_cut:]
    trajectory2_a, trajectory1_b = trajectory2[:index_of_cut], trajectory2[index_of_cut:]

    trajectory1 = np.concatenate((trajectory1_a, trajectory1_b), axis=0)
    trajectory2 = np.concatenate((trajectory2_a, trajectory2_b), axis=0)

    return trajectory1, trajectory2


def trajectory_error_correcter(trajectories):
    """Corrects 3D trajectories from a numpy array with shape (n_birds, n_parameters, n_time_steps)"""

    n_birds, n_paramaters, n_time_steps = np.shape(trajectories)

    for i in range(n_birds):
        if squared_distance_calculator(trajectories[i, :, 1],
                                       trajectories[i, :, 0]) > 1.5 * min(squared_distance_calculator(
            trajectories[i, :, 1], trajectories[i, :, 2]), squared_distance_calculator(
            trajectories[i, :, 2], trajectories[i, :, 3]), squared_distance_calculator(
            trajectories[i, :, 3], trajectories[i, :, 4])):
            for l in range(n_birds):
                if squared_distance_calculator(trajectories[i, :, 0],
                                               trajectories[l, :, 1]) < 1.5 * min(squared_distance_calculator(
                    trajectories[i, :, 1], trajectories[i, :, 2]), squared_distance_calculator(
                    trajectories[i, :, 2], trajectories[i, :, 3]), squared_distance_calculator(
                    trajectories[i, :, 3], trajectories[i, :, 4])):
                    trajectories[i, :, :], trajectories[l, :, :] = trajectory_switcher(trajectories[i, :, :],
                                                                                       trajectories[l, :, :], 1)
                    break
        for j in range(2, n_time_steps):
            if squared_distance_calculator(trajectories[i, :, j - 1],
                                           trajectories[i, :, j]) > 1.5 * squared_distance_calculator(
                    trajectories[i, :, j - 1], trajectories[i, :, j - 2]):
                for l in range(n_birds):
                    if squared_distance_calculator(trajectories[i, :, j - 1],
                                                   trajectories[l, :, j]) < 2 * squared_distance_calculator(
                            trajectories[i, :, j - 1], trajectories[i, :, j - 2]):
                        trajectories[i, :, :], trajectories[l, :, :] = trajectory_switcher(trajectories[i, :, :],
                                                                                           trajectories[l, :, :], j)
                        break
    return trajectories


def trajectory_error_correcter_improved(trajectories):
    """Corrects 3D trajectories from a numpy array with shape (n_birds, n_parameters, n_time_steps)"""

    n_birds, n_paramaters, n_time_steps = np.shape(trajectories)

    conditional_squared_distance = 3 * min(squared_distance_calculator(
        trajectories[0, :, 1], trajectories[0, :, 2]), squared_distance_calculator(
        trajectories[0, :, 2], trajectories[0, :, 3]), squared_distance_calculator(
        trajectories[0, :, 3], trajectories[0, :, 4]))

    difference_array = trajectories[:, :, 1:] - trajectories[:, :, :-1]
    squared_distance_array = np.sum(difference_array ** 2, axis=1)  # creates array with shape (n_birds, n_time_steps-1)
    splits_array = squared_distance_array > conditional_squared_distance  # Creates boolean array with True at location of splits
    splits_indices = np.array(np.nonzero(splits_array))  # Returns array with shape (n_axes, n_splits)

    counter = 0
    while len(splits_indices[0, :]) != 0 and counter < 3000:
        counter = + 1
        indices_of_birds_with_same_split = list(np.nonzero(splits_indices[1, :] == splits_indices[1, 0]))[0]
        position_of_first_bird = trajectories[splits_indices[0, 0], :, splits_indices[1, 0]]
        for counter, i in enumerate(indices_of_birds_with_same_split):
            position_of_second_bird = trajectories[splits_indices[0, i], :, splits_indices[1, i] + 1]
            if squared_distance_calculator(position_of_first_bird,
                                           position_of_second_bird) < conditional_squared_distance:
                trajectories[splits_indices[0, 0], :, :], trajectories[splits_indices[0, i], :,
                                                          :] = trajectory_switcher(
                    trajectories[splits_indices[0, 0], :, :],
                    trajectories[splits_indices[0, i], :, :], splits_indices[1, i] + 1)
                splits_array[splits_indices[0, 0], :], splits_array[splits_indices[0, i], :] = splits_array_switcher(
                    splits_array[splits_indices[0, 0], :],
                    splits_array[splits_indices[0, i], :], splits_indices[1, i])

                splits_array[splits_indices[0, 0], splits_indices[
                    1, 0]] = False  # CHANGE SPLITS_ARRAY AT LOCATION OF SPLIT MANUALLY.
                # splits_array[splits_indices[0, i], splits_indices[1, i]] = False
                splits_indices = np.array(np.nonzero(splits_array))
                break
        if counter == 3000:
            print("The trajectory correction failed")
        # trajectory_plotter(trajectories)
    return trajectories


def trajectory_reformatter(input_filename, folder_save_location= "trajectories_as_arrays", plot_trajectories=False):
    """Takes in .dat files, corrects trajectories and save them in pickled arrays. Plots trajectories in process."""
    trajectories = read_trajectory(input_filename)
    trajectories = trajectory_error_correcter_improved(trajectories)
    if plot_trajectories:
        trajectory_plotter(trajectories)
    n_birds, n_parameters, n_time_steps = np.shape(trajectories)
    savetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{n_birds}birds{n_time_steps}timesteps{savetime}"
    output_filename = f"{folder_save_location}/{filename}"
    with open(output_filename, 'wb') as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def folder_reformatter(folder_name, folder_save_location= "trajectories_as_arrays", plot_trajectories=False):
    """Takes folder name as input and reformats files and saves them into trajectories_as_arrays as """
    for filename in os.listdir(folder_name):
        start_time = time()
        input_filename = f"{folder_name}\{filename}"
        trajectory_reformatter(input_filename, folder_save_location=folder_save_location, plot_trajectories=plot_trajectories)
        print("\n")
        print(f"{filename}")
        print("--- %s ---" % seconds_to_str((time() - start_time)))
    return


def array_unpacker(file, plot_trajectories=False):
    with open(file, 'rb') as f:
        trajectories = pickle.load(f)
    if plot_trajectories:
        trajectory_plotter(trajectories)
    return trajectories


def folder_unpacker(folder_name, plot_trajectories=False):
    number_of_files = len(os.listdir(folder_name))
    trajectories_arrays = []
    for counter, filename in enumerate(sorted(os.listdir(folder_name))):
        start_time = time()
        input_filename = f"{folder_name}\{filename}"
        trajectories_arrays.append(array_unpacker(input_filename, plot_trajectories))
        print("\n")
        print(f"{filename}")
        print("--- %s ---" % seconds_to_str((time() - start_time)))
    print(f"The number of files is {number_of_files}")
    return trajectories_arrays


def array_fractional_reducer(array, fraction, axis=0):
    """Takes a fraction of evenly spaced elements from an array."""
    number_of_total_elements = np.shape(array)[axis]
    number_of_kept_elements = round(number_of_total_elements * fraction)
    index = np.round(np.linspace(0, np.shape(array)[axis] - 1, number_of_kept_elements)).astype(int)
    return np.take(array, index, axis)


def GPy_reformat_3D(array):
    """Take array of shape (n_axis, n_timesteps) or (n_timesteps) and return list of arrays appropriate for
    3D coregionalised regression. ie list of 3 arrays, each with shape (n_timesteps, 1)"""
    n_timesteps = np.shape(array)[-1]
    if len(np.shape(array)) == 1:
        array = array.reshape(n_timesteps, 1)
        return [array, array, array]
    elif len(np.shape(array)) == 2:
        array = array.T
        array1 = array[:, 0, None]
        array2 = array[:, 1, None]
        array3 = array[:, 2, None]
        return [array1, array2, array3]
    else:
        return print("Error in GPy_reformat, input array is wrong shape.")


def trajectory_masker(trajectory, split_start, split_length):
    """Take trajectory of shape [number of axis, number of timesteps] and return two trajectories of same size as before
    but one has a mask starting from the split and the other has a mask that finishes at the end of the split."""
    input_mask = np.zeros(np.shape(trajectory), dtype=bool)
    output_mask = np.zeros(np.shape(trajectory), dtype=bool)

    input_mask[:, split_start:] = 1
    output_mask[:, :split_start+split_length] = 1
    # print(np.ma.masked_array(trajectory, mask=mask_array))
    return np.ma.masked_array(trajectory, mask=input_mask), np.ma.masked_array(trajectory, mask=output_mask)


def train_GPs_on_position(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)"""
    # get list of Xs that line up with inputs and outputs and are limited to them.
    # for each input:
    #     train a GP
    #     compare actual outputs to predicted using defined function for MSLL
    #     store MSLL in array size that is the same size as the inputs and the outputs.
    # For the last line, if you do it for all inputs, can end up with a square array of inputs to outputs
    # Then I need some method of choosing the maximum combination of inputs and outputs. Research this...

    cost_matrix = np.zeros((len(list_of_input_trajectories),len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(6.25, 2.8))
        outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3, bottom=0.15)

        left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])

        ax = fig.add_subplot(left_cell[:, :])
        right_cell = outer_grid[1].subgridspec(5, 3, hspace=0.05)
        upper_right_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_cell[:5, :], hspace=0.0)
        # lower_right_cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_cell[3:, :], hspace=0.0)
        # upper_right_cell = right_Cell[:3, :].subgridspec(3, 1)
        # lower_right_cell = right_Cell[3:, :].subgridspec(2, 1)

        # axx = fig.add_subplot(right_cell[0, :])
        # axy = fig.add_subplot(right_cell[1, :])
        # axz = fig.add_subplot(right_cell[2, :])
        # ax2 = fig.add_subplot(right_cell[3, :])
        # ax3 = fig.add_subplot(right_cell[4, :])
        axx = fig.add_subplot(upper_right_cell[0])
        axy = fig.add_subplot(upper_right_cell[1], sharex=axx)
        axz = fig.add_subplot(upper_right_cell[2], sharex=axx)
        # ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
        # ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)

        ax.set_xlabel('Z (m)')
        ax.set_ylabel('X (m)')

        axins = ax.inset_axes([0.3, 0.6, 0.375, 0.35])   #Bottom left: [0.175, 0.15, 0.375, 0.35]; Top Left: [0.175, 0.6, 0.375, 0.35]
        axins.set_xlabel('Z (m)')
        axins.set_ylabel('Y (m)')

        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax2.tick_params(axis="both", labelbottom=False)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        # axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        axz.set_xlabel('Time (s)')
        # ax3.set_ylabel('Incorrect\nMatching')
        # ax2.set_ylabel('Correct\nMatching')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_input_trajectories))]
        alpha = 0.1
        alpha2 = 1

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3,-1))
        times_input_mask = input_mask[0,:]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        # REFORMAT THE ARRAY TO BE SUITABLE FOR GPy
        Y_List = GPy_reformat_3D(input_trajectory) # make sure input_trajectory has shape (3, n_timesteps)
        X_List = GPy_reformat_3D(input_times) # times should have shape (n_timesteps)


        icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=3, kernel=GPy.kern.RBF(1))
        # print(icm)

        gp = GPy.models.GPCoregionalizedRegression(X_List, Y_List, kernel=icm)
        gp['.*rbf.var'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
        # gp['.*mixed_noise.Gaussian_noise_1.variance'].constrain_fixed(1e-6)
        if lengthscales or verbose:
            print(f"\nInput: {i}")
        gp.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
    #     print(f"Cost: {gp.objective_function()}")
        if lengthscales:
            # print(gp.ICM.rbf.lengthscale)
            print(gp)
            print(gp.ICM.B.W)
            print(gp.ICM.B.kappa)
            # print(gp.ICM.B)
            # print(gp.mixed_noise)
            plot_3outputs_coregionalised(X_List, Y_List, gp)

        # if i == 0:
            # print(list_of_output_trajectories[0][0,80:85])
            # print(list_of_output_trajectories[1][0, 80:85])
            # print(list_of_output_trajectories[2][0, 80:85])
            # print(list_of_output_trajectories[3][0, 80:85])
            # print(list_of_output_trajectories[4][0, 80:85])

        # FINDING INDIVIDUAL COSTS
        if plot_cost:
            ax.plot(input_trajectory[2, :], input_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(input_trajectory[2, :], input_trajectory[0, :], marker='+', color=colors[i])
            axins.plot(input_trajectory[2, :], input_trajectory[1, :], ls='-', color=colors[i])

            output_trajectory_masked = list_of_output_trajectories[i] # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]
            ax.plot(output_trajectory[2, :], output_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(output_trajectory[2, :], output_trajectory[0, :], marker='1', color=colors[i])
            axins.plot(output_trajectory[2, :], output_trajectory[1, :], ls='-', color=colors[i])
            axx.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[0, :], color=colors[i], marker='x', s=50 * (72. / fig.dpi) ** 2)
            axy.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[1, :], color=colors[i], marker='1', s=50 * (72. / fig.dpi) ** 2)
            axz.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[2, :], color=colors[i], marker='+', s=50 * (72. / fig.dpi) ** 2)


            output_trajectory_masked = min(list_of_output_trajectories, key=np.ma.count_masked)
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            duration = max(output_times)-min(output_times)
            output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))

            xlim = [min(output_times), max(output_times)]
            axx.set_xlim(xlim)

            X_reshaped = output_times[:, None]
            array1 = output_trajectory.T[:, 0, None]
            array2 = output_trajectory.T[:, 1, None]
            array3 = output_trajectory.T[:, 2, None]
            Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
            noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
            Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)

            Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
            noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
            Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)

            Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
            noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
            Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)


            ax.plot(Zpred, Xpred, ls='--', color=colors[i], alpha=0.5)
            # axins.plot(Zpred, Ypred, ls='--', color=colors[i], alpha=0.5)
            axins.set_ylim([142, 148])
            axx.plot(output_times, Xpred, ls='--', color=colors[i], alpha=0.5)
            axy.plot(output_times, Ypred, ls='--', color=colors[i], alpha=0.5)
            axz.plot(output_times, Zpred, ls='--', color=colors[i], alpha=0.5)
            # axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axy.fill_between(x=output_times, y1=Ypred[:, 0] - Yvar[:, 0] ** 0.5, y2=Ypred[:, 0] + Yvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axz.fill_between(x=output_times, y1=Zpred[:, 0] - Zvar[:, 0] ** 0.5, y2=Zpred[:, 0] + Zvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            # plt.tight_layout()

        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = output_mask[0,:]
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]
            cost_matrix[i,j] = individual_cost_function(gp, output_trajectory, output_times, combined_axis_mean, plot_cost= plot_cost)
    if print_cost_matrix:
        print(cost_matrix)
    row_ind, col_ind, cost = combined_costs(cost_matrix)
    list_of_masked_times = []
    list_of_connected_trajectories = []
    trajectories = np.zeros((len(list_of_input_trajectories),3,np.shape(list_of_input_trajectories[0])[-1]))
    large_mask = np.zeros((len(list_of_input_trajectories),3, np.shape(list_of_input_trajectories[0])[-1]),dtype=bool)
    for i, row_index in enumerate(row_ind):
        col_index = col_ind[i]
        output_mask = np.ma.getmask(list_of_output_trajectories[col_index])
        input_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        trajectories[i, ~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        trajectories[i, ~input_mask] = list_of_input_trajectories[row_index][~input_mask]
        # list_of_input_trajectories[row_index][~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        # new_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        new_mask = (input_mask==True) & (output_mask==True)
        times_new_mask = new_mask[0, :]
        list_of_masked_times.append(np.ma.masked_array(times_array, mask=times_new_mask))
        # list_of_connected_trajectories.append(list_of_input_trajectories[row_index])
        # trajectories[i,:,:] = list_of_input_trajectories[row_index]
        large_mask[i,:,:] = new_mask
    masked_trajectories = np.ma.array(trajectories, mask=large_mask)
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    # ax1.set_ylabel("Y")
    # ax1.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_xlabel("X")
    # ax3.set_ylabel("Y")
    # ax3.set_xlabel("Z")
    # for i in range(len(list_of_input_trajectories)):
    #     ax1.plot(masked_trajectories[i,0,:],masked_trajectories[i,1,:])
    #     ax2.plot(masked_trajectories[i, 0, :], masked_trajectories[i, 2, :])
    #     ax3.plot(masked_trajectories[i, 2, :], masked_trajectories[i, 1, :])
    # fig.show()
    return masked_trajectories, col_ind, cost, row_ind


def combined_costs(matrix_MSLL_IO):
    """Choose the optimum combination based on the minimum combined cost. The method for combining the cost has yet to be determined (ie sum or product). Take the MSLL for each input and output and return the best combination of inputs and outputs as well as the probability of that choice."""
    # print(matrix_MSLL_IO)
    row_ind, col_ind = linear_sum_assignment(np.absolute(matrix_MSLL_IO), maximize=False)
    # cost = matrix_MSLL_IO[row_ind, col_ind].sum()/np.sum(matrix_MSLL_IO)
    if np.shape(matrix_MSLL_IO)[0]==np.shape(matrix_MSLL_IO)[1] and np.shape(matrix_MSLL_IO)[0]<=7:
        a = matrix_MSLL_IO[row_ind, col_ind]
        b = np.exp(a)
        c = b.prod()
        d = TP.perm(np.exp(matrix_MSLL_IO))
        e = c/d
        cost=np.exp(matrix_MSLL_IO[row_ind, col_ind]).prod()/TP.perm(np.exp(matrix_MSLL_IO))

    else:
        cost=np.nan
    return row_ind, col_ind, cost


def individual_cost_function(gp, output_trajectory, output_times, combining_axis_cost, plot_cost=False):
    """Calculate the cost function for a given input and output. Suggested cost function is the MSLL """
    # GET RIGHT PART OF ARRAY
    # REFORMAT
    # NOISE DATA
    # PREDICT NEW VALUES
    # GET COST.
    X_reshaped = output_times[:,None]
    # X_list = GPy_reformat_3D(output_times)
    # Y_list = GPy_reformat_3D(output_trajectory)

    # X_list = np.concatenate((X_reshaped,X_reshaped,X_reshaped), axis=1)
    # X_list = X_reshaped
    array1 = output_trajectory.T[:, 0, None]
    array2 = output_trajectory.T[:, 1, None]
    array3 = output_trajectory.T[:, 2, None]
    # Y_list = np.concatenate((array1,array2,array3),axis=1)
    # Y_list = array1
    # X_list = np.concatenate((X_reshaped,np.zeros_like(X_reshaped)),axis=1)


    Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)-1), axis=1)
    noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
    Xpred, Xvar = gp.predict(Times_pred_1,Y_metadata=noise_dict1)
    predictive_variance_X = Xvar + gp.mixed_noise.Gaussian_noise_0.variance**2
    # LL_X = - 0.5*np.log(2*np.pi*predictive_variance_X**2) - (array1-Xpred)**2/(2*predictive_variance_X**2)
    LL_X = - (array1 - Xpred) ** 2 / (2 * predictive_variance_X)#  ** 2)

    Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
    noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
    Ypred, Yvar = gp.predict(Times_pred_2,Y_metadata=noise_dict2)
    predictive_variance_Y = Yvar + gp.mixed_noise.Gaussian_noise_1.variance**2
    # LL_Y = - 0.5*np.log(2*np.pi*predictive_variance_Y**2) - (array2-Ypred)**2/(2*predictive_variance_Y**2)
    LL_Y = - (array2 - Ypred) ** 2 / (2 * predictive_variance_Y) #  ** 2)

    Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)+1), axis=1)
    noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
    Zpred, Zvar = gp.predict(Times_pred_3,Y_metadata=noise_dict3)
    predictive_variance_Z = Zvar + gp.mixed_noise.Gaussian_noise_2.variance**2
    # LL_Z = - 0.5*np.log(2*np.pi*predictive_variance_Z**2) - (array3-Zpred)**2/(2*predictive_variance_Z**2)
    LL_Z = - (array3 - Zpred) ** 2 / (2 * predictive_variance_Z) #  ** 2)
    #print(f"log term = {- 0.5*np.log(2*np.pi*predictive_variance_X**2)}")
    #print(f"difference term = {- (array1-Xpred)**2/(2*predictive_variance_X**2)}")
    # print(f"The actual x-coordinates are: {array1}")
    # print(f"The predicted x-coordinates are: {Xpred}")
    # print(f"The mean Xvar is:{np.mean(Xvar)}")
    # print(f"The mean Yvar is:{np.mean(Yvar)}")
    # print(f"The mean Zvar is:{np.mean(Zvar)}")

    if plot_cost:
        fig=plt.figure()
        plt.scatter(X_reshaped,LL_X, marker='x', label=r"Log Loss for X-axis")
        plt.scatter(X_reshaped,LL_Y, marker='x', label=r"Log Loss for Y-axis")
        plt.scatter(X_reshaped,LL_Z, marker='x', label=r"Log Loss for Z-axis")
        fig.legend()
        fig.show()
    cost = combining_axis_cost(LL_X,LL_Y,LL_Z)
    return cost


def kernel_function_1D(x, y, variance=1, lengthscale=1):
    """Define squared exponential kernel function."""
    kernel = variance * np.exp(- ((x - y) ** 2) / (2 * lengthscale ** 2))
    return kernel


def combined_axis_mean(array1, array2, array3):
    array = np.concatenate((array1,array2,array3), axis=1)
    return np.mean(array)


def square_dist_array(position, Array):
    x,y,z = position[0], position[1], position[2]
    deltax = x - Array[:,0]                             #Creating an array of differences in x position of particle and every other particle in array.
    deltay = y - Array[:,1]
    deltaz = z - Array[:,2]
    return deltax**2+deltay**2+deltaz**2


def break_trajectories(trajectories, dist_threshold):
    """Function that takes input array of trajectories and returns trajectories with masks in correct places. And a list of their indices in the trajectories array."""
    n_birds, n_axis, n_timesteps = np.shape(trajectories)
    trajectories_altered = []
    trajectories_altered_between_times = []
    trajectories = np.ma.array(trajectories)
    for t in range(n_timesteps):
        t_save = t
        for b in range(n_birds):
            time_test = False
            sq_distances_array = square_dist_array(trajectories[b,:,t],trajectories[:,:,t]) # Shape: (n_birds)
            # close_array = sq_distances_array < dist_threshold**2  # Creates boolean array with True at elements where birds are closer than d_threshold. Shape: (n_birds)
            # if np.count_nonzero(close_array) > 1:
            #     close_indices = np.nonzero(close_array)[0]  # Creates set of indices: (n_close_birds)
            close_indices = np.ma.where(sq_distances_array < dist_threshold ** 2)[0]
            if len(close_indices)>1:
                unique_close_indices1 = np.array([0]) # initialise array that will always be different to unique_close_indices2 for while loop start.
                unique_close_indices2 = np.unique(close_indices)
                time_test = True
                over_time_close_indices = unique_close_indices2
                times_of_alteration = np.array([t,0])
                while time_test and t<n_timesteps:
                    while np.any(unique_close_indices1 != unique_close_indices2):
                        unique_close_indices1 = unique_close_indices2
                        for b_i in unique_close_indices1:
                            intermediate_sq_distances_array = square_dist_array(trajectories[b_i, :, t], trajectories[:, :, t])
                            intermediate_close_array = intermediate_sq_distances_array < dist_threshold**2
                            intermediate_close_indices = np.nonzero(intermediate_close_array)[0]
                            close_indices = np.ma.concatenate((close_indices, intermediate_close_indices), axis=0)

                        unique_close_indices2 = np.unique(close_indices)
                        trajectories[unique_close_indices2,:,t] = np.ma.masked
                    over_time_close_indices = np.concatenate((over_time_close_indices,unique_close_indices2),axis=0)
                    over_time_close_indices = np.unique(over_time_close_indices)
                    t = t+1
                    if t == n_timesteps:
                        unique_close_indices1 = np.array([0])
                        break
                    time_test = False
                    close_indices = np.empty((0), dtype=np.int8)
                    for b_j in unique_close_indices1:
                        sq_distances_array = square_dist_array(trajectories[b_j, :, t], trajectories[:, :, t])  # Shape: (n_birds)
                        # close_array = sq_distances_array < dist_threshold ** 2  # Creates boolean array with True at elements where birds are closer than d_threshold. Shape: (n_birds)
                        # if np.count_nonzero(close_array) > 1:
                        #     other_intermediate_close_indices = np.nonzero(close_array)[0]  # Creates set of indices: (n_close_birds)
                        other_intermediate_close_indices = np.ma.where(sq_distances_array < dist_threshold ** 2)[0]
                        if len(other_intermediate_close_indices)>1:
                            close_indices = np.concatenate((close_indices, other_intermediate_close_indices), axis=0)
                            unique_close_indices2 = np.unique(close_indices)
                            time_test = True
                    unique_close_indices1 = np.array([0])  # initialise array with impossible index for while loop start.
                times_of_alteration[1] = t
                trajectories_altered_between_times.append(times_of_alteration)
                trajectories_altered.append(np.array(over_time_close_indices))
                t=t_save
    return trajectories, trajectories_altered, trajectories_altered_between_times

def flat_trajectory_plotter(trajectories):
    n_birds, n_axis, n_timesteps = np.shape(trajectories)
    fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    ax1.set_ylabel("Y")
    ax1.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_xlabel("Z")
    for i in range(n_birds):
        ax1.plot(trajectories[i,0,:],trajectories[i,1,:])
        ax2.plot(trajectories[i, 0, :], trajectories[i, 2, :])
        ax3.plot(trajectories[i, 2, :], trajectories[i, 1, :])
    fig.show()
    return


def mask_before(masked_array, mask_end):
    """Takes 2D array (axis,time) and masks all elements before mask end not including mask end."""
    mask = np.ma.getmaskarray(masked_array)
    mask[:, :mask_end] = 1
    return np.ma.masked_array(masked_array, mask=mask)


def mask_after(masked_array, mask_start):
    """Takes 2D array (axis,time) and masks all elements after (inclusive) mask_start."""
    mask = np.ma.getmaskarray(masked_array)
    mask[:, mask_start:] = 1
    return np.ma.masked_array(masked_array, mask=mask)


def masker(trajectories, trajectories_altered, trajectories_altered_between_times):
    list_of_input_lists = []
    list_of_output_lists = []
    list_of_occlusion_lists = []
    for i in range(len(trajectories_altered)):
        relevant_birds = trajectories_altered[i]
        relevant_trajectories = trajectories[relevant_birds, :, :]
        split_start0 = trajectories_altered_between_times[i][0]
        split_end0 = trajectories_altered_between_times[i][1]
        input_list = []
        output_list = []
        occlusion_list = []
        for j in range(np.shape(relevant_trajectories)[0]):
            intermediate1 = relevant_trajectories[j,0, split_start0:split_end0]
            intermediate2 = np.ma.getmaskarray(intermediate1)
            intermediate3 = np.nonzero(intermediate2)
            intermediate4 = intermediate3[0]
            intermdiate5 = np.amin(intermediate4)
            split_start1 = np.amin(np.nonzero(np.ma.getmaskarray(relevant_trajectories[j,0, split_start0:split_end0]))[0])
            split_end1 = np.amax(np.nonzero(np.ma.getmaskarray(relevant_trajectories[j,0, split_start0:split_end0]))[0])
            split_start = split_start0 + split_start1
            split_length = split_end1-split_start1+1
            input, output = trajectory_masker(relevant_trajectories[j,:,:], split_start, split_length)
            intermediate_test1 = np.nonzero(np.ma.getmaskarray(relevant_trajectories[j,0, :split_start0]))[0]
            intermediate_test2 = np.nonzero(np.ma.getmaskarray(relevant_trajectories[j,0, split_end0:]))[0]
            if np.shape(intermediate_test1)[0] != 0:
                prior_mask_end = np.amax(intermediate_test1)
                input = mask_before(input, prior_mask_end)
            if np.shape(intermediate_test2)[0] != 0:
                post_mask_start = np.amin(intermediate_test2)
                output = mask_after(output, post_mask_start+split_end0)
            input_list.append(input)
            output_list.append(output)
            occlusion_list.append(split_length)
            relevant_trajectories[j, :, split_start:split_end1+split_start] = np.ma.masked
            trajectories[relevant_birds[j], :, :] = relevant_trajectories[j]
        # assert(sum(occlusion_list)==np.sum(broken_trajectories.mask))
        list_of_input_lists.append(input_list)
        list_of_output_lists.append(output_list)
        list_of_occlusion_lists.append(occlusion_list)
    return list_of_input_lists, list_of_output_lists, list_of_occlusion_lists, trajectories

def repairer(list_of_input_lists, list_of_output_lists, list_of_occlusion_lists, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, swapping_IO=False, coregionalised=True, acceleration=False, fraction=0.1):
    assert(len(list_of_input_lists)==len(list_of_output_lists))
    total_obscurations = 0                              # Number of masked positions
    total_obscurations_assigned_correctly = 0           # Number of masked positions that have been correctly reassigned.
    number_of_breakages = 0                             # Number of interruptions of trajectories with input and output.
    successful_assignments = 0                          # Number of trajectories correctly assigned.
    total_assignments = 0                               # Number of solvable Assignment problems
    total_partial_assignments = 0                       # Number of Assignment problems that are not fully solvable due to missing I/Os
    successful_combination = 0                          # Number of Solved assignment problems
    obscured_assignment_problems = 0                    # Number of Assignment problems with 0 inputs or 0 outputs.
    number_of_output_positions = 0
    number_of_input_positions = 0
    list_of_assignments = []
    costs_for_successes = []
    costs_for_failures = []
    acceleration_data_too_short = 0
    for i in range(len(list_of_input_lists)):
        if i%5 == 0:
            print(i)

        average_input_length = statistics.median(map(np.ma.count_masked, list_of_input_lists[i]))# sum(map(np.ma.count_masked, list_of_input_lists[i])) / len(list_of_input_lists)  #ACTUALLY COUNTS MASK SO INVERSE INPUT LENGTH
        average_output_length = statistics.median(map(np.ma.count_masked, list_of_output_lists[i]))# sum(map(np.ma.count_masked, list_of_output_lists[i])) / len(list_of_output_lists)   # SEE ABOVE!!
        switchingIO = False
        if swapping_IO and average_input_length>=average_output_length:
            switchingIO = True
            input_list = list_of_output_lists[i]
            output_list = list_of_input_lists[i]
            input_length = np.shape(list_of_input_lists[i][0])[1] - average_output_length / 3
            output_length = np.shape(list_of_input_lists[i][0])[1] - average_input_length / 3

        else:
            input_list = list_of_input_lists[i]
            output_list = list_of_output_lists[i]
            input_length = np.shape(list_of_input_lists[i][0])[1] - average_input_length / 3
            output_length = np.shape(list_of_input_lists[i][0])[1] - average_output_length / 3
        number_of_input_positions += input_length
        if adjusting_output_len:
            number_of_output_positions += min(output_length, output_max_length)
        else:
            number_of_output_positions += output_length
        if not (any(array.mask.all()==True for array in input_list) or any(array.mask.all() for array in output_list)):
            total_obscurations += sum(list_of_occlusion_lists[i])
            if coregionalised:
                _, col_ind, cost, row_ind = train_GPs_on_position(input_list, output_list, times_array, n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales, print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, switchingIO=switchingIO)
            elif acceleration:
                input_list_length=len(input_list)
                input_list = [array for array in input_list if np.shape(list_of_input_lists[i][0])[1]-np.ma.count_masked(array)/3>=3]
                acceleration_data_too_short += input_list_length-len(input_list)
                if len(input_list)==0:
                    print(f"{i}: Input trajectory too short")
                    continue
                _, col_ind, cost, row_ind = train_GPs_on_position_acceleration(input_list, output_list, times_array,
                                                   n_restarts=3, verbose=verbose, lengthscales=lengthscales,
                                                   print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len,
                                                   output_max_length=output_max_length, plot_cost=False, switchingIO=False,
                                                   deltat=0.01/fraction)
            else:
                _, col_ind, cost, row_ind = train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=n_restarts,
                                                         verbose=verbose, lengthscales=lengthscales,
                                                         print_cost_matrix=print_cost_matrix,
                                                         adjusting_output_len=adjusting_output_len,
                                                         output_max_length=output_max_length, switchingIO=switchingIO)

            # col_ind = np.arange(0,len(input_list))
            total_assignments += 1
            number_of_breakages += len(col_ind)
            # number_of_breakages += len(output_list)
            successful_assignments += np.sum(col_ind == row_ind)
            list_of_assignments.append(col_ind)
            if np.all(col_ind == row_ind):
                successful_combination += 1
                costs_for_successes.append(cost)
                total_obscurations_assigned_correctly += sum(list_of_occlusion_lists[i])
                # if i == 5:
                #
            else:
                for index, col_index in enumerate(col_ind):
                    if col_index == row_ind[index]:
                        total_obscurations_assigned_correctly += list_of_occlusion_lists[i][index]
                print(f"Check here with debugger. i={i}")
                # if average_input_length<average_output_length:
                #     input_list = list_of_input_lists[i]
                #     output_list = list_of_output_lists[i]
                #     if coregionalised:
                #         train_GPs_on_position(input_list, output_list, times_array, n_restarts=n_restarts, verbose=verbose,
                #                               lengthscales=lengthscales, print_cost_matrix=print_cost_matrix,
                #                               adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, plot_cost=True, switchingIO=switchingIO)
                #     else:
                #         train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=n_restarts,
                #                               verbose=verbose,
                #                               lengthscales=lengthscales, print_cost_matrix=print_cost_matrix,
                #                               adjusting_output_len=adjusting_output_len,
                #                               output_max_length=output_max_length, plot_cost=True,
                #                               switchingIO=switchingIO)
                ##################
                # print(f"Input Lengths: {list(map(np.ma.count_masked, list_of_input_lists[i]))}")
                # print(f"Output Lengths: {list(map(np.ma.count_masked, list_of_output_lists[i]))}")
                # if average_input_length<average_output_length:
                #     print("Therefore, I => I")
                # else:
                #     print("Therefore, I => O")
                # print(f"Assignment: {col_ind}")
                # # print(list_of_input_lists[i])
                # # print(list_of_output_lists[i])
                # print()
                costs_for_failures.append(cost)
        elif not (len([array for array in input_list if not(array.mask.all()==True)])==0 or len([array for array in output_list if not(array.mask.all()==True)])==0):
        # elif not (all(array.mask.all() == True for array in input_list) or all(array.mask.all() for array in output_list)):
            total_partial_assignments += 1
            Ninput_list = []#[array for array in input_list if array.mask.all() == False]
            Noutput_list = []#[array for array in output_list if array.mask.all() == False]
            input_index_test = []
            output_index_test = []
            for input_index, array in enumerate(input_list):
                if array.mask.all() == False:
                    Ninput_list.append(array)
                    input_index_test.append(input_index)
            for output_index, array in enumerate(output_list):
                if array.mask.all() == False:
                    Noutput_list.append(array)
                    output_index_test.append(output_index)
            if coregionalised:
                _, col_ind, cost, row_ind = train_GPs_on_position(Ninput_list, Noutput_list, times_array, n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales, print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, switchingIO=switchingIO)
            elif acceleration:
                Ninput_list_length=len(Ninput_list)
                input_index_test = [index for array, index in zip(Ninput_list, input_index_test) if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]  # range(len(Ninput_list))
                Ninput_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
                acceleration_data_too_short += Ninput_list_length-len(Ninput_list)
                if len(Ninput_list)==0:
                    print(f"{i}: Input trajectory too short")
                    continue
                _, col_ind, cost, row_ind = train_GPs_on_position_acceleration(Ninput_list, Noutput_list, times_array,
                                                   n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales,
                                                   print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len,
                                                   output_max_length=output_max_length, plot_cost=False, switchingIO=False,
                                                   deltat=0.01/fraction)
                if type(_) == str:
                    print(f"{i}: Input trajectory too short")
                    acceleration_data_too_short+=1
                    continue
            else:
                _, col_ind, cost, row_ind = train_GPs_on_position_1D(Ninput_list, Noutput_list, times_array, n_restarts=n_restarts,
                                                         verbose=verbose, lengthscales=lengthscales,
                                                         print_cost_matrix=print_cost_matrix,
                                                         adjusting_output_len=adjusting_output_len,
                                                         output_max_length=output_max_length, switchingIO=switchingIO)

            # col_ind = np.arange(0,len(input_list))
            # #######total_assignments += 1
            # #######number_of_breakages += len(col_ind)
            # #######successful_assignments += np.sum(col_ind == np.arange(0,len(col_ind)))
            # #######list_of_assignments.append(col_ind)
            # if np.all(col_ind == np.arange(0,len(col_ind))):
            #     successful_combination += 1
            #     total_obscurations_assigned_correctly += sum(list_of_occlusion_lists[i])
            # else:
            #     for index, col_index in enumerate(col_ind):
            #         if col_index == np.arange(0,len(col_ind))[index]:
            #             total_obscurations_assigned_correctly += list_of_occlusion_lists[i][index]
            common_inputs_outputs = list(set(input_index_test).intersection(output_index_test))
            number_of_breakages += len(common_inputs_outputs)
            total_obscurations += sum([list_of_occlusion_lists[i][cio_index] for cio_index in common_inputs_outputs])
            for row_index, col_index in zip(row_ind,col_ind):
                if input_index_test[row_index]==output_index_test[col_index]:
                    total_obscurations_assigned_correctly += list_of_occlusion_lists[i][output_index_test[col_index]]
                    successful_assignments += 1
                    list_of_assignments.append(output_index_test[col_index])
        else:
            obscured_assignment_problems += 1
    # if total_assignments!=0:
    #     successful_combination_percentage = 100*successful_combination/total_assignments
    # else:
    #     successful_combination_percentage = np.nan
    # print(f"Rate of Successful Recombinations: {successful_combination_percentage}%")
    print(obscured_assignment_problems)
    print(len(list_of_input_lists)-total_assignments-total_partial_assignments)
    # assert(obscured_assignment_problems == len(list_of_input_lists)-total_assignments-total_partial_assignments)
    if len(list_of_input_lists)!=0:
        average_number_of_input_positions = number_of_input_positions/len(list_of_input_lists)
        average_number_of_output_positions = number_of_output_positions/len(list_of_output_lists)
    else:
        average_number_of_input_positions = np.nan
        average_number_of_output_positions = np.nan
    return average_number_of_input_positions, average_number_of_output_positions, list_of_assignments, total_partial_assignments, total_assignments, successful_combination, successful_assignments, number_of_breakages, obscured_assignment_problems, total_obscurations, total_obscurations_assigned_correctly, costs_for_successes, costs_for_failures, acceleration_data_too_short


def gp_reconstructor(trajectories, threshold_distance, from_scratch=False, fraction=0.1, length=80., n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, plot_trajectories=False, adjusting_output_len=False, output_max_length=5, swapping_IO=False, plot_cost=False, index=[0], coregionalised=True, acceleration=False):
    """Putting it all together."""
    if acceleration:
        assert(swapping_IO == False)
    trajectories = array_fractional_reducer(trajectories, fraction, 2)
    times_array = np.linspace(0, length, np.shape(trajectories)[2])
    if from_scratch:
        broken_trajectories, trajectories_altered, trajectories_altered_between_times = break_trajectories_as_though_from_scratch(
            trajectories, threshold_distance)
    else:
        broken_trajectories, trajectories_altered, trajectories_altered_between_times = break_trajectories(trajectories, threshold_distance)
    print(len(trajectories_altered))
    print(len(trajectories_altered_between_times))
    print(trajectories_altered)
    print(trajectories_altered_between_times)
    list_of_input_lists, list_of_output_lists, list_of_occlusion_lists, trajectories = masker(broken_trajectories, trajectories_altered,
                                                           trajectories_altered_between_times)
    # lst_input_lengths = [np.shape(array)[2] for input_list in list_of_input_lists for array in input_list]
    # average_input_length = sum(lst_input_lengths)/len(lst_input_lengths)
    # lst_output_lengths = [np.shape(array)[2] for output_list in list_of_input_lists for array in output_list]
    # average_output_length = sum(lst_output_lengths) / len(lst_output_lengths)
    # if plot_trajectories:
    #     TP.plot_2d_projections(broken_trajectories, trajectories_altered, bold_trajectories=index)
    #     # flat_trajectory_plotter(trajectories)
    if plot_cost:
        TP.array_save(times_array, f"times_array", f"code_testing_results")
        plottable_lsts_of_inputs = []
        plottable_lsts_of_outputs = []
        for i in index:
            plottable_lsts_of_inputs.append(list_of_input_lists[i ])
            plottable_lsts_of_outputs.append(list_of_output_lists[i])
            train_GPs_on_position_acceleration(list_of_input_lists[i], list_of_output_lists[i], times_array, n_restarts=n_restarts, verbose=verbose,
                                                            lengthscales=lengthscales, print_cost_matrix=print_cost_matrix,
                                                            adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, plot_cost=True)
        TP.array_save(plottable_lsts_of_inputs, f"Input_lsts", f"code_testing_results")
        TP.array_save(plottable_lsts_of_outputs, f"Output_lsts", f"code_testing_results")
    approx_number_of_occlusions = np.sum(trajectories.mask)/3
    start_time=time()
    average_number_of_input_positions, average_number_of_output_positions, list_of_assignments, total_partial_assignments, total_assignments, successful_combination, successful_assignments, number_of_breakages, obscured_assignment_problems, total_obscurations, total_obscurations_assigned_correctly, costs_for_successes, costs_for_failures, acceleration_data_too_short = repairer(list_of_input_lists, list_of_output_lists, list_of_occlusion_lists, times_array, n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales, print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, swapping_IO=swapping_IO, coregionalised=coregionalised, acceleration=acceleration, fraction=fraction)
    duration = time()-start_time
    # return average_input_length, average_output_length, list_of_assignments, total_partial_assignments, total_assignments, successful_combination, successful_assignments, number_of_breakages, obscured_assignment_problems, approx_number_of_occlusions, total_obscurations, total_obscurations_assigned_correctly, duration
    return average_number_of_input_positions, average_number_of_output_positions, list_of_assignments, total_partial_assignments, total_assignments, successful_combination, successful_assignments, number_of_breakages, obscured_assignment_problems, approx_number_of_occlusions, total_obscurations, total_obscurations_assigned_correctly, duration, costs_for_successes, costs_for_failures, acceleration_data_too_short


def break_trajectories_as_though_from_scratch(trajectories, dist_threshold):
    """Function that takes input array of trajectories and returns trajectories with masks in correct places. And a list of their indices in the trajectories array."""
    n_birds, n_axis, n_timesteps = np.shape(trajectories)
    trajectories_altered = []
    trajectories_altered_between_times = []
    trajectories = np.ma.array(trajectories)
    active_input_lst = []
    stable_input_lst = []
    active_output_lst = []
    stable_output_lst = []
    lst_of_running_output_sets = []
    active_alteration_times = []
    super_mask = np.zeros_like(trajectories, dtype=bool)
    for t in range(n_timesteps-2):      # Take away 1 cause it is not from len() but from np.shape, Take away 1 cause comparing against birds at time t+1.
        for b in range(n_birds):
            time_test = False
            sq_distances_array = square_dist_array(trajectories[b,:,t],trajectories[:,:,t+1]) # Shape: (n_birds)
            # close_array = sq_distances_array < dist_threshold**2  # Creates boolean array with True at elements where birds are closer than d_threshold. Shape: (n_birds)
            # if np.count_nonzero(close_array) > 1:
            #     close_indices = np.nonzero(close_array)[0]  # Creates set of indices: (n_close_birds)
            close_indices = np.ma.where(sq_distances_array < dist_threshold ** 2)[0]
            if len(close_indices)==1:
                for index, running_output_set in enumerate(lst_of_running_output_sets):
                    # for index, output_ind in enumerate(running_output_lst):
                    while b in lst_of_running_output_sets[index]:
                        lst_of_running_output_sets[index].remove(b)
                        if lst_of_running_output_sets[index] == set():
                            # lst_of_running_output_sets[index] = "delete this"
                            # copy active_input_lst[index] to stable stable_input_lst
                            stable_input_lst.append(set(active_input_lst[index]))
                            active_input_lst[index] = "delete this"
                            # copy active_output_lst[index] to stable stable_output_lst
                            stable_output_lst.append(set(active_output_lst[index]))
                            active_output_lst[index] = "delete this"
                            # find time altered loop and end it and append it to grand list
                            active_alteration_times[index][1] = t+1####Have added 1 to see if this corrects bug in line 667 where min is taken on empty list.
                            trajectories_altered_between_times.append(np.array(active_alteration_times[index]))
                            active_alteration_times[index] = "delete this"
                active_input_lst = [active_input_lst[index1] for index1 in range(len(active_input_lst)) if type(active_input_lst[index1]) == set]
                active_output_lst = [active_output_lst[index1] for index1 in range(len(active_output_lst)) if type(active_output_lst[index1]) == set]
                active_alteration_times = [active_alteration_times[index1] for index1 in range(len(active_alteration_times)) if type(active_alteration_times[index1]) == np.ndarray]
                lst_of_running_output_sets = [lst_of_running_output_sets[index1] for index1 in range(len(lst_of_running_output_sets)) if len(lst_of_running_output_sets[index1]) > 0]
                    #     while close_indices in output_lst:
                    #         output_lst.remove(close_indices)
                    #     if output_lst

            # IF SMALLER LEN IS EQUAL TO ONE
            #     THEN CHECK IF IN ANY OUTPUTS AND IF SO REMOVE THEM.
            #      IF IT IS, CHECK THAT THAT INPUT HAS MORE THAN ONE OUTPUT OTHER WISE REMOVE FROM INPUT LISTS.
            #     then check if input is any of outputs and if so DO SOMETHING TO RUNNING LIST.
            #     IF RUNNING LIST EMPTY, MOVE ACTIVE LIST OF INPUTS AND OUTPUTS TO STABLE LIST OF INPUTS AND OUTPUTS.
            elif len(close_indices)>1:
                super_mask[close_indices,:, t+1] = True
                test_for_continuation = False
                test_for_joining_another_input = False
                close_indices_set = set(close_indices)
                for index, output_set in enumerate(active_output_lst):
                    if b in output_set: # then this should imply b is also in input list.###BUT WHAT ABOUT CASE WHERE ONE INPUT LINKS TO TWO OUTPUTS BUT OTHER INPUT ONLY LINKS TO ONE.
                        active_input_lst[index].add(b)
                        # add b to set input list (or unique array)
                        active_output_lst[index] = set(active_output_lst[index]|close_indices_set)
                        # add close indices to active_output_lst
                        lst_of_running_output_sets[index] = set(lst_of_running_output_sets[index]|close_indices_set)
                        # add close indices to running_output_lst
                        test_for_continuation = True
                        test_for_joining_another_input = True
                if test_for_continuation == False:
                    lst_of_relevant_lsts = []
                    for index, output_set in enumerate(lst_of_running_output_sets):
                        if close_indices_set.intersection(output_set) != set():
                            lst_of_relevant_lsts.append(index)
                            active_input_lst[index].add(b)
                            # add b to input list
                            lst_of_running_output_sets[index] = set(lst_of_running_output_sets[index]|close_indices_set)
                            active_output_lst[index] = set(active_output_lst[index]|close_indices_set)
                            #] add close indices to set running_output_lst
                            test_for_joining_another_input = True
                    if len(lst_of_relevant_lsts)>1:         #Considering the case where a bird's position is a viable test in multiple outputs.
                        combined_set = set()                # initialising assignment problems to be joined (by bird index).
                        combined_times = []                 # Create list of alteration times
                        combined_set_inputs = set()
                        for counter, index in enumerate(lst_of_relevant_lsts):
                            combined_set = set(combined_set|lst_of_running_output_sets[index-counter])
                            combined_times.append(active_alteration_times[index-counter][0])
                            combined_set_inputs = set(combined_set_inputs|active_input_lst[index-counter])
                            # lst_of_running_output_sets.remove(lst_of_running_output_sets[index-counter].all())
                            # active_output_lst.remove(active_output_lst[index-counter].all())
                            # active_alteration_times.remove(active_alteration_times[index-counter].all())
                            del lst_of_running_output_sets[index-counter]
                            del active_output_lst[index-counter]
                            del active_alteration_times[index - counter]
                            del active_input_lst[index-counter]
                        lst_of_running_output_sets.append(set(combined_set))
                        active_output_lst.append(set(combined_set))
                        active_alteration_times.append(np.array([min(combined_times), n_timesteps-1]))
                        active_input_lst.append(set(combined_set_inputs))
                if test_for_joining_another_input == False:
                    active_alteration_times.append(np.array([t, n_timesteps-1]))
                    active_input_lst.append(set([b]))
                    active_output_lst.append(set(close_indices_set))
                    lst_of_running_output_sets.append(set(close_indices_set))
            else:
                raise ValueError("Threshold distance is too small")
    for index in range(len(active_input_lst)):
        trajectories_altered_between_times.append(active_alteration_times[index])
        stable_output_lst.append(active_output_lst[index])
        stable_input_lst.append(active_input_lst[index])
    print(f"Input lst is : {stable_input_lst}")
    print(f"Output lst is : {stable_output_lst}")
    assert(len(stable_input_lst)==len(stable_output_lst))
    assert(len(stable_input_lst)==len(trajectories_altered_between_times))
    # assert(stable_input_lst == stable_output_lst)
    trajectories = np.ma.masked_array(trajectories, mask=super_mask)
    trajectories_altered = [list(stable_input_lst[i].intersection(stable_output_lst[i])) for i in range(len(stable_input_lst)) if len(stable_input_lst[i].intersection(stable_output_lst[i]))>1]
    trajectories_altered_between_times = [trajectories_altered_between_times[i] for i in range(len(trajectories_altered_between_times)) if len(stable_input_lst[i].intersection(stable_output_lst[i]))>1]
    # for i in range(len(stable_input_lst)):
    #     trajectories_altered.append(list(stable_input_lst[i]))      #If input != output, could make this the intersection between each set, this should eliminate case where more outputs than inputs.
    return trajectories, trajectories_altered, trajectories_altered_between_times


# def framerate_to_threshold_distance(frame_rate, delta_t=0.01, max_speed=13.5):
#
#     return
def framerate_to_fraction_and_threshold_distance(frame_rate, delta_t=0.01, max_speed=13.5, leniency=1.3):
    assert(type(frame_rate)!=int)
    if type(frame_rate) == float:
        assert(frame_rate<=1/delta_t)
    elif type(frame_rate) == np.ndarray:
        assert(np.all(frame_rate<=1/delta_t))
    time_between_frames = 1/frame_rate
    fraction = delta_t/time_between_frames
    max_distance = time_between_frames*max_speed
    threshold_distance = leniency*max_distance
    return fraction, threshold_distance
# def presentation_graph_plotter(gp, # trajectory1, trajectory2, n_split, n_length, fraction, length):
#     """Function to make figure for report."""
#
#     # X, Y = GPy_get_X(trajectory1, keep_length=False, output_dim=3, length = length)
#     # Xprime, Yprime = GPy_get_X(trajectory2, keep_length=False, output_dim=3, length= length)
#     #
#     # # n_split = int(np.floor(np.shape(trajectory1)[-1]/2))
#     # # n_length = 3
#     #
#     # Y1, Y2, Y_mask = trajectory_splitter(trajectory1, n_split, n_length)
#     # Y1prime, Y2prime, Yprime_mask = trajectory_splitter(trajectory2, n_split, n_length)
#     # X1, X2, X_mask = trajectory_splitter(X.T, n_split, n_length)
#     # X1prime, X2prime, Xprime_mask = trajectory_splitter(X.T, n_split, n_length)
#     # X1, X2, X_mask = X1.T, X2.T, X_mask.T
#     # X1prime, X2prime, Xprime_mask = X1prime.T, X2prime.T, Xprime_mask.T
#     #
#     # throwaway1, GP1 = multi_dimensional_gaussian_plotter(Y1, extension_ratio=0., length=n_split*0.01/fraction, n_dimensions=3, fraction=1.)
#     # throwaway2, GP2 = multi_dimensional_gaussian_plotter(Y1prime, extension_ratio=0., length=n_split*0.01/fraction, n_dimensions=3, fraction=1.)
#     #
#     # # assert((intermediate1 == X1).all())
#     # # assert((intermediate2 == X1prime).all())
#     #
#     # Y1 = Y1[None, :, :] # treating each trajectory fragment as a separate bird.
#     # Y2 = Y2[None, :, :]
#     # Y1prime = Y1[None, :, :]
#     # Y2prime = Y2prime[None, :, :]
#     # # Y_mask = Y_mask[None, :, :]
#     # # Yprime_mask = Yprime_mask[None, :, :]
#     #
#     # # trajectories = np.ma.concatenate((Y_mask,Yprime_mask), axis=0)
#     # # print(f'The shape of the trajectories before is [2,{np.shape(trajectory1)}]\nThe shape of Y_mask is {np.shape(Y_mask)}\nThe shape of trajectories is {np.shape(trajectories)}')
#
#     fig = plt.figure(figsize=(9.,4.))
#     outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975,top=0.975, wspace=0.3)
#
#     left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
#
#     ax = fig.add_subplot(left_cell[:, :])
#     right_cell = outer_grid[1].subgridspec(5, 3, hspace=0.05)
#     upper_right_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_cell[:3, :], hspace=0.0)
#     lower_right_cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_cell[3:, :], hspace=0.0)
#     # upper_right_cell = right_Cell[:3, :].subgridspec(3, 1)
#     # lower_right_cell = right_Cell[3:, :].subgridspec(2, 1)
#
#     # axx = fig.add_subplot(right_cell[0, :])
#     # axy = fig.add_subplot(right_cell[1, :])
#     # axz = fig.add_subplot(right_cell[2, :])
#     # ax2 = fig.add_subplot(right_cell[3, :])
#     # ax3 = fig.add_subplot(right_cell[4, :])
#     axx = fig.add_subplot(upper_right_cell[0])
#     axy = fig.add_subplot(upper_right_cell[1], sharex=axx)
#     axz = fig.add_subplot(upper_right_cell[2], sharex=axx)
#     ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
#     ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)
#
#
#     ax.set_xlabel('Z')
#     ax.set_ylabel('Y')
#
#     # ax.plot(Y_mask[2,:],Y_mask[1,:], 'k-')
#     # ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b-')
#
#
#     # inverse_mask = ~np.array(np.ma.getmask(Y_mask), dtype=bool)
#     # Y_no_mask = np.ma.masked_array(Y_mask, ~np.ma.getmask(Y_mask))
#     # Yprime_no_mask = np.ma.masked_array(Yprime_mask, ~np.ma.getmask(Yprime_mask))
#     # ax.plot(Y_no_mask[2,:],Y_no_mask[1,:], 'k--')
#     # ax.plot(Yprime_no_mask[2, :], Yprime_no_mask[1, :], 'b--')
#
#
#     axins = ax.inset_axes([0.175, 0.15, 0.375, 0.35])
#     axins.plot(Yprime_mask[2, :], Yprime_mask[0, :], 'b-')
#     axins.plot(Y_mask[2,:],Y_mask[0,:], 'k-')
#     # axins.plot(Yprime_no_mask[2, :], Yprime_no_mask[0, :], 'b--')
#     # axins.plot(Y_no_mask[2,:],Y_no_mask[0,:], 'k--')
#     axins.set_xlabel('Z')
#     axins.set_ylabel('X')
#
#     Y_mask.mask = np.ma.nomask
#     Yprime_mask.mask = np.ma.nomask
#     ax.plot(Y_mask[2,:],Y_mask[1,:], 'k:')
#     ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b:')
#     axins.plot(Yprime_mask[2, :], Yprime_mask[0, :], 'b:')
#     axins.plot(Y_mask[2,:],Y_mask[0,:], 'k:')
#
#     ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#     axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#
#     axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#     axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#     axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#     ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#     ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
#     ax2.tick_params(axis="both", labelbottom=False)
#     axx.tick_params(axis="both", labelbottom=False)
#     axy.tick_params(axis="both", labelbottom=False)
#     axz.tick_params(axis="both", labelbottom=False)
#
#     X_list = [X1, X1, X1]
#     Xprime_list = [X1prime, X1prime, X1prime]
#     slices = GPy.util.multioutput.get_slices(X_list)
#     slicesprime = GPy.util.multioutput.get_slices(Xprime_list)
#     assert((slices == slicesprime))
#
#     axx.set_ylabel('X')
#     axy.set_ylabel('Y')
#     axz.set_ylabel('Z')
#     ax3.set_xlabel('Time')
#     ax3.set_ylabel('Incorrect\nMatching')
#     ax2.set_ylabel('Correct\nMatching')
#
#     xlim = [(n_split+n_length)*0.01/fraction + 0.1, length] # Can't be bothered to do the maths to work out why the 0.1 works. multiply by 0.01 for the timesteps.divid by fraction for beginning.
#     ax2.set_xlim(xlim)
#     ax2.set_ylim(-4,3.5)
#     ax3.set_ylim(-11,15)
#     # ax3.set_xlim(xlim)
#
#     # GP1.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)], which_data_rows=slices[0], legend=False, marker='+')
#     # GP1.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], which_data_rows=slices[1], legend=False, marker='+')
#     # GP1.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], which_data_rows=slices[2], legend=False, marker='+')
#     # GP2.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)], which_data_rows=slicesprime[0], legend=False, marker='+')
#     # GP2.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], which_data_rows=slicesprime[1], legend=False, marker='+')
#     # GP2.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], which_data_rows=slicesprime[2], legend=False, marker='+')
#
#
#     num_samples = 5
#
#     Xnewx = np.concatenate((X2, np.ones_like(X2)-1), axis=1)
#     noise_dict = {'output_index': Xnewx[:, 1:].astype(int)}
#     Xpred, Xvar = GP1.predict(Xnewx,Y_metadata=noise_dict)
#     Xnewx = np.concatenate((X2, np.ones_like(X2)-1), axis=1)
#     noise_dict = {'output_index': Xnewx[:, 1:].astype(int)}
#     Xpred_prime, Xprime_var = GP2.predict(Xnewx,Y_metadata=noise_dict)
#     Xquantiles = np.array(GP1.predict_quantiles(Xnewx,Y_metadata=noise_dict))
#     Xsamples = GP1.posterior_samples(Xnewx, Y_metadata=noise_dict, size=num_samples)
#     Xprime_quantiles = np.array(GP2.predict_quantiles(Xnewx,Y_metadata=noise_dict))
#     Xprime_samples = GP2.posterior_samples(Xnewx, Y_metadata=noise_dict, size=num_samples)
#
#     Ynewx = np.concatenate((X2, np.ones_like(X2)), axis=1)
#     noise_dict = {'output_index': Ynewx[:, 1:].astype(int)}
#     Ypred, Yvar = GP1.predict(Ynewx,Y_metadata=noise_dict)
#     Ynewx = np.concatenate((X2, np.ones_like(X2)), axis=1)
#     noise_dict = {'output_index': Ynewx[:, 1:].astype(int)}
#     Ypred_prime, Yprime_var = GP2.predict(Ynewx,Y_metadata=noise_dict)
#     Yquantiles = np.array(GP1.predict_quantiles(Ynewx,Y_metadata=noise_dict))
#     Ysamples = GP1.posterior_samples(Ynewx, Y_metadata=noise_dict, size=num_samples)
#     Yprime_quantiles = np.array(GP2.predict_quantiles(Ynewx,Y_metadata=noise_dict))
#     Yprime_samples = GP2.posterior_samples(Ynewx, Y_metadata=noise_dict, size=num_samples)
#
#
#     Znewx = np.concatenate((X2, np.ones_like(X2)+1), axis=1)
#     noise_dict = {'output_index': Znewx[:, 1:].astype(int)}
#     Zpred, Zvar = GP1.predict(Znewx,Y_metadata=noise_dict)
#     Znewx = np.concatenate((X2, np.ones_like(X2)+1), axis=1)
#     noise_dict = {'output_index': Znewx[:, 1:].astype(int)}
#     Zpred_prime, Zprime_var = GP2.predict(Znewx,Y_metadata=noise_dict)
#     Zquantiles = np.array(GP1.predict_quantiles(Znewx,Y_metadata=noise_dict))
#     Zsamples = GP1.posterior_samples(Znewx, Y_metadata=noise_dict, size=num_samples)
#     Zprime_quantiles = np.array(GP2.predict_quantiles(Znewx,Y_metadata=noise_dict))
#     Zprime_samples = GP2.posterior_samples(Znewx, Y_metadata=noise_dict, size=num_samples)
#
#
#     # axx.fill_between(x=Xnewx[:, 0], y1=Xquantiles[0, :, 0], y2=Xquantiles[1, :, 0], color='black', alpha=0.05)
#     # axx.fill_between(x=Xnewx[:, 0], y1=Xprime_quantiles[0, :, 0], y2=Xprime_quantiles[1, :, 0], color='blue', alpha=0.05)
#     # axy.fill_between(x=Xnewx[:, 0], y1=Yquantiles[0, :, 0], y2=Yquantiles[1, :, 0], color='black', alpha=0.05)
#     # axy.fill_between(x=Xnewx[:, 0], y1=Yprime_quantiles[0, :, 0], y2=Yprime_quantiles[1, :, 0], color='blue',alpha=0.05)
#     # axz.fill_between(x=Xnewx[:, 0], y1=Zquantiles[0, :, 0], y2=Zquantiles[1, :, 0], color='black', alpha=0.05)
#     # axz.fill_between(x=Xnewx[:, 0], y1=Zprime_quantiles[0, :, 0], y2=Zprime_quantiles[1, :, 0], color='blue',alpha=0.05)
#
#     axx.fill_between(x=Xnewx[:, 0], y1=Xpred[:,0]-Xvar[:,0]**0.5, y2=Xpred[:,0]+Xvar[:,0]**0.5, color='black', alpha=0.05)
#     axx.fill_between(x=Xnewx[:, 0], y1=Xpred_prime[:,0]-Xprime_var[:,0]**0.5, y2=Xpred_prime[:,0]+Xprime_var[:,0]**0.5, color='blue', alpha=0.05)
#     axy.fill_between(x=Xnewx[:, 0], y1=Ypred[:,0]-Yvar[:,0]**0.5, y2=Ypred[:,0]+Yvar[:,0]**0.5, color='black', alpha=0.05)
#     axy.fill_between(x=Xnewx[:, 0], y1=Ypred_prime[:,0]-Yprime_var[:,0]**0.5, y2=Ypred_prime[:,0]+Yprime_var[:,0]**0.5, color='blue', alpha=0.05)
#     axz.fill_between(x=Xnewx[:, 0], y1=Zpred[:,0]-Zvar[:,0]**0.5, y2=Zpred[:,0]+Zvar[:,0]**0.5, color='black', alpha=0.05)
#     axz.fill_between(x=Xnewx[:, 0], y1=Zpred_prime[:,0]-Zprime_var[:,0]**0.5, y2=Zpred_prime[:,0]+Zprime_var[:,0]**0.5, color='blue', alpha=0.05)
#
#     # axx.plot(X2, Xpred,  'k--', alpha=0.5)
#     # axx.plot(X2, Xpred_prime, 'b--', alpha=0.5)
#     # axy.plot(X2, Ypred, 'k--', alpha=0.5)
#     # axy.plot(X2, Ypred_prime, 'b--', alpha=0.5)
#     # axz.plot(X2, Zpred, 'k--', alpha=0.5)
#     # axz.plot(X2, Zpred_prime, 'b--', alpha=0.5)
#
#     # axx.scatter(X2[:, 0], Y2[0, 0, :],  color='k', marker='x', s=50*(72./fig.dpi)**2)
#     # axx.scatter(X2[:, 0], Y2prime[0, 0, :], color='b', marker='x', s=50*(72./fig.dpi)**2)
#     # axy.scatter(X2[:, 0], Y2[0, 1, :],  color='k', marker='1', s=50*(72./fig.dpi)**2)
#     # axy.scatter(X2[:, 0], Y2prime[0, 1, :], color='b', marker='1', s=50*(72./fig.dpi)**2)
#     # axz.scatter(X2[:, 0], Y2[0, 2, :],  color='k', marker='+', s=50*(72./fig.dpi)**2)
#     # axz.scatter(X2[:, 0], Y2prime[0, 2, :], color='b', marker='+', s=50*(72./fig.dpi)**2)
#
#     axx.plot(X2[:, 0], Y2[0, 0, :],  color='k', linestyle=':', alpha=0.5) # , s=50*(72./fig.dpi)**2)
#     axx.plot(X2[:, 0], Y2prime[0, 0, :], color='b', linestyle=':', alpha=0.5) # , s=50*(72./fig.dpi)**2)
#     axy.plot(X2[:, 0], Y2[0, 1, :],  color='k', linestyle='--', alpha=0.5) # , s=50*(72./fig.dpi)**2)
#     axy.plot(X2[:, 0], Y2prime[0, 1, :], color='b', linestyle='--', alpha=0.5) # , s=50*(72./fig.dpi)**2)
#     axz.plot(X2[:, 0], Y2[0, 2, :],  color='k', linestyle='-.', alpha=0.5) # , s=50*(72./fig.dpi)**2)
#     axz.plot(X2[:, 0], Y2prime[0, 2, :], color='b', linestyle='-.', alpha=0.5) # , s=50*(72./fig.dpi)**2)
#
#     Xresiduals = (Y2[0, 0, :] - Xpred[:,0])/Xvar[:,0]**0.5
#     Yresiduals = (Y2[0, 1, :] - Ypred[:,0])/Yvar[:,0]**0.5
#     Zresiduals = (Y2[0, 2, :] - Zpred[:,0])/Zvar[:,0]**0.5
#     # ax2.scatter(X2[:, 0], Xresiduals, color='k', marker='x', alpha=0.3)# , s=(72./fig.dpi)**2)
#     # ax2.scatter(X2[:, 0], Yresiduals, color='k', marker='1', alpha=0.3)# , s=(72./fig.dpi)**2)
#     # ax2.scatter(X2[:, 0], Zresiduals, color='k', marker='+', alpha=0.3)# , s=(72./fig.dpi)**2)
#     ax2.plot(X2[:, 0], Xresiduals, color='k', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     ax2.plot(X2[:, 0], Yresiduals, color='k', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     ax2.plot(X2[:, 0], Zresiduals, color='k', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)
#
#     Xprime_residuals = (Y2prime[0, 0, :] - Xpred_prime[:, 0])/Xprime_var[:,0]**0.5
#     Yprime_residuals = (Y2prime[0, 1, :] - Ypred_prime[:, 0])/Yprime_var[:,0]**0.5
#     Zprime_residuals = (Y2prime[0, 2, :] - Zpred_prime[:, 0])/Zprime_var[:,0]**0.5
#     # ax2.scatter(X2[:, 0], Xprime_residuals, color='b', marker='x', alpha=0.3)  # , s=(72./fig.dpi)**2)
#     # ax2.scatter(X2[:, 0], Yprime_residuals, color='b', marker='1', alpha=0.3)  # , s=(72./fig.dpi)**2)
#     # ax2.scatter(X2[:, 0], Zprime_residuals, color='b', marker='+', alpha=0.3)  # , s=(72./fig.dpi)**2)
#     ax2.plot(X2[:, 0], Xprime_residuals, color='b', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     ax2.plot(X2[:, 0], Yprime_residuals, color='b', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     ax2.plot(X2[:, 0], Zprime_residuals, color='b', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)
#
#
#     bad_Xresiduals = (Y2prime[0, 0, :] - Xpred[:, 0])/Xvar[:,0]**0.5
#     bad_Yresiduals = (Y2prime[0, 1, :] - Ypred[:, 0])/Yvar[:,0]**0.5
#     bad_Zresiduals = (Y2prime[0, 2, :] - Zpred[:, 0])/Zvar[:,0]**0.5
#     # ax3.scatter(X2[:, 0], bad_Xresiduals, color='k', marker='x', alpha=0.3)# , s=(72./fig.dpi)**2)
#     # ax3.scatter(X2[:, 0], bad_Yresiduals, color='k', marker='1', alpha=0.3)# , s=(72./fig.dpi)**2)
#     # ax3.scatter(X2[:, 0], bad_Zresiduals, color='k', marker='+', alpha=0.3)# , s=(72./fig.dpi)**2)
#     ax3.plot(X2[:, 0], bad_Xresiduals, color='k', linestyle=':', alpha=0.5)# , s=(72./fig.dpi)**2)
#     ax3.plot(X2[:, 0], bad_Yresiduals, color='k', linestyle='--', alpha=0.5)# , s=(72./fig.dpi)**2)
#     ax3.plot(X2[:, 0], bad_Zresiduals, color='k', linestyle='-.', alpha=0.5)# , s=(72./fig.dpi)**2)
#
#
#     bad_Xprime_residuals = (Y2[0, 0, :] - Xpred_prime[:,0])/Xprime_var[:,0]**0.5
#     bad_Yprime_residuals = (Y2[0, 1, :] - Ypred_prime[:,0])/Yprime_var[:,0]**0.5
#     bad_Zprime_residuals = (Y2[0, 2, :] - Zpred_prime[:,0])/Zprime_var[:,0]**0.5
#     # ax3.scatter(X2[:, 0], bad_Xprime_residuals, color='b', marker='x', alpha=0.3)  # , s=(72./fig.dpi)**2)
#     # ax3.scatter(X2[:, 0], bad_Yprime_residuals, color='b', marker='1', alpha=0.3)  # , s=(72./fig.dpi)**2)
#     # ax3.scatter(X2[:, 0], bad_Zprime_residuals, color='b', marker='+', alpha=0.3)  # , s=(72./fig.dpi)**2)
#     ax3.plot(X2[:, 0], bad_Xprime_residuals, color='b', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     ax3.plot(X2[:, 0], bad_Yprime_residuals, color='b', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     ax3.plot(X2[:, 0], bad_Zprime_residuals, color='b', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)
#     # ylabels = axz.get_yticklabels()
#     # print(ylabels)
#
#     # X2_list = [X2, X2, X2]
#     #
#     # print(GP3.log_predictive_density())
#     return fig.show()


def train_GPs_on_position_1D(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)"""
    # get list of Xs that line up with inputs and outputs and are limited to them.
    # for each input:
    #     train a GP
    #     compare actual outputs to predicted using defined function for MSLL
    #     store MSLL in array size that is the same size as the inputs and the outputs.
    # For the last line, if you do it for all inputs, can end up with a square array of inputs to outputs
    # Then I need some method of choosing the maximum combination of inputs and outputs. Research this...

    cost_matrix = np.zeros((len(list_of_input_trajectories),len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(6.25, 2.8))
        outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3, bottom=0.15)

        left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])

        ax = fig.add_subplot(left_cell[:, :])
        right_cell = outer_grid[1].subgridspec(5, 3, hspace=0.05)
        upper_right_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_cell[:5, :], hspace=0.0)
        # lower_right_cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_cell[3:, :], hspace=0.0)
        # upper_right_cell = right_Cell[:3, :].subgridspec(3, 1)
        # lower_right_cell = right_Cell[3:, :].subgridspec(2, 1)

        # axx = fig.add_subplot(right_cell[0, :])
        # axy = fig.add_subplot(right_cell[1, :])
        # axz = fig.add_subplot(right_cell[2, :])
        # ax2 = fig.add_subplot(right_cell[3, :])
        # ax3 = fig.add_subplot(right_cell[4, :])
        axx = fig.add_subplot(upper_right_cell[0])
        axy = fig.add_subplot(upper_right_cell[1], sharex=axx)
        axz = fig.add_subplot(upper_right_cell[2], sharex=axx)
        # ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
        # ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)

        ax.set_xlabel('Z (m)')
        ax.set_ylabel('X (m)')

        axins = ax.inset_axes([0.3, 0.6, 0.375, 0.35])   #Bottom left: [0.175, 0.15, 0.375, 0.35]; Top Left: [0.175, 0.6, 0.375, 0.35]
        axins.set_xlabel('Z (m)')
        axins.set_ylabel('Y (m)')

        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax2.tick_params(axis="both", labelbottom=False)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        # axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        axz.set_xlabel('Time (s)')
        # ax3.set_ylabel('Incorrect\nMatching')
        # ax2.set_ylabel('Correct\nMatching')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_input_trajectories))]
        alpha = 0.1
        alpha2 = 1

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3,-1))
        X = input_trajectory[0, :, None]
        Y = input_trajectory[1, :, None]
        Z = input_trajectory[2, :, None]
        times_input_mask = input_mask[0,:]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        input_times = input_times[:, None]
        kernelx = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        kernely = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        kernelz = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

        gpx = GPy.models.GPRegression(input_times, X, kernelx)
        gpy = GPy.models.GPRegression(input_times, Y, kernely)
        gpz = GPy.models.GPRegression(input_times, Z, kernelz)
        # gpx.rbf.lengthscale.unconstrain()
        # gpy.rbf.lengthscale.unconstrain()
        # gpz.rbf.lengthscale.unconstrain()
        # gpx.rbf.lengthscale.constrain_bounded(0.1, 100)
        # gpy.rbf.lengthscale.constrain_bounded(547, 549)
        # gpz.rbf.lengthscale.constrain_bounded(0.1, 100)
        # gpy.Gaussian_noise.variance.unconstrain()
        noise = 1e-6
        # gpx.Gaussian_noise.variance.constrain_fixed(noise)
        gpy.Gaussian_noise.variance.constrain_fixed(noise)
        # gpz.Gaussian_noise.variance.constrain_fixed(noise)
        # gpy.Gaussian_noise.variance.constrain_bounded(1e-7, 1e-3)

        if lengthscales or verbose:
            print(f"\nInput: {i}")
        gpx.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        gpy.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        gpz.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
    #     print(f"Cost: {gp.objective_function()}")
        if lengthscales:
            # print(f"X timescale: {gpx.rbf.lengthscale}")
            # print(f"Y timescale: {gpy.rbf.lengthscale}")
            # print(f"Z timescale: {gpz.rbf.lengthscale}")
            print(f"X timescale: \n{gpx}")
            print(f"Y timescale: \n{gpy}")
            print(f"Z timescale: \n{gpz}")
            plot_3outputs_independent(gpx, gpy, gpz, input_times, X,Y,Z)

        # FINDING INDIVIDUAL COSTS
        if plot_cost:
            ax.plot(input_trajectory[2, :], input_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(input_trajectory[2, :], input_trajectory[0, :], marker='+', color=colors[i])
            axins.plot(input_trajectory[2, :], input_trajectory[1, :], ls='-', color=colors[i])

            output_trajectory_masked = list_of_output_trajectories[i] # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]

            ax.plot(output_trajectory[2, :], output_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(output_trajectory[2, :], output_trajectory[0, :], marker='1', color=colors[i])
            axins.plot(output_trajectory[2, :], output_trajectory[1, :], ls='-', color=colors[i])
            axx.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[0, :], color=colors[i], marker='x', s=50 * (72. / fig.dpi) ** 2)
            axy.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[1, :], color=colors[i], marker='1', s=50 * (72. / fig.dpi) ** 2)
            axz.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[2, :], color=colors[i], marker='+', s=50 * (72. / fig.dpi) ** 2)


            output_trajectory_masked = min(list_of_output_trajectories, key=np.ma.count_masked)
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            duration = max(output_times)-min(output_times)
            output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))

            xlim = [min(output_times), max(output_times)]
            axx.set_xlim(xlim)

            X_reshaped = output_times[:, None]
            array1 = output_trajectory.T[:, 0, None]
            array2 = output_trajectory.T[:, 1, None]
            array3 = output_trajectory.T[:, 2, None]
            # Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
            # noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
            # Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
            #
            # Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
            # noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
            # Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
            #
            # Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
            # noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
            # Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)
            Xpred, Xvar = gpx.predict(X_reshaped)
            Ypred, Yvar = gpy.predict(X_reshaped)
            Zpred, Zvar = gpz.predict(X_reshaped)

            ax.plot(Zpred, Xpred, ls='--', color=colors[i], alpha=0.5)
            # axins.plot(Zpred, Ypred, ls='--', color=colors[i], alpha=0.5)
            axins.set_ylim([142, 148])
            axx.plot(output_times, Xpred, ls='--', color=colors[i], alpha=0.5)
            axy.plot(output_times, Ypred, ls='--', color=colors[i], alpha=0.5)
            axz.plot(output_times, Zpred, ls='--', color=colors[i], alpha=0.5)
            # axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axy.fill_between(x=output_times, y1=Ypred[:, 0] - Yvar[:, 0] ** 0.5, y2=Ypred[:, 0] + Yvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axz.fill_between(x=output_times, y1=Zpred[:, 0] - Zvar[:, 0] ** 0.5, y2=Zpred[:, 0] + Zvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axy.set_ylim([axy.get_ylim()[0], axy.get_ylim()[1] + 5])
            axy.yaxis.set_major_locator(MultipleLocator(175))
            # plt.tight_layout()
        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = output_mask[0,:]
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]

            cost_matrix[i,j] = individual_cost_function_1D(gpx, gpy, gpz, output_trajectory, output_times, combined_axis_mean, plot_cost=plot_cost)
    if print_cost_matrix:
        print(cost_matrix)
    row_ind, col_ind, cost = combined_costs(cost_matrix)
    # print(f"The row indices are: {row_ind}")
    # print(f"The column indices are: {col_ind}")
    list_of_masked_times = []
    list_of_connected_trajectories = []
    trajectories = np.zeros((len(list_of_input_trajectories),3,np.shape(list_of_input_trajectories[0])[-1]))
    large_mask = np.zeros((len(list_of_input_trajectories),3, np.shape(list_of_input_trajectories[0])[-1]),dtype=bool)
    for i, row_index in enumerate(row_ind):
        col_index = col_ind[i]
        output_mask = np.ma.getmask(list_of_output_trajectories[col_index])
        input_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        trajectories[i, ~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        trajectories[i, ~input_mask] = list_of_input_trajectories[row_index][~input_mask]
        # list_of_input_trajectories[row_index][~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        # new_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        new_mask = (input_mask==True) & (output_mask==True)
        times_new_mask = new_mask[0, :]
        list_of_masked_times.append(np.ma.masked_array(times_array, mask=times_new_mask))
        # list_of_connected_trajectories.append(list_of_input_trajectories[row_index])
        # trajectories[i,:,:] = list_of_input_trajectories[row_index]
        large_mask[i,:,:] = new_mask
    masked_trajectories = np.ma.array(trajectories, mask=large_mask)
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    # ax1.set_ylabel("Y")
    # ax1.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_xlabel("X")
    # ax3.set_ylabel("Y")
    # ax3.set_xlabel("Z")
    # for i in range(len(list_of_input_trajectories)):
    #     ax1.plot(masked_trajectories[i,0,:],masked_trajectories[i,1,:])
    #     ax2.plot(masked_trajectories[i, 0, :], masked_trajectories[i, 2, :])
    #     ax3.plot(masked_trajectories[i, 2, :], masked_trajectories[i, 1, :])
    # fig.show()
    return masked_trajectories, col_ind, cost, row_ind


def individual_cost_function_1D(gpx, gpy, gpz, output_trajectory, output_times, combining_axis_cost, plot_cost=False):
    """Calculate the cost function for a given input and output. Suggested cost function is the MSLL """
    # GET RIGHT PART OF ARRAY
    # REFORMAT
    # NOISE DATA
    # PREDICT NEW VALUES
    # GET COST.
    X_reshaped = output_times[:,None]
    # X_list = GPy_reformat_3D(output_times)
    # Y_list = GPy_reformat_3D(output_trajectory)

    # X_list = np.concatenate((X_reshaped,X_reshaped,X_reshaped), axis=1)
    # X_list = X_reshaped
    array1 = output_trajectory.T[:, 0, None]
    array2 = output_trajectory.T[:, 1, None]
    array3 = output_trajectory.T[:, 2, None]
    # Y_list = np.concatenate((array1,array2,array3),axis=1)
    # Y_list = array1
    # X_list = np.concatenate((X_reshaped,np.zeros_like(X_reshaped)),axis=1)
    Xpred, Xvar = gpx.predict(X_reshaped)
    Ypred, Yvar = gpy.predict(X_reshaped)
    Zpred, Zvar = gpz.predict(X_reshaped)
    predictive_variance_X = Xvar + gpx.Gaussian_noise.variance ** 2
    predictive_variance_Y = Yvar + gpy.Gaussian_noise.variance ** 2
    predictive_variance_Z = Zvar + gpz.Gaussian_noise.variance ** 2
    # Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)-1), axis=1)
    # noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
    # Xpred, Xvar = gpx.predict(Times_pred_1,Y_metadata=noise_dict1)
    # predictive_variance_X = Xvar + gp.mixed_noise.Gaussian_noise_0.variance**2
    # # LL_X = - 0.5*np.log(2*np.pi*predictive_variance_X**2) - (array1-Xpred)**2/(2*predictive_variance_X**2)
    LL_X = - (array1 - Xpred) ** 2 / (2 * predictive_variance_X)#  ** 2)
    #
    # Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
    # noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
    # Ypred, Yvar = gpy.predict(Times_pred_2,Y_metadata=noise_dict2)
    # predictive_variance_Y = Yvar + gp.mixed_noise.Gaussian_noise_1.variance**2
    # # LL_Y = - 0.5*np.log(2*np.pi*predictive_variance_Y**2) - (array2-Ypred)**2/(2*predictive_variance_Y**2)
    LL_Y = - (array2 - Ypred) ** 2 / (2 * predictive_variance_Y) #  ** 2)
    #
    # Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)+1), axis=1)
    # noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
    # Zpred, Zvar = gpz.predict(Times_pred_3,Y_metadata=noise_dict3)
    # predictive_variance_Z = Zvar + gp.mixed_noise.Gaussian_noise_2.variance**2
    # # LL_Z = - 0.5*np.log(2*np.pi*predictive_variance_Z**2) - (array3-Zpred)**2/(2*predictive_variance_Z**2)
    LL_Z = - (array3 - Zpred) ** 2 / (2 * predictive_variance_Z) #  ** 2)


    if plot_cost:
        fig=plt.figure()
        plt.scatter(X_reshaped,LL_X, marker='x', label=r"Log Loss for X-axis")
        plt.scatter(X_reshaped,LL_Y, marker='x', label=r"Log Loss for Y-axis")
        plt.scatter(X_reshaped,LL_Z, marker='x', label=r"Log Loss for Z-axis")
        fig.legend()
        fig.show()
    cost = combining_axis_cost(LL_X,LL_Y,LL_Z)
    return cost

def train_GPs_on_position_acceleration(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False, deltat=0.01):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)
    IMPORTANT: ALL INPUTS MUST HAVE SIZE GREATER THAN 3!!!!!!"""
    # get list of Xs that line up with inputs and outputs and are limited to them.
    # for each input:
    #     train a GP
    #     compare actual outputs to predicted using defined function for MSLL
    #     store MSLL in array size that is the same size as the inputs and the outputs.
    # For the last line, if you do it for all inputs, can end up with a square array of inputs to outputs
    # Then I need some method of choosing the maximum combination of inputs and outputs. Research this...

    cost_matrix = np.zeros((len(list_of_input_trajectories),len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(6.25, 2.8))
        outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3, bottom=0.15)

        left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])

        ax = fig.add_subplot(left_cell[:, :])
        right_cell = outer_grid[1].subgridspec(5, 3, hspace=0.05)
        upper_right_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_cell[:5, :], hspace=0.0)
        # lower_right_cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_cell[3:, :], hspace=0.0)
        # upper_right_cell = right_Cell[:3, :].subgridspec(3, 1)
        # lower_right_cell = right_Cell[3:, :].subgridspec(2, 1)

        # axx = fig.add_subplot(right_cell[0, :])
        # axy = fig.add_subplot(right_cell[1, :])
        # axz = fig.add_subplot(right_cell[2, :])
        # ax2 = fig.add_subplot(right_cell[3, :])
        # ax3 = fig.add_subplot(right_cell[4, :])
        axx = fig.add_subplot(upper_right_cell[0])
        axy = fig.add_subplot(upper_right_cell[1], sharex=axx)
        axz = fig.add_subplot(upper_right_cell[2], sharex=axx)
        # ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
        # ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)

        ax.set_xlabel('Z (m)')
        ax.set_ylabel('X (m)')

        axins = ax.inset_axes([0.3, 0.6, 0.375, 0.35])   #Bottom left: [0.175, 0.15, 0.375, 0.35]; Top Left: [0.175, 0.6, 0.375, 0.35]
        axins.set_xlabel('Z (m)')
        axins.set_ylabel('Y (m)')

        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        # ax2.tick_params(axis="both", labelbottom=False)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        # axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        axz.set_xlabel('Time (s)')
        # ax3.set_ylabel('Incorrect\nMatching')
        # ax2.set_ylabel('Correct\nMatching')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_output_trajectories))]
        alpha = 0.1
        alpha2 = 1

        for i in range(len(list_of_output_trajectories)):
            output_trajectory_masked = list_of_output_trajectories[i]  # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3, -1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.ma.getmaskarray(output_trajectory_masked)[0]
            # times_output_mask = np.zeros_like(times_array, dtype=bool)
            # times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            # times_output_mask[:np.amin(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0]))] = True
            # times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                output_times = output_times[:output_length]

            ax.plot(output_trajectory[2, :], output_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(output_trajectory[2, :], output_trajectory[0, :], marker='1', color=colors[i])
            axins.plot(output_trajectory[2, :], output_trajectory[1, :], ls='-', color=colors[i])
            axx.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[0, :],
                        color=colors[i], marker='x', s=50 * (72. / fig.dpi) ** 2)
            axy.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[1, :],
                        color=colors[i], marker='1', s=50 * (72. / fig.dpi) ** 2)
            axz.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[2, :],
                        color=colors[i], marker='+', s=50 * (72. / fig.dpi) ** 2)

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3,-1))
        if len(input_trajectory[0, :])<3:
            return "", np.roll(np.array(range(len([array for array in list_of_input_trajectories if array.mask.all() == True]))), 1), np.nan
        velocity = differentiater(input_trajectory, timedelta=deltat)
        acceleration = differentiater(velocity, timedelta=deltat)
        X = acceleration[0, :, None]
        Y = acceleration[1, :, None]
        Z = acceleration[2, :, None]
        times_input_mask = input_mask[0,:]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        input_times = input_times[:-2, None]    # CHECK THIS BUT I THINK IT IS RIGHT.
        kernelx = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)#  * GPy.kern.sde_StdPeriodic(1)
        kernely = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)#  * GPy.kern.sde_StdPeriodic(1)
        kernelz = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)#  * GPy.kern.sde_StdPeriodic(1)
        # kernelx = GPy.kern.sde_StdPeriodic(1)
        # kernely = GPy.kern.sde_StdPeriodic(1)
        # kernelz = GPy.kern.sde_StdPeriodic(1)

        gpx = GPy.models.GPRegression(input_times, X, kernelx)
        gpy = GPy.models.GPRegression(input_times, Y, kernely)
        gpz = GPy.models.GPRegression(input_times, Z, kernelz)
        # print(gpx)
        gpx.rbf.lengthscale.unconstrain()
        gpy.rbf.lengthscale.unconstrain()
        gpz.rbf.lengthscale.unconstrain()
        gpx.rbf.lengthscale.constrain_bounded(0.1, 5.)#00)
        gpy.rbf.lengthscale.constrain_bounded(0.1, 5.)#0)
        gpz.rbf.lengthscale.constrain_bounded(0.1, 5.)#00)
        # gpx.mul.rbf.lengthscale.constrain_bounded(1., 5.)#00)
        # gpy.mul.rbf.lengthscale.constrain_bounded(1., 5.)#0)
        # gpz.mul.rbf.lengthscale.constrain_bounded(1., 5.)#00)
        # gpx.mul.std_periodic.period.constrain_bounded(0.3, 1.5)  # 00)
        # gpy.mul.std_periodic.period.constrain_bounded(0.3, 1.5)  # 0)
        # gpz.mul.std_periodic.period.constrain_bounded(0.3, 1.5)  # 00)
        # gpx.mul.std_periodic.lengthscale.constrain_bounded(0.3, 10.)  # 00)
        # gpy.mul.std_periodic.lengthscale.constrain_bounded(0.3, 10.)  # 0)
        # gpz.mul.std_periodic.lengthscale.constrain_bounded(0.3, 10.)  # 00)
        #
        # gpx.Gaussian_noise.variance.unconstrain()
        # gpy.Gaussian_noise.variance.unconstrain()
        # gpz.Gaussian_noise.variance.unconstrain()
        # gpx.Gaussian_noise.variance.constrain_fixed(0.1)
        # gpy.Gaussian_noise.variance.constrain_fixed(0.1)
        # gpz.Gaussian_noise.variance.constrain_fixed(0.1)


        if lengthscales or verbose:
            print(f"\nInput: {i}")
        # gpx.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        # gpy.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        # gpz.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        gpx.optimize_restarts(num_restarts=3, verbose=verbose)
        gpy.optimize_restarts(num_restarts=3, verbose=verbose)
        gpz.optimize_restarts(num_restarts=3, verbose=True)
    #     print(f"Cost: {gp.objective_function()}")
        if lengthscales:
            # print(f"X timescale: {gpx.rbf.lengthscale}")
            # print(f"Y timescale: {gpy.rbf.lengthscale}")
            # print(f"Z timescale: {gpz.rbf.lengthscale}")
            # print(f"X timescale: \n{gpx}")
            # gpx.plot()
            # print(f"Y timescale: \n{gpy}")
            # gpy.plot()
            # print(f"Z timescale: \n{gpz}")
            # gpz.plot()
            print(f"X timescale: \n{gpx}")
            print(f"Y timescale: \n{gpy}")
            print(f"Z timescale: \n{gpz}")
            plot_3outputs_independent(gpx, gpy, gpz, input_times, X,Y,Z)
            # TP.LML_landscape(input_times, X, gpx, noise_lower=-3., noise_upper=3., time_lower=-3., time_upper=3.)
            # TP.LML_landscape(input_times, Y, gpy, noise_lower=-3., noise_upper=3., time_lower=-3., time_upper=3.)
            # TP.LML_landscape(input_times, Z, gpz, noise_lower=0., noise_upper=3., time_lower=-3, time_upper=3, rbf_variance_lower=-10, rbf_variance_upper=1) # time_lower=np.log(0.1), time_upper=np.log(5),

        # FINDING INDIVIDUAL COSTS
        if plot_cost:
            ax.plot(input_trajectory[2, :], input_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(input_trajectory[2, :], input_trajectory[0, :], marker='+', color=colors[i])
            axins.plot(input_trajectory[2, :], input_trajectory[1, :], ls='-', color=colors[i])

            #################################################################

        output_trajectory_masked = max(list_of_output_trajectories, key=latest_unmasked_index)
        output_mask = np.ma.getmask(output_trajectory_masked)
        output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
        times_output_mask = np.zeros_like(times_array, dtype=bool)
        x2 = np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0])) # Check this !!!
        times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))-1] = True           #I think this is correct because need to predict acceleration for last TWO times of position.
        times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0]))-1:] = True        #I think this should be the correct size to ensure the acceleration has the correct number of steps and therefore, the position should finish in the right spot.
        times_output_masked = np.ma.masked_array(times_array, times_output_mask)
        output_times = np.array(times_output_masked[~times_output_mask])
            # duration = max(output_times)-min(output_times)
            # output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))
        if plot_cost:
            xlim = [min(output_times), max(output_times)] # (max(output_times)-min(output_times)) * 1.03 + min(output_times)]
            axx.set_xlim(xlim)


        X_reshaped = output_times[:, None]
        array1 = output_trajectory.T[:, 0, None]
        array2 = output_trajectory.T[:, 1, None]
        array3 = output_trajectory.T[:, 2, None]
            # Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
            # noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
            # Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
            #
            # Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
            # noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
            # Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
            #
            # Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
            # noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
            # Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)
        Xpred_a, Xvar_a = gpx.predict(X_reshaped)
        Ypred_a, Yvar_a = gpy.predict(X_reshaped)
        Zpred_a, Zvar_a = gpz.predict(X_reshaped)
        ###################################################################
        Xpred, Xvel = evolve(input_trajectory[0,-1], velocity[0,-1], Xpred_a[:, 0], deltat)
        Ypred, Yvel = evolve(input_trajectory[1, -1], velocity[1, -1], Ypred_a[:, 0], deltat)
        Zpred, Zvel = evolve(input_trajectory[2, -1], velocity[2, -1], Zpred_a[:, 0], deltat)
        Xpred_lower, _ = evolve(input_trajectory[0,-1], velocity[0,-1], Xpred_a[:, 0] - Xvar_a[:, 0] ** 0.5, deltat)
        Ypred_lower, _ = evolve(input_trajectory[1, -1], velocity[1, -1], Ypred_a[:, 0] - Yvar_a[:, 0] ** 0.5, deltat)
        Zpred_lower, _ = evolve(input_trajectory[2, -1], velocity[2, -1], Zpred_a[:, 0] - Zvar_a[:, 0] ** 0.5, deltat)
        Xpred_upper, _ = evolve(input_trajectory[0,-1], velocity[0,-1], Xpred_a[:, 0] + Xvar_a[:, 0] ** 0.5, deltat)
        Ypred_upper, _ = evolve(input_trajectory[1, -1], velocity[1, -1], Ypred_a[:, 0] + Yvar_a[:, 0] ** 0.5, deltat)
        Zpred_upper, _ = evolve(input_trajectory[2, -1], velocity[2, -1], Zpred_a[:, 0] + Zvar_a[:, 0] ** 0.5, deltat)
        Xpred, Xvel = Xpred[1:], Xvel[1:]
        Ypred, Yvel = Ypred[1:], Yvel[1:]
        Zpred, Zvel = Zpred[1:], Zvel[1:]
        Xpred_lower = Xpred_lower[1:]
        Ypred_lower = Ypred_lower[1:]
        Zpred_lower = Zpred_lower[1:]
        Xpred_upper = Xpred_upper[1:]
        Ypred_upper = Ypred_upper[1:]
        Zpred_upper = Zpred_upper[1:]


        if plot_cost:
            ax.plot(Zpred, Xpred, ls='--', color=colors[i], alpha=0.5)
            # axins.plot(Zpred, Ypred, ls='--', color=colors[i], alpha=0.5)
            axins.set_ylim([142, 148])
            axx.plot(output_times+0.2, Xpred, ls='--', color=colors[i], alpha=0.5)
            axy.plot(output_times+0.2, Ypred, ls='--', color=colors[i], alpha=0.5)
            axz.plot(output_times+0.2, Zpred, ls='--', color=colors[i], alpha=0.5)
            # axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axx.fill_between(x=output_times+0.2, y1=Xpred_lower, y2=Xpred_upper, color=colors[i], alpha=0.05)
            axy.fill_between(x=output_times+0.2, y1=Ypred_lower, y2=Ypred_upper, color=colors[i], alpha=0.05)
            axz.fill_between(x=output_times+0.2, y1=Zpred_lower, y2=Zpred_upper, color=colors[i], alpha=0.05)
            # plt.tight_layout()
        # final_index = np.amax(np.nonzero(~np.ma.getmaskarray(min(list_of_output_trajectories, key=np.ma.count_masked))[0]))     # Index corresponding to latest position information.
        # first_index = np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))                                      # Index corresponding to last position information of corresponding input.
        final_index = latest_unmasked_index(max(list_of_output_trajectories, key=latest_unmasked_index))     # Index corresponding to latest position information.
        first_index = latest_unmasked_index(input_trajectory_masked)+1      # Index corresponding to first unknown position information.
        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            prediction_output_mask = output_mask[0,first_index:final_index+1]
            Xpred_a_masked = np.ma.masked_array(Xpred, prediction_output_mask)
            Ypred_a_masked = np.ma.masked_array(Ypred, prediction_output_mask)
            Zpred_a_masked = np.ma.masked_array(Zpred, prediction_output_mask)
            Xpred_lower_masked = np.ma.masked_array(Xpred_lower, prediction_output_mask)
            Ypred_lower_masked = np.ma.masked_array(Ypred_lower, prediction_output_mask)
            Zpred_lower_masked = np.ma.masked_array(Zpred_lower, prediction_output_mask)
            Xpred_upper_masked = np.ma.masked_array(Xpred_upper, prediction_output_mask)
            Ypred_upper_masked = np.ma.masked_array(Ypred_upper, prediction_output_mask)
            Zpred_upper_masked = np.ma.masked_array(Zpred_upper, prediction_output_mask)
            Xpred_a_prime = np.array(Xpred_a_masked[~prediction_output_mask])
            Ypred_a_prime = np.array(Ypred_a_masked[~prediction_output_mask])
            Zpred_a_prime = np.array(Zpred_a_masked[~prediction_output_mask])
            Xpred_lower_prime = np.array(Xpred_lower_masked[~prediction_output_mask])
            Ypred_lower_prime = np.array(Ypred_lower_masked[~prediction_output_mask])
            Zpred_lower_prime = np.array(Zpred_lower_masked[~prediction_output_mask])
            Xpred_upper_prime = np.array(Xpred_upper_masked[~prediction_output_mask])
            Ypred_upper_prime = np.array(Ypred_upper_masked[~prediction_output_mask])
            Zpred_upper_prime = np.array(Zpred_upper_masked[~prediction_output_mask])
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                output_trajectory = output_trajectory[:, :output_length]
                Xpred_a_prime = Xpred_a_prime[:output_length]
                Ypred_a_prime = Ypred_a_prime[:output_length]
                Zpred_a_prime = Zpred_a_prime[:output_length]
                Xpred_upper_prime = Xpred_upper_prime[:output_length]
                Ypred_upper_prime = Ypred_upper_prime[:output_length]
                Zpred_upper_prime = Zpred_upper_prime[:output_length]
                Xpred_lower_prime = Xpred_lower_prime[:output_length]
                Ypred_lower_prime = Ypred_lower_prime[:output_length]
                Zpred_lower_prime = Zpred_lower_prime[:output_length]
            times_output_mask = output_mask[0,:]                #
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                output_times = output_times[:output_length]

            cost_matrix[i,j] = individual_cost_function_acceleration(Xpred_a_prime, Ypred_a_prime, Zpred_a_prime, Xpred_upper_prime, Ypred_upper_prime, Zpred_upper_prime, output_trajectory, combined_axis_mean, output_times)# plot_cost)
    if print_cost_matrix:
        print(cost_matrix)
    row_ind, col_ind, cost = combined_costs(cost_matrix)
    # print(f"The row indices are: {row_ind}")
    # print(f"The column indices are: {col_ind}")
    list_of_masked_times = []
    list_of_connected_trajectories = []
    trajectories = np.zeros((len(list_of_input_trajectories),3,np.shape(list_of_input_trajectories[0])[-1]))
    large_mask = np.zeros((len(list_of_input_trajectories),3, np.shape(list_of_input_trajectories[0])[-1]),dtype=bool)
    for i, row_index in enumerate(row_ind):
        col_index = col_ind[i]
        output_mask = np.ma.getmask(list_of_output_trajectories[col_index])
        input_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        trajectories[i, ~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        trajectories[i, ~input_mask] = list_of_input_trajectories[row_index][~input_mask]
        # list_of_input_trajectories[row_index][~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        # new_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        new_mask = (input_mask==True) & (output_mask==True)
        times_new_mask = new_mask[0, :]
        list_of_masked_times.append(np.ma.masked_array(times_array, mask=times_new_mask))
        # list_of_connected_trajectories.append(list_of_input_trajectories[row_index])
        # trajectories[i,:,:] = list_of_input_trajectories[row_index]
        large_mask[i,:,:] = new_mask
    masked_trajectories = np.ma.array(trajectories, mask=large_mask)
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    # ax1.set_ylabel("Y")
    # ax1.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_xlabel("X")
    # ax3.set_ylabel("Y")
    # ax3.set_xlabel("Z")
    # for i in range(len(list_of_input_trajectories)):
    #     ax1.plot(masked_trajectories[i,0,:],masked_trajectories[i,1,:])
    #     ax2.plot(masked_trajectories[i, 0, :], masked_trajectories[i, 2, :])
    #     ax3.plot(masked_trajectories[i, 2, :], masked_trajectories[i, 1, :])
    # fig.show()
    return masked_trajectories, col_ind, cost, row_ind


def individual_cost_function_acceleration(Xpred_a_prime, Ypred_a_prime, Zpred_a_prime, Xpred_upper_prime, Ypred_upper_prime, Zpred_upper_prime, output_trajectory, combining_axis_cost, output_times):
    """Calculate the cost function for a given input and output. Suggested cost function is the MSLL """

    array1 = output_trajectory.T[:, 0]
    array2 = output_trajectory.T[:, 1]
    array3 = output_trajectory.T[:, 2]

    Xpred, Xvar = Xpred_a_prime, Xpred_upper_prime-Xpred_a_prime
    Ypred, Yvar = Ypred_a_prime, Ypred_upper_prime-Ypred_a_prime
    Zpred, Zvar = Zpred_a_prime, Zpred_upper_prime-Zpred_a_prime
    predictive_variance_X = Xvar
    predictive_variance_Y = Yvar
    predictive_variance_Z = Zvar
    if len(array1)!=len(Xpred):
        cost = -100 # Above if statement only occurs if the output starts before the input has finished, which breaks it, so assign very large cost which is appropriate.
        return cost
    # Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)-1), axis=1)
    # noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
    # Xpred, Xvar = gpx.predict(Times_pred_1,Y_metadata=noise_dict1)
    # predictive_variance_X = Xvar + gp.mixed_noise.Gaussian_noise_0.variance**2
    # # LL_X = - 0.5*np.log(2*np.pi*predictive_variance_X**2) - (array1-Xpred)**2/(2*predictive_variance_X**2)
    LL_X = - (array1 - Xpred) ** 2 / (predictive_variance_X  ** 2)
    #
    # Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
    # noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
    # Ypred, Yvar = gpy.predict(Times_pred_2,Y_metadata=noise_dict2)
    # predictive_variance_Y = Yvar + gp.mixed_noise.Gaussian_noise_1.variance**2
    # # LL_Y = - 0.5*np.log(2*np.pi*predictive_variance_Y**2) - (array2-Ypred)**2/(2*predictive_variance_Y**2)
    LL_Y = - (array2 - Ypred) ** 2 / (predictive_variance_Y  ** 2)
    #
    # Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)+1), axis=1)
    # noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
    # Zpred, Zvar = gpz.predict(Times_pred_3,Y_metadata=noise_dict3)
    # predictive_variance_Z = Zvar + gp.mixed_noise.Gaussian_noise_2.variance**2
    # # LL_Z = - 0.5*np.log(2*np.pi*predictive_variance_Z**2) - (array3-Zpred)**2/(2*predictive_variance_Z**2)
    LL_Z = - (array3 - Zpred) ** 2 / (predictive_variance_Z  ** 2)
    cost = np.mean([np.mean(LL_X), np.mean(LL_Y), np.mean(LL_Z)])
    # fig=plt.figure()
    # plt.scatter(output_times,LL_X, marker='x', label=r"Log Loss for X-axis")
    # plt.scatter(output_times,LL_Y, marker='x', label=r"Log Loss for Y-axis")
    # plt.scatter(output_times,LL_Z, marker='x', label=r"Log Loss for Z-axis")
    # fig.legend()
    # fig.show()
    return cost


def differentiater(trajectories, timedelta=0.01):
    """Function to differentiate either positions or velocities with respect to time. (n_birds, n_axis, n_time_steps) or (n_axis, n_timesteps)"""
    number_of_array_axis = len(np.shape(trajectories))
    if number_of_array_axis == 2:
        velocities = (trajectories[:, 1:] - trajectories[:, :-1])/timedelta
    elif number_of_array_axis == 3:
        velocities = (trajectories[:, :, 1:] - trajectories[:, :, :-1]) / timedelta
    else:
        raise ValueError("The size of the trajectories array was not correct.")
    return velocities


def evolve(x2, v1, acceleration, dt):
    n_steps = len(acceleration)
    xs = np.zeros((n_steps + 1))
    vs = np.zeros((n_steps + 1))

    xs[0], vs[0] = x2, v1

    for i in range(n_steps):
        vs[i+1] = vs[i] + dt * acceleration[i]
        xs[i+1] = xs[i] + dt * vs[i+1]
    return xs, vs


def latest_unmasked_index(masked_array):
    a = np.ma.getmaskarray(masked_array)[0]
    b = ~a
    c = np.nonzero(b)
    d = np.amax(c)
    return d


def edge_case_finder(list_of_input_lists, list_of_output_lists, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, swapping_IO=False, fraction=0.1, trajectory_title="Trajectory", threshold_distance=1.32):
    assert(len(list_of_input_lists)==len(list_of_output_lists))

    for i in range(len(list_of_input_lists)):
        if i%5 == 0:
            print(i)

        coregionalised_success = True
        not_coregionalised_success = True
        acceleration_success = True
        coregionalised_file_code = "S"
        not_coregionalised_file_code = "S"
        acceleration_file_code = "S"

        average_input_length = statistics.median(map(np.ma.count_masked, list_of_input_lists[i]))# sum(map(np.ma.count_masked, list_of_input_lists[i])) / len(list_of_input_lists)  #ACTUALLY COUNTS MASK SO INVERSE INPUT LENGTH
        average_output_length = statistics.median(map(np.ma.count_masked, list_of_output_lists[i]))# sum(map(np.ma.count_masked, list_of_output_lists[i])) / len(list_of_output_lists)   # SEE ABOVE!!
        switchingIO = False
        if swapping_IO and average_input_length>=average_output_length:
            switchingIO = True
            input_list = list_of_output_lists[i]
            output_list = list_of_input_lists[i]
            input_length = np.shape(list_of_input_lists[i][0])[1] - average_output_length / 3
            output_length = np.shape(list_of_input_lists[i][0])[1] - average_input_length / 3

        else:
            input_list = list_of_input_lists[i]
            output_list = list_of_output_lists[i]
            input_length = np.shape(list_of_input_lists[i][0])[1] - average_input_length / 3
            output_length = np.shape(list_of_input_lists[i][0])[1] - average_output_length / 3

        if not (any(array.mask.all()==True for array in input_list) or any(array.mask.all() for array in output_list)):
            # total_obscurations += sum(list_of_occlusion_lists[i])

            _, col_inda, cost, row_inda = train_GPs_on_position(input_list, output_list, times_array, n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales, print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, switchingIO=switchingIO)

            _, col_indb, cost, row_indb = train_GPs_on_position_1D(input_list, output_list, times_array, n_restarts=n_restarts,
                                                     verbose=verbose, lengthscales=lengthscales,
                                                     print_cost_matrix=print_cost_matrix,
                                                     adjusting_output_len=adjusting_output_len,
                                                     output_max_length=output_max_length, switchingIO=switchingIO)
            input_list_length=len(input_list)
            input_list = [array for array in input_list if np.shape(list_of_input_lists[i][0])[1]-np.ma.count_masked(array)/3>=3]
            # acceleration_data_too_short += input_list_length-len(input_list)
            if len(input_list)==0:
                print(f"{i}: Input trajectory too short")
            else:
                _, col_indc, cost, row_indc = train_GPs_on_position_acceleration(input_list, output_list, times_array,
                                                   n_restarts=3, verbose=verbose, lengthscales=lengthscales,
                                                   print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len,
                                                   output_max_length=output_max_length, plot_cost=False, switchingIO=False,
                                                   deltat=0.01/fraction)

            if np.all(col_inda != row_inda):
                coregionalised_success=False
                coregionalised_file_code = "F"
            if np.all(col_indb != row_indb):
                not_coregionalised_success = False
                not_coregionalised_file_code = "F"
            if len(input_list)==0:
                acceleration_success = False
                acceleration_file_code = "N"
            elif np.all(col_indc != row_indc):
                acceleration_success = False
                acceleration_file_code = "F"

            # if not(coregionalised_success and not_coregionalised_success and acceleration_success) and not(coregionalised_success==False and not_coregionalised_success==False and acceleration_success==False):
            input_filename = f"{threshold_distance}TD_{trajectory_title}_{i}_input"
            output_filename = f"{threshold_distance}TD_{trajectory_title}_{i}_output"
            TP.array_save(list_of_input_lists[i], filename=input_filename, folder_save_location=f"Edge Cases1\\A{acceleration_file_code}_C{coregionalised_file_code}_N{not_coregionalised_file_code}")
            TP.array_save(list_of_output_lists[i], filename=output_filename, folder_save_location=f"Edge Cases1\\A{acceleration_file_code}_C{coregionalised_file_code}_N{not_coregionalised_file_code}")
            TP.array_save(times_array, "Times_array", f"Edge Cases1\\A{acceleration_file_code}_C{coregionalised_file_code}_N{not_coregionalised_file_code}")

    #     elif not (len([array for array in input_list if not(array.mask.all()==True)])==0 or len([array for array in output_list if not(array.mask.all()==True)])==0):
    #         Ninput_list = []#[array for array in input_list if array.mask.all() == False]
    #         Noutput_list = []#[array for array in output_list if array.mask.all() == False]
    #         input_index_test = []
    #         output_index_test = []
    #         for input_index, array in enumerate(input_list):
    #             if array.mask.all() == False:
    #                 Ninput_list.append(array)
    #                 input_index_test.append(input_index)
    #         for output_index, array in enumerate(output_list):
    #             if array.mask.all() == False:
    #                 Noutput_list.append(array)
    #                 output_index_test.append(output_index)
    #
    #         _, col_inda, cost, row_inda = train_GPs_on_position(Ninput_list, Noutput_list, times_array, n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales, print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len, output_max_length=output_max_length, switchingIO=switchingIO)
    #
    #         _, col_indb, cost, row_indb = train_GPs_on_position_1D(Ninput_list, Noutput_list, times_array, n_restarts=n_restarts,
    #                                                  verbose=verbose, lengthscales=lengthscales,
    #                                                  print_cost_matrix=print_cost_matrix,
    #                                                  adjusting_output_len=adjusting_output_len,
    #                                                  output_max_length=output_max_length, switchingIO=switchingIO)
    #
    #         Ninput_list_length=len(Ninput_list)
    #         input_index_test = [index for array, index in zip(Ninput_list, input_index_test) if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]  # range(len(Ninput_list))
    #         Ninput_list = [array for array in input_list if np.shape(input_list[0])[1]-np.ma.count_masked(array)/3>=3]
    #         if len(Ninput_list)==0:
    #             print(f"{i}: Input trajectory too short")
    #         else:
    #             _, col_indc, cost, row_indc = train_GPs_on_position_acceleration(Ninput_list, Noutput_list, times_array,
    #                                                n_restarts=n_restarts, verbose=verbose, lengthscales=lengthscales,
    #                                                print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len,
    #                                                output_max_length=output_max_length, plot_cost=False, switchingIO=False,
    #                                                deltat=0.01/fraction)
    #
    #
    #         common_inputs_outputs = list(set(input_index_test).intersection(output_index_test))
    #         number_of_breakages += len(common_inputs_outputs)
    #         total_obscurations += sum([list_of_occlusion_lists[i][cio_index] for cio_index in common_inputs_outputs])
    #         for row_index, col_index in zip(row_ind,col_ind):
    #             if input_index_test[row_index]==output_index_test[col_index]:
    #                 total_obscurations_assigned_correctly += list_of_occlusion_lists[i][output_index_test[col_index]]
    #                 successful_assignments += 1
    #                 list_of_assignments.append(output_index_test[col_index])
    #     else:
    #         obscured_assignment_problems += 1
    # # if total_assignments!=0:
    # #     successful_combination_percentage = 100*successful_combination/total_assignments
    # # else:
    # #     successful_combination_percentage = np.nan
    # # print(f"Rate of Successful Recombinations: {successful_combination_percentage}%")
    # print(obscured_assignment_problems)
    # print(len(list_of_input_lists)-total_assignments-total_partial_assignments)
    # # assert(obscured_assignment_problems == len(list_of_input_lists)-total_assignments-total_partial_assignments)
    # if len(list_of_input_lists)!=0:
    #     average_number_of_input_positions = number_of_input_positions/len(list_of_input_lists)
    #     average_number_of_output_positions = number_of_output_positions/len(list_of_output_lists)
    # else:
    #     average_number_of_input_positions = np.nan
    #     average_number_of_output_positions = np.nan
    return


def edge_case_reconstructor(trajectories, threshold_distance, from_scratch=False, fraction=0.1, length=80., n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, swapping_IO=False, acceleration=False, trajectory_title="Trajectory"):
    """Putting it all together."""
    if acceleration:
        assert(swapping_IO == False)
    trajectories = array_fractional_reducer(trajectories, fraction, 2)
    times_array = np.linspace(0, length, np.shape(trajectories)[2])
    if from_scratch:
        broken_trajectories, trajectories_altered, trajectories_altered_between_times = break_trajectories_as_though_from_scratch(
            trajectories, threshold_distance)
    else:
        broken_trajectories, trajectories_altered, trajectories_altered_between_times = break_trajectories(trajectories, threshold_distance)
    list_of_input_lists, list_of_output_lists, list_of_occlusion_lists, trajectories = masker(broken_trajectories, trajectories_altered,
                                                           trajectories_altered_between_times)

    edge_case_finder(list_of_input_lists, list_of_output_lists, times_array, n_restarts=n_restarts, verbose=verbose,
                     lengthscales=lengthscales, print_cost_matrix=print_cost_matrix, adjusting_output_len=adjusting_output_len, output_max_length=output_max_length,
                     swapping_IO=swapping_IO, fraction=fraction, trajectory_title=trajectory_title, threshold_distance=threshold_distance)
    return


def plot_3outputs_coregionalised(X_List, Y_List, gp):
    """Takes as input an array (n_training_points, n_output_dimensions=3)"""
    alpha = 0.7
    marker = '.'

    input_times = X_List[0]
    print(np.shape(input_times))
    X, Y, Z = Y_List
    slices = GPy.util.multioutput.get_slices(X_List)
    Times_pred_1 = np.concatenate((input_times, np.ones_like(input_times) - 1), axis=1)
    noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
    Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
    Times_pred_2 = np.concatenate((input_times, np.ones_like(input_times)), axis=1)
    noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
    Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
    Times_pred_3 = np.concatenate((input_times, np.ones_like(input_times) + 1), axis=1)
    noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
    Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)


    fig = plt.figure(figsize=(6.25,2.))#  (9, 2))
    # outer_grid = gridspec.GridSpec(1, 21, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.23)
    outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=0.1, right=0.99, top=0.99, bottom=0.2)
    outer_grid = outer_grid[0, :18].subgridspec(2, 3, hspace=0.0, wspace=0.44)

    # outer_grid = gridspec.GridSpecFromSubplotSpec(1, :18, subplot_spec=outer_grid[0])
    # right_cell = outer_grid[1].subgridspec(2, 1, hspace=0.3)
    # outer_grid = gridspec.GridSpec(2, 7, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3, hspace=0.)
    axx = fig.add_subplot(outer_grid[0, 0])
    ax1 = fig.add_subplot(outer_grid[1, 0], sharex=axx)
    axy = fig.add_subplot(outer_grid[0, 1], sharex=axx)
    ax2 = fig.add_subplot(outer_grid[1, 1], sharex=axx, sharey=ax1)
    axz = fig.add_subplot(outer_grid[0, 2], sharex=axx)
    ax3 = fig.add_subplot(outer_grid[1, 2], sharex=axx, sharey=ax1)


    axx.xaxis.set_major_locator(MultipleLocator(1.))
    axy.xaxis.set_major_locator(MultipleLocator(1.))
    axz.xaxis.set_major_locator(MultipleLocator(1.))
    ax1.xaxis.set_major_locator(MultipleLocator(1.))
    ax2.xaxis.set_major_locator(MultipleLocator(1.))
    ax3.xaxis.set_major_locator(MultipleLocator(1.))
    axx.xaxis.set_minor_locator(AutoMinorLocator(2))
    axy.xaxis.set_minor_locator(AutoMinorLocator(2))
    axz.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
    axx.tick_params(axis="both", which='both',direction="in", bottom=True, top=True, left=True, right=True)
    axy.tick_params(axis="both", which='both',direction="in", bottom=True, top=True, left=True, right=True)
    axz.tick_params(axis="both", which='both',direction="in", bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis="both", which='both',direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="both", which='both',direction="in", bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis="both", which='both',direction="in", bottom=True, top=True, left=True, right=True)

    # ax2.tick_params(axis="both", labelbottom=False)
    axx.tick_params(axis="both", labelbottom=False)
    axy.tick_params(axis="both", labelbottom=False)
    axz.tick_params(axis="both", labelbottom=False)

    axx.set_ylabel('X (m)')
    axy.set_ylabel('Y (m)')
    axz.set_ylabel('Z (m)')
    ax1.set_xlabel('Time (s)')
    ax2.set_xlabel('Time (s)')
    ax3.set_xlabel('Time (s)')

    # left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])

    xlim  = [min(input_times)-0.1, max(input_times)+0.1]
    normalised = True

    times = np.linspace(xlim[0],xlim[1], 100)
    Times_pred_1_prime = np.concatenate((times, np.ones_like(times) - 1), axis=1)
    noise_dict1_prime = {'output_index': Times_pred_1_prime[:, 1:].astype(int)}
    Xpred_prime, Xvar_prime = gp.predict(Times_pred_1_prime, Y_metadata=noise_dict1_prime)
    Times_pred_2_prime = np.concatenate((times, np.ones_like(times)), axis=1)
    noise_dict2_prime = {'output_index': Times_pred_2_prime[:, 1:].astype(int)}
    Ypred_prime, Yvar_prime = gp.predict(Times_pred_2_prime, Y_metadata=noise_dict2_prime)
    Times_pred_3_prime = np.concatenate((times, np.ones_like(times) + 1), axis=1)
    noise_dict3_prime = {'output_index': Times_pred_3_prime[:, 1:].astype(int)}
    Zpred_prime, Zvar_prime = gp.predict(Times_pred_3_prime, Y_metadata=noise_dict3_prime)
    std_scaled = 10.
    lower_error = ((1-erf(std_scaled/np.sqrt(2)))/2)*100
    upper_error = 100-lower_error
    print(lower_error)
    print(upper_error)
    # axx.set_ylabel('X')
    # gpx.plot(plot_limits=xlim, ax=axx, legend=False, plot_data=False)  # , color='r', marker='x')
    # gpy.plot(plot_limits=xlim, ax=axy, legend=False, plot_data=False)  # , color='b', marker='1')
    # gpz.plot(plot_limits=xlim, ax=axz, legend=False, plot_data=False)  # , color='g', marker='+')
    gp.plot_mean(plot_limits=xlim, ax=axx, fixed_inputs=[(1,0)], alpha=0.3, color='k', linestyle='-', linewidth=1.)
    gp.plot_mean(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], alpha=0.3, color='k', linestyle='-', linewidth=1.)
    gp.plot_mean(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], alpha=0.3, color='k', linestyle='-', linewidth=1.)

    scaling_factor = 100.
    scaling_factor1 = 10.
    axx.fill_between(times.squeeze(), (Xpred_prime-scaling_factor*Xvar_prime**0.5).squeeze(), (Xpred_prime+scaling_factor*Xvar_prime**0.5).squeeze(), color='k', alpha=0.1)
    axy.fill_between(times.squeeze(), (Ypred_prime - scaling_factor1 * Yvar_prime ** 0.5).squeeze(),
                     (Ypred_prime + scaling_factor1 * Yvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)
    axz.fill_between(times.squeeze(), (Zpred_prime - scaling_factor * Zvar_prime ** 0.5).squeeze(),
                     (Zpred_prime + scaling_factor * Zvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)

    # gp.plot_confidence(lower=lower_error,upper=upper_error, plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)],
    #               alpha=0.1, color='grey')

    # gp.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1,0)], which_data_rows=slices[0],  legend=False, plot_data=False, alpha=0.1)
    # gp.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1,1)], which_data_rows=slices[1],  legend=False, plot_data=False)
    # gp.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1,2)], which_data_rows=slices[2],  legend=False, plot_data=False)
    axx.scatter(input_times, X, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
    axy.scatter(input_times, Y, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
    axz.scatter(input_times, Z, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
    axx.set_xlim(xlim)
    axy.set_xlim(xlim)
    axz.set_xlim(xlim)
    axx.set_ylim([axx.get_ylim()[0] - 2, axx.get_ylim()[1] + 1])
    axy.set_ylim([axy.get_ylim()[0] - 0.1, axy.get_ylim()[1] + 0.1])
    axz.set_ylim([axz.get_ylim()[0] - 1, axz.get_ylim()[1] + 1])
    # his.xticks(numpy.arange(1,5), numpy.arange(1,5))
    # his.set_xticks(numpy.arange(0,5))

    if normalised:
        ax1.set_ylabel('Norm. Res.')
        ax2.set_ylabel('Norm. Res.')
        ax3.set_ylabel('Norm. Res.')
        ax1.scatter(input_times, (X - Xpred) / Xvar ** 0.5, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        ax2.scatter(input_times, (Y - Ypred) / Yvar ** 0.5, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        ax3.scatter(input_times, (Z - Zpred) / Zvar ** 0.5, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        ylim1, ylim2 = -2.9, 2.9
        ylim = [ylim1, ylim2]
        ax1.set_ylim(ylim)
        binlocations = np.linspace(ylim1, ylim2, 10)
        alpha = 0.2
        # right_cell = outer_grid[0, 18:].subgridspec(2, 1, hspace=0.0)
        # axhis = fig.add_subplot(right_cell[1, 0], sharey=ax1)
        # axhis.tick_params(axis='both', direction='in', top=True, right=True, which='both', labelleft=False)
        # axhis.set_xlabel('Occurences')
        # colors = ['r', 'b', 'g']
        # hist_data = [(X - gpx.predict(input_times)[0]) / gpx.predict(input_times)[1] ** 0.5, (Y - gpy.predict(input_times)[0]) / gpy.predict(input_times)[1] ** 0.5, (Z - gpz.predict(input_times)[0])/gpz.predict(input_times)[1]**0.5]
        # axhis.hist(hist_data, binlocations, histtype='bar', color=colors)
        # axhis.hist((X - Xpred) / Xvar ** 0.5, bins=binlocations,
        #            orientation='horizontal', color='red', edgecolor='r', histtype='step', fill=True, alpha=alpha,
        #            linewidth=1.0)
        # axhis.hist((Y - Ypred) / Yvar ** 0.5, bins=binlocations,
        #            orientation='horizontal', color='blue', edgecolor='b', histtype='step', fill=True, alpha=alpha,
        #            linewidth=1.0)
        # axhis.hist((Z - Zpred) / Zvar, bins=binlocations,
        #            orientation='horizontal', color='green', edgecolor='g', histtype='step', fill=True,
        #            linewidth=1.0, alpha=alpha)
    else:
        ax1.scatter(input_times, X - gpx.predict(input_times)[0])
        ax2.scatter(input_times, Y - gpy.predict(input_times)[0])
        ax3.scatter(input_times, Z - gpz.predict(input_times)[0])
        ax1.set_ylabel('Residuals')
        ax2.set_ylabel('Residuals')
        ax3.set_ylabel('Residuals')
    # axy.set_ylabel('Y')

    # axz.set_ylabel('Z')

    # ax3.plot(X3, Y3, 'r,', mew=1.5)
    # xlim = [min(X_List[0])-2., max(X_List[0])+2.]
    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 8))
    # slices = GPy.util.multioutput.get_slices(X_List)
    # #Output 1
    # ax1.set_xlim(xlim)
    # ax1.set_title('X')
    # gp.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1,0)], which_data_rows=slices[0],  legend=False, plot_data=False)
    # # ax1.plot(X1,Y1,'r,',mew=1.5)
    # #Output 2
    # ax2.set_xlim(xlim)
    # ax2.set_title('Y')
    # gp.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1,1)], which_data_rows=slices[1],  legend=False, plot_data=False)
    # # ax2.plot(X2,Y2,'r,',mew=1.5)
    # # Output 3
    # ax3.set_xlim(xlim)
    # ax3.set_title('Z')
    # gp.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1,2)], which_data_rows=slices[2],  legend=False, plot_data=False)
    # ax3.plot(X3, Y3, 'r,', mew=1.5)
    plt.tight_layout()
    return


# def plot_3outputs_independent(X, gp,xlim):
#     """Takes as input an array (n_training_points, n_output_dimensions=3)"""
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 8))
#     slices = GPy.util.multioutput.get_slices([X, X, X])
#     #Output 1
#     ax1.set_xlim(xlim)
#     ax1.set_title('X')
#     gp.plot(plot_limits=xlim, ax=ax1, fixed_inputs=[(1,0)], which_data_rows=slices[0])
#     # ax1.plot(X1,Y1,'r,',mew=1.5)
#     #Output 2
#     ax2.set_xlim(xlim)
#     ax2.set_title('Y')
#     gp.plot(plot_limits=xlim, ax=ax2, fixed_inputs=[(1,1)], which_data_rows=slices[1])
#     # ax2.plot(X2,Y2,'r,',mew=1.5)
#     # Output 3
#     ax3.set_xlim(xlim)
#     ax3.set_title('Z')
#     gp.plot(plot_limits=xlim, ax=ax3, fixed_inputs=[(1,2)], which_data_rows=slices[2])
#     # ax3.plot(X3, Y3, 'r,', mew=1.5)
#     return
def plot_3outputs_independent(gpx, gpy, gpz, input_times, X,Y,Z):
    """Takes as input an array (n_training_points, n_output_dimensions=3)"""
    alpha=0.8
    marker='.'
    figsize = (6.25, 2.) #(9, 2.5)
    left = 0.1
    right = 0.99
    top = 0.99
    bottom = 0.2
    wspace1 = 0.42 # For all three, use 0.42. For just y, use 0.45
    wspace2 = 0.47
    # fig, ((axx, axy, axz),(ax1,ax2,ax3)) = plt.subplots(2, 3, sharex=True, figsize=(15, 8))
    acceleration = True
    fontsize = 12
    labelsize = 10
    no_residuals = False

    xlim = [min(input_times) - 0.1, max(input_times) + 0.1]
    times = np.linspace(xlim[0], 5.,200)#xlim[1], 200)
    a = times[:, None]
    Xpred_prime, Xvar_prime = gpx.predict(times)
    Ypred_prime, Yvar_prime = gpy.predict(times)
    Zpred_prime, Zvar_prime = gpz.predict(times)

    if acceleration:
        no_residuals = True
    if no_residuals == False:
        fig = plt.figure(figsize=figsize)
        # outer_grid = gridspec.GridSpec(1, 21, figure=fig, left=0.1, right=0.975, top=0.99, wspace=0.23, bottom=0.2)
        outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=left, right=right, top=top, bottom=bottom)
        outer_grid = outer_grid[0, :18].subgridspec(2, 3, hspace=0.0, wspace=wspace1)

        axx = fig.add_subplot(outer_grid[0, 0])
        ax1 = fig.add_subplot(outer_grid[1, 0], sharex=axx)
        axy = fig.add_subplot(outer_grid[0, 1], sharex=axx)
        ax2 = fig.add_subplot(outer_grid[1, 1], sharex=axx, sharey=ax1)
        axz = fig.add_subplot(outer_grid[0, 2], sharex=axx)
        ax3 = fig.add_subplot(outer_grid[1, 2], sharex=axx, sharey=ax1)
        axx.xaxis.set_major_locator(MultipleLocator(1.))
        axy.xaxis.set_major_locator(MultipleLocator(1.))
        axz.xaxis.set_major_locator(MultipleLocator(1.))
        ax1.xaxis.set_major_locator(MultipleLocator(1.))
        ax2.xaxis.set_major_locator(MultipleLocator(1.))
        ax3.xaxis.set_major_locator(MultipleLocator(1.))
        ax1.yaxis.set_major_locator(MultipleLocator(2.))
        ax2.yaxis.set_major_locator(MultipleLocator(2.))
        ax3.yaxis.set_major_locator(MultipleLocator(2.))
        axy.yaxis.set_major_locator(MultipleLocator(1.))
        axx.set_xlim(xlim)
        axy.set_xlim(xlim)
        axz.set_xlim(xlim)
        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axy.xaxis.set_minor_locator(AutoMinorLocator(2))
        axz.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.xaxis.set_minor_locator(AutoMinorLocator(2))
        axx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axy.yaxis.set_minor_locator(AutoMinorLocator(2))
        axz.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax3.yaxis.set_minor_locator(AutoMinorLocator(2))

        axx.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True,)
        axy.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True,)
        axz.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True,)
        ax1.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True,)
        ax2.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True,)
        ax3.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True,)

        # ax2.tick_params(axis="both", labelbottom=False)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        ax1.set_xlabel('Time (s)')
        ax2.set_xlabel('Time (s)')
        ax3.set_xlabel('Time (s)')



        normalised = True

        gpx.plot_mean(plot_limits=xlim, ax=axx, alpha=0.3, color='k', linestyle='-',
                      linewidth=1.)
        gpy.plot_mean(plot_limits=xlim, ax=axy, alpha=0.3, color='k', linestyle='-',
                      linewidth=1.)
        gpz.plot_mean(plot_limits=xlim, ax=axz, alpha=0.3, color='k', linestyle='-',
                      linewidth=1.)

        scaling_factor = 100.
        scaling_factor1 = 100.
        axy.set_ylim([143.3,145.])
        axx.fill_between(times.squeeze(), (Xpred_prime - scaling_factor * Xvar_prime ** 0.5).squeeze(),
                         (Xpred_prime + scaling_factor * Xvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)
        axy.fill_between(times.squeeze(), (Ypred_prime - scaling_factor1 * Yvar_prime ** 0.5).squeeze(),
                         (Ypred_prime + scaling_factor1 * Yvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)
        axz.fill_between(times.squeeze(), (Zpred_prime - scaling_factor * Zvar_prime ** 0.5).squeeze(),
                         (Zpred_prime + scaling_factor * Zvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)

        # gpx.plot(plot_limits=xlim, ax=axx, legend=False, plot_data=False, color='k')# , color='r', marker='x')
        # gpy.plot(plot_limits=xlim, ax=axy, legend=False, plot_data=False)#, color='b', marker='1')
        # gpz.plot(plot_limits=xlim, ax=axz, legend=False, plot_data=False)#, color='g', marker='+')
        axx.scatter(input_times, X, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        axy.scatter(input_times, Y, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        axz.scatter(input_times, Z, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        if normalised:
            ax1.set_ylabel('Norm. Res.')
            ax2.set_ylabel('Norm. Res.')
            ax3.set_ylabel('Norm. Res.')
            ax1.scatter(input_times, (X - gpx.predict(input_times)[0])/gpx.predict(input_times)[1]**0.5, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
            ax2.scatter(input_times, (Y - gpy.predict(input_times)[0])/gpy.predict(input_times)[1]**0.5, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
            ax3.scatter(input_times, (Z - gpz.predict(input_times)[0])/gpz.predict(input_times)[1]**0.5, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
            ylim1, ylim2 = -2.9, 2.9
            ylim = [ylim1, ylim2]
            ax1.set_ylim(ylim)
            binlocations = np.linspace(ylim1, ylim2, 10)
            alpha = 0.2
            axx.set_ylim([axx.get_ylim()[0]-2, axx.get_ylim()[1]+1])
            axy.set_ylim([axy.get_ylim()[0]-0.1, axy.get_ylim()[1]+0.1])
            axz.set_ylim([axz.get_ylim()[0]-1, axz.get_ylim()[1]+1])
            # right_cell = outer_grid[0, 18:].subgridspec(2, 1, hspace=0.0)
            # axhis = fig.add_subplot(right_cell[1, 0], sharey=ax1)
            # axhis.tick_params(axis='both', direction='in', top=True, right=True, which='both', labelleft=False)
            # axhis.set_xlabel('Occurences')
            # axhis.hist((X - gpx.predict(input_times)[0]) / gpx.predict(input_times)[1] ** 0.5, bins=binlocations,
            #            orientation='horizontal', color='red', edgecolor='r', histtype='step', fill=True, alpha=alpha,
            #            linewidth=1.0)
            # axhis.hist((Y - gpy.predict(input_times)[0]) / gpy.predict(input_times)[1] ** 0.5, bins=binlocations,
            #            orientation='horizontal', color='blue', edgecolor='b', histtype='step', fill=True, alpha=alpha,
            #            linewidth=1.0)
            # axhis.hist((Z - gpz.predict(input_times)[0])/gpz.predict(input_times)[1]**0.5, bins=binlocations, orientation='horizontal', color='green', edgecolor='g',histtype='step', fill=True,
            #            linewidth=1.0, alpha=alpha)
        else:
            ax1.scatter(input_times, X - gpx.predict(input_times)[0])
            ax2.scatter(input_times, Y - gpy.predict(input_times)[0])
            ax3.scatter(input_times, Z - gpz.predict(input_times)[0])
            ax1.set_ylabel('Residuals')
            ax2.set_ylabel('Residuals')
            ax3.set_ylabel('Residuals')
        # axy.set_ylabel('Y')

        # axz.set_ylabel('Z')


        # ax3.plot(X3, Y3, 'r,', mew=1.5)
    else:
        fig = plt.figure(figsize=(6.25, 1.75))
        outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=left, right=right, top=top, bottom=0.25)
        if acceleration:
            outer_grid = outer_grid[0, :18].subgridspec(1, 3, hspace=0.0, wspace=wspace2)
        else:
            outer_grid = outer_grid[0, :18].subgridspec(1, 3, hspace=0.0, wspace=wspace1)
        axx = fig.add_subplot(outer_grid[0, 0])
        # ax1 = fig.add_subplot(outer_grid[1, 0], sharex=axx)
        axy = fig.add_subplot(outer_grid[0, 1], sharex=axx)
        # ax2 = fig.add_subplot(outer_grid[1, 1], sharex=axx, sharey=ax1)
        axz = fig.add_subplot(outer_grid[0, 2], sharex=axx)
        # ax3 = fig.add_subplot(outer_grid[1, 2], sharex=axx, sharey=ax1)

        axx.xaxis.set_major_locator(MultipleLocator(1.))
        axy.xaxis.set_major_locator(MultipleLocator(1.))
        axz.xaxis.set_major_locator(MultipleLocator(1.))
        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True, labelsize=labelsize)
        axx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axy.xaxis.set_minor_locator(AutoMinorLocator(2))
        axz.xaxis.set_minor_locator(AutoMinorLocator(2))
        axx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axy.yaxis.set_minor_locator(AutoMinorLocator(2))
        axz.yaxis.set_minor_locator(AutoMinorLocator(2))

        axx.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True, )
        axy.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True, )
        axz.tick_params(axis="both", which='minor', direction="in", bottom=True, top=True, )
        axx.set_ylabel('Acc. in X (m s$^{-2}$)', fontsize=fontsize)
        axy.set_ylabel('Acc. in Y (m s$^{-2}$)', fontsize=fontsize)
        axz.set_ylabel('Acc. in Z (m s$^{-2}$)', fontsize=fontsize)
        axx.set_xlabel('Time (s)', fontsize=fontsize)
        axy.set_xlabel('Time (s)', fontsize=fontsize)
        axz.set_xlabel('Time (s)', fontsize=fontsize)

        xlim = [min(input_times) - 0.1, max(input_times) + 0.1]
        normalised = True
        gpz.plot_samples(plot_limits=xlim, samples=5, ax=axz, alpha=0.3, color='r', linestyle='-',
                      linewidth=1.)
        gpx.plot_mean(plot_limits=xlim, ax=axx, alpha=0.3, color='k', linestyle='-',
                      linewidth=1.)
        gpy.plot_mean(plot_limits=xlim, ax=axy, alpha=0.3, color='k', linestyle='-',
                      linewidth=1.)
        gpz.plot_mean(plot_limits=xlim, ax=axz, alpha=0.3, color='k', linestyle='-',
                      linewidth=1.)
        axx.set_xlim(xlim)
        axy.set_xlim(xlim)
        axz.set_xlim(xlim)
        scaling_factor = 1.
        scaling_factor1 = 1.
        axx.fill_between(times.squeeze(), (Xpred_prime - scaling_factor * Xvar_prime ** 0.5).squeeze(),
                         (Xpred_prime + scaling_factor * Xvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)
        axy.fill_between(times.squeeze(), (Ypred_prime - scaling_factor1 * Yvar_prime ** 0.5).squeeze(),
                         (Ypred_prime + scaling_factor1 * Yvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)
        axz.fill_between(times.squeeze(), (Zpred_prime - scaling_factor * Zvar_prime ** 0.5).squeeze(),
                         (Zpred_prime + scaling_factor * Zvar_prime ** 0.5).squeeze(), color='k', alpha=0.1)

        # gpx.plot(plot_limits=xlim, ax=axx, legend=False, plot_data=False, color='k')# , color='r', marker='x')
        # gpy.plot(plot_limits=xlim, ax=axy, legend=False, plot_data=False)#, color='b', marker='1')
        # gpz.plot(plot_limits=xlim, ax=axz, legend=False, plot_data=False)#, color='g', marker='+')
        axx.scatter(input_times, X, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        axy.scatter(input_times, Y, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        axz.scatter(input_times, Z, color='k', marker=marker, s=5 * (72. / fig.dpi) ** 2, alpha=alpha)
        #
        # gpx.plot(plot_limits=xlim, ax=axx, legend=False, plot_data=False)  # , color='r', marker='x')
        # gpy.plot(plot_limits=xlim, ax=axy, legend=False, plot_data=False)  # , color='b', marker='1')
        # gpz.plot(plot_limits=xlim, ax=axz, legend=False, plot_data=False)  # , color='g', marker='+')
        # axx.scatter(input_times, X, color='r', marker=marker, s=15 * (72. / fig.dpi) ** 2, alpha=alpha)
        # axy.scatter(input_times, Y, color='r', marker=marker, s=15 * (72. / fig.dpi) ** 2, alpha=alpha)
        # axz.scatter(input_times, Z, color='r', marker=marker, s=15 * (72. / fig.dpi) ** 2, alpha=alpha)
        # axx.set_ylim([axx.get_ylim()[0] - 2, axx.get_ylim()[1] + 1])
        # axy.set_ylim([axy.get_ylim()[0] - 0.1, axy.get_ylim()[1] + 0.1])
        # axz.set_ylim([axz.get_ylim()[0] - 1, axz.get_ylim()[1] + 1])
        # if normalised:
        #     ax1.set_ylabel('Norm. Res.')
        #     ax2.set_ylabel('Norm. Res.')
        #     ax3.set_ylabel('Norm. Res.')
        #     ax1.scatter(input_times, (X - gpx.predict(input_times)[0]) / gpx.predict(input_times)[1] ** 0.5, color='r',
        #                 marker='x')
        #     ax2.scatter(input_times, (Y - gpy.predict(input_times)[0]) / gpy.predict(input_times)[1] ** 0.5, color='b',
        #                 marker='1')
        #     ax3.scatter(input_times, (Z - gpz.predict(input_times)[0]) / gpz.predict(input_times)[1] ** 0.5, color='g',
        #                 marker='+')
        #     ylim1, ylim2 = -2.9, 2.9
        #     ylim = [ylim1, ylim2]
        #     ax1.set_ylim(ylim)
        #     binlocations = np.linspace(ylim1, ylim2, 10)
        #     alpha = 0.2
        #
        #     # right_cell = outer_grid[0, 18:].subgridspec(2, 1, hspace=0.0)
        #     # axhis = fig.add_subplot(right_cell[1, 0], sharey=ax1)
        #     # axhis.tick_params(axis='both', direction='in', top=True, right=True, which='both', labelleft=False)
        #     # axhis.set_xlabel('Occurences')
        #     # axhis.hist((X - gpx.predict(input_times)[0]) / gpx.predict(input_times)[1] ** 0.5, bins=binlocations,
        #     #            orientation='horizontal', color='red', edgecolor='r', histtype='step', fill=True, alpha=alpha,
        #     #            linewidth=1.0)
        #     # axhis.hist((Y - gpy.predict(input_times)[0]) / gpy.predict(input_times)[1] ** 0.5, bins=binlocations,
        #     #            orientation='horizontal', color='blue', edgecolor='b', histtype='step', fill=True, alpha=alpha,
        #     #            linewidth=1.0)
        #     # axhis.hist((Z - gpz.predict(input_times)[0])/gpz.predict(input_times)[1]**0.5, bins=binlocations, orientation='horizontal', color='green', edgecolor='g',histtype='step', fill=True,
        #     #            linewidth=1.0, alpha=alpha)
        # else:
        #     ax1.scatter(input_times, X - gpx.predict(input_times)[0])
        #     ax2.scatter(input_times, Y - gpy.predict(input_times)[0])
        #     ax3.scatter(input_times, Z - gpz.predict(input_times)[0])
        #     ax1.set_ylabel('Residuals')
        #     ax2.set_ylabel('Residuals')
        #     ax3.set_ylabel('Residuals')
        # axy.set_ylabel('Y')

        # axz.set_ylabel('Z')

        # ax3.plot(X3, Y3, 'r,', mew=1.5)
    plt.tight_layout()
    return



def summative_report_graph_plotter_coreg(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)"""
    # get list of Xs that line up with inputs and outputs and are limited to them.
    # for each input:
    #     train a GP
    #     compare actual outputs to predicted using defined function for MSLL
    #     store MSLL in array size that is the same size as the inputs and the outputs.
    # For the last line, if you do it for all inputs, can end up with a square array of inputs to outputs
    # Then I need some method of choosing the maximum combination of inputs and outputs. Research this...

    cost_matrix = np.zeros((len(list_of_input_trajectories),len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(6.25, 2.5))
        outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=0.12, right=0.99, top=0.99, bottom=0.2)
        outer_grid = outer_grid[0, :18].subgridspec(2, 3, hspace=0.0, wspace=0.45)

        axx = fig.add_subplot(outer_grid[0, 0])
        ax1 = fig.add_subplot(outer_grid[1, 0], sharex=axx)
        axy = fig.add_subplot(outer_grid[0, 1], sharex=axx)
        ax2 = fig.add_subplot(outer_grid[1, 1], sharex=axx)
        axz = fig.add_subplot(outer_grid[0, 2], sharex=axx)
        ax3 = fig.add_subplot(outer_grid[1, 2], sharex=axx)
        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        axz.set_xlabel('Time (s)')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_input_trajectories))]
        alpha = 0.1
        alpha2 = 1

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3,-1))
        times_input_mask = input_mask[0,:]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        # REFORMAT THE ARRAY TO BE SUITABLE FOR GPy
        Y_List = GPy_reformat_3D(input_trajectory) # make sure input_trajectory has shape (3, n_timesteps)
        X_List = GPy_reformat_3D(input_times) # times should have shape (n_timesteps)


        icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=3, kernel=GPy.kern.RBF(1))
        # print(icm)

        gp = GPy.models.GPCoregionalizedRegression(X_List, Y_List, kernel=icm)
        gp['.*rbf.var'].constrain_fixed(1.)  # For this kernel, B.kappa encodes the variance now.
        # gp['.*mixed_noise.Gaussian_noise_1.variance'].constrain_fixed(1e-6)
        if lengthscales or verbose:
            print(f"\nInput: {i}")
        gp.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
    #     print(f"Cost: {gp.objective_function()}")
        if lengthscales:
            # print(gp.ICM.rbf.lengthscale)
            print(gp)
            print(gp.ICM.B.W)
            print(gp.ICM.B.kappa)
            # print(gp.ICM.B)
            # print(gp.mixed_noise)
            plot_3outputs_coregionalised(X_List, Y_List, gp)

        # if i == 0:
            # print(list_of_output_trajectories[0][0,80:85])
            # print(list_of_output_trajectories[1][0, 80:85])
            # print(list_of_output_trajectories[2][0, 80:85])
            # print(list_of_output_trajectories[3][0, 80:85])
            # print(list_of_output_trajectories[4][0, 80:85])

        # FINDING INDIVIDUAL COSTS
        if plot_cost:
            output_trajectory_masked = list_of_output_trajectories[i] # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]
            axx.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[0, :], color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2, alpha=0.7)
            axy.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[1, :], color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2, alpha=0.7)
            axz.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[2, :], color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2, alpha=0.7)


            output_trajectory_masked = min(list_of_output_trajectories, key=np.ma.count_masked)
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            duration = max(output_times)-min(output_times)
            output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))

            xlim = [min(output_times), max(output_times)]
            axx.set_xlim(xlim)

            X_reshaped = output_times[:, None]
            array1 = output_trajectory.T[:, 0, None]
            array2 = output_trajectory.T[:, 1, None]
            array3 = output_trajectory.T[:, 2, None]
            Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
            noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
            Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)

            Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
            noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
            Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)

            Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
            noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
            Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)


            axx.plot(output_times, Xpred, ls='--', color=colors[i], alpha=0.3)
            axy.plot(output_times, Ypred, ls='--', color=colors[i], alpha=0.3)
            axz.plot(output_times, Zpred, ls='--', color=colors[i], alpha=0.3)
            # axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axy.fill_between(x=output_times, y1=Ypred[:, 0] - Yvar[:, 0] ** 0.5, y2=Ypred[:, 0] + Yvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axz.fill_between(x=output_times, y1=Zpred[:, 0] - Zvar[:, 0] ** 0.5, y2=Zpred[:, 0] + Zvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            # plt.tight_layout()

        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = output_mask[0,:]
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]
            if i==0 and j==0:
                color1=colors[i]
                X_reshaped = output_times[:, None]
                Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
                noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
                Xpred1, Xvar1 = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
                Xstd1 = Xvar1**0.5
                X1 = output_trajectory.T[:, 0, None]
                Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
                noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
                Ypred1, Yvar1 = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
                Ystd1 = Yvar1**0.5
                Y1 = output_trajectory.T[:, 1, None]
                Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
                noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
                Zpred1, Zvar1 = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
                Zstd1 = Zvar1**0.5
                Z1 = output_trajectory.T[:, 2, None]
            if i==1 and j==1:
                color2=colors[i]
                X_reshaped = output_times[:, None]
                Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
                noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
                Xpred2, Xvar2 = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
                Xstd2 = Xvar2 ** 0.5
                X2 = output_trajectory.T[:, 0, None]
                Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
                noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
                Ypred2, Yvar2 = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
                Ystd2 = Yvar2 ** 0.5
                Y2 = output_trajectory.T[:, 1, None]
                Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
                noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
                Zpred2, Zvar2 = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
                Zstd2 = Zvar2 ** 0.5
                Z2 = output_trajectory.T[:, 2, None]
            cost_matrix[i,j] = individual_cost_function(gp, output_trajectory, output_times, combined_axis_mean, plot_cost=False)# plot_cost)
    if plot_cost:
        axx.xaxis.set_major_locator(MultipleLocator(0.5))
        axx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axy.yaxis.set_minor_locator(AutoMinorLocator(2))
        axz.yaxis.set_minor_locator(AutoMinorLocator(2))
        axx.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axis_NR_plotter(ax1, output_times, Xpred1, Xpred2, Xstd1, Xstd2, X1, X2, color1=color1, color2=color2, ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
        axis_NR_plotter(ax2, output_times, Ypred1, Ypred2, Ystd1, Ystd2, Y1, Y2, color1=color1, color2=color2, ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
        axis_NR_plotter(ax3, output_times, Zpred1, Zpred2, Zstd1, Zstd2, Z1, Z2, color1=color1, color2=color2, ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
    if print_cost_matrix:
        print(cost_matrix)
    row_ind, col_ind, cost = combined_costs(cost_matrix)
    list_of_masked_times = []
    list_of_connected_trajectories = []
    trajectories = np.zeros((len(list_of_input_trajectories),3,np.shape(list_of_input_trajectories[0])[-1]))
    large_mask = np.zeros((len(list_of_input_trajectories),3, np.shape(list_of_input_trajectories[0])[-1]),dtype=bool)
    for i, row_index in enumerate(row_ind):
        col_index = col_ind[i]
        output_mask = np.ma.getmask(list_of_output_trajectories[col_index])
        input_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        trajectories[i, ~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        trajectories[i, ~input_mask] = list_of_input_trajectories[row_index][~input_mask]
        # list_of_input_trajectories[row_index][~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        # new_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        new_mask = (input_mask==True) & (output_mask==True)
        times_new_mask = new_mask[0, :]
        list_of_masked_times.append(np.ma.masked_array(times_array, mask=times_new_mask))
        # list_of_connected_trajectories.append(list_of_input_trajectories[row_index])
        # trajectories[i,:,:] = list_of_input_trajectories[row_index]
        large_mask[i,:,:] = new_mask
    masked_trajectories = np.ma.array(trajectories, mask=large_mask)
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    # ax1.set_ylabel("Y")
    # ax1.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_xlabel("X")
    # ax3.set_ylabel("Y")
    # ax3.set_xlabel("Z")
    # for i in range(len(list_of_input_trajectories)):
    #     ax1.plot(masked_trajectories[i,0,:],masked_trajectories[i,1,:])
    #     ax2.plot(masked_trajectories[i, 0, :], masked_trajectories[i, 2, :])
    #     ax3.plot(masked_trajectories[i, 2, :], masked_trajectories[i, 1, :])
    # fig.show()
    return masked_trajectories, col_ind, cost, row_ind


def summative_report_graph_plotter_ind(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)"""
    # get list of Xs that line up with inputs and outputs and are limited to them.
    # for each input:
    #     train a GP
    #     compare actual outputs to predicted using defined function for MSLL
    #     store MSLL in array size that is the same size as the inputs and the outputs.
    # For the last line, if you do it for all inputs, can end up with a square array of inputs to outputs
    # Then I need some method of choosing the maximum combination of inputs and outputs. Research this...

    cost_matrix = np.zeros((len(list_of_input_trajectories),len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(6.25, 2.5))
        outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=0.12, right=0.99, top=0.99, bottom=0.2)
        outer_grid = outer_grid[0, :18].subgridspec(2, 3, hspace=0.0, wspace=0.45)

        axx = fig.add_subplot(outer_grid[0, 0])
        ax1 = fig.add_subplot(outer_grid[1, 0], sharex=axx)
        axy = fig.add_subplot(outer_grid[0, 1], sharex=axx)
        ax2 = fig.add_subplot(outer_grid[1, 1], sharex=axx)
        axz = fig.add_subplot(outer_grid[0, 2], sharex=axx)
        ax3 = fig.add_subplot(outer_grid[1, 2], sharex=axx)
        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        axz.set_xlabel('Time (s)')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_input_trajectories))]
        alpha = 0.1
        alpha2 = 1

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3,-1))
        X = input_trajectory[0, :, None]
        Y = input_trajectory[1, :, None]
        Z = input_trajectory[2, :, None]
        times_input_mask = input_mask[0,:]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        input_times = input_times[:, None]
        kernelx = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        kernely = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        kernelz = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

        gpx = GPy.models.GPRegression(input_times, X, kernelx)
        gpy = GPy.models.GPRegression(input_times, Y, kernely)
        gpz = GPy.models.GPRegression(input_times, Z, kernelz)
        # gpx.rbf.lengthscale.unconstrain()
        # gpy.rbf.lengthscale.unconstrain()
        # gpz.rbf.lengthscale.unconstrain()
        # gpx.rbf.lengthscale.constrain_bounded(0.1, 100)
        # gpy.rbf.lengthscale.constrain_bounded(0.05, 10.)
        # gpz.rbf.lengthscale.constrain_bounded(0.1, 100)
        # gpy.Gaussian_noise.variance.unconstrain()
        noise = 1e-6
        # gpx.Gaussian_noise.variance.constrain_fixed(noise)
        gpy.Gaussian_noise.variance.constrain_fixed(noise)
        # gpz.Gaussian_noise.variance.constrain_fixed(noise)
        # gpy.Gaussian_noise.variance.constrain_bounded(1e-7, 1e-3)

        if lengthscales or verbose:
            print(f"\nInput: {i}")
        gpx.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        gpy.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        gpz.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
    #     print(f"Cost: {gp.objective_function()}")
        if lengthscales:
            # print(f"X timescale: {gpx.rbf.lengthscale}")
            # print(f"Y timescale: {gpy.rbf.lengthscale}")
            # print(f"Z timescale: {gpz.rbf.lengthscale}")
            print(f"X timescale: \n{gpx}")
            print(f"Y timescale: \n{gpy}")
            print(f"Z timescale: \n{gpz}")
            plot_3outputs_independent(gpx, gpy, gpz, input_times, X,Y,Z)

        # FINDING INDIVIDUAL COSTS
        if plot_cost:
            # ax.plot(input_trajectory[2, :], input_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(input_trajectory[2, :], input_trajectory[0, :], marker='+', color=colors[i])
            # axins.plot(input_trajectory[2, :], input_trajectory[1, :], ls='-', color=colors[i])

            output_trajectory_masked = list_of_output_trajectories[i] # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]

            # ax.plot(output_trajectory[2, :], output_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(output_trajectory[2, :], output_trajectory[0, :], marker='1', color=colors[i])
            # axins.plot(output_trajectory[2, :], output_trajectory[1, :], ls='-', color=colors[i])
            axx.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[0, :], color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2, alpha=0.7)
            axy.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[1, :], color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2, alpha=0.7)
            axz.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[2, :], color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2, alpha=0.7)


            output_trajectory_masked = min(list_of_output_trajectories, key=np.ma.count_masked)
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            duration = max(output_times)-min(output_times)
            output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))

            xlim = [min(output_times), max(output_times)]
            axx.set_xlim(xlim)

            X_reshaped = output_times[:, None]
            array1 = output_trajectory.T[:, 0, None]
            array2 = output_trajectory.T[:, 1, None]
            array3 = output_trajectory.T[:, 2, None]
            # Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
            # noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
            # Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
            #
            # Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
            # noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
            # Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
            #
            # Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
            # noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
            # Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)
            Xpred, Xvar = gpx.predict(X_reshaped)
            Ypred, Yvar = gpy.predict(X_reshaped)
            Zpred, Zvar = gpz.predict(X_reshaped)

            # ax.plot(Zpred, Xpred, ls='--', color=colors[i], alpha=0.5)
            # axins.plot(Zpred, Ypred, ls='--', color=colors[i], alpha=0.5)
            # axins.set_ylim([142, 148])
            axx.plot(output_times, Xpred, ls='--', color=colors[i], alpha=0.3)
            axy.plot(output_times, Ypred, ls='--', color=colors[i], alpha=0.3)
            axz.plot(output_times, Zpred, ls='--', color=colors[i], alpha=0.3)
            # axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axy.fill_between(x=output_times, y1=Ypred[:, 0] - Yvar[:, 0] ** 0.5, y2=Ypred[:, 0] + Yvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axz.fill_between(x=output_times, y1=Zpred[:, 0] - Zvar[:, 0] ** 0.5, y2=Zpred[:, 0] + Zvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axy.set_ylim([axy.get_ylim()[0], axy.get_ylim()[1] + 5])
            axy.yaxis.set_major_locator(MultipleLocator(175))
            # plt.tight_layout()
        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = output_mask[0,:]
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]
            if i==0 and j==0:
                color1=colors[i]
                X_reshaped = output_times[:, None]
                Xpred1, Xvar1 = gpx.predict(X_reshaped)
                Xstd1 = Xvar1**0.5
                X1 = output_trajectory.T[:, 0, None]
                Ypred1, Yvar1 = gpy.predict(X_reshaped)
                Ystd1 = Yvar1**0.5
                Y1 = output_trajectory.T[:, 1, None]
                Zpred1, Zvar1 = gpz.predict(X_reshaped)
                Zstd1 = Zvar1**0.5
                Z1 = output_trajectory.T[:, 2, None]
            if i==1 and j==1:
                color2=colors[i]
                X_reshaped = output_times[:, None]
                Xpred2, Xvar2 = gpx.predict(X_reshaped)
                Xstd2 = Xvar2 ** 0.5
                X2 = output_trajectory.T[:, 0, None]
                Ypred2, Yvar2 = gpy.predict(X_reshaped)
                Ystd2 = Yvar2 ** 0.5
                Y2 = output_trajectory.T[:, 1, None]
                Zpred2, Zvar2 = gpz.predict(X_reshaped)
                Zstd2 = Zvar2 ** 0.5
                Z2 = output_trajectory.T[:, 2, None]
            cost_matrix[i,j] = individual_cost_function_1D(gpx, gpy, gpz, output_trajectory, output_times, combined_axis_mean, plot_cost=False)
    if plot_cost:
        axx.xaxis.set_major_locator(MultipleLocator(0.5))
        axx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axy.yaxis.set_minor_locator(AutoMinorLocator(2))
        axz.yaxis.set_minor_locator(AutoMinorLocator(2))
        axx.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axis_NR_plotter(ax1, output_times, Xpred1, Xpred2, Xstd1, Xstd2, X1, X2, color1=color1, color2=color2,
                        ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
        axis_NR_plotter(ax2, output_times, Ypred1, Ypred2, Ystd1, Ystd2, Y1, Y2, color1=color1, color2=color2,
                        ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
        axis_NR_plotter(ax3, output_times, Zpred1, Zpred2, Zstd1, Zstd2, Z1, Z2, color1=color1, color2=color2,
                        ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)

    if print_cost_matrix:
        print(cost_matrix)
    row_ind, col_ind, cost = combined_costs(cost_matrix)
    # print(f"The row indices are: {row_ind}")
    # print(f"The column indices are: {col_ind}")
    list_of_masked_times = []
    list_of_connected_trajectories = []
    trajectories = np.zeros((len(list_of_input_trajectories),3,np.shape(list_of_input_trajectories[0])[-1]))
    large_mask = np.zeros((len(list_of_input_trajectories),3, np.shape(list_of_input_trajectories[0])[-1]),dtype=bool)
    for i, row_index in enumerate(row_ind):
        col_index = col_ind[i]
        output_mask = np.ma.getmask(list_of_output_trajectories[col_index])
        input_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        trajectories[i, ~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        trajectories[i, ~input_mask] = list_of_input_trajectories[row_index][~input_mask]
        # list_of_input_trajectories[row_index][~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        # new_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        new_mask = (input_mask==True) & (output_mask==True)
        times_new_mask = new_mask[0, :]
        list_of_masked_times.append(np.ma.masked_array(times_array, mask=times_new_mask))
        # list_of_connected_trajectories.append(list_of_input_trajectories[row_index])
        # trajectories[i,:,:] = list_of_input_trajectories[row_index]
        large_mask[i,:,:] = new_mask
    masked_trajectories = np.ma.array(trajectories, mask=large_mask)
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    # ax1.set_ylabel("Y")
    # ax1.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_xlabel("X")
    # ax3.set_ylabel("Y")
    # ax3.set_xlabel("Z")
    # for i in range(len(list_of_input_trajectories)):
    #     ax1.plot(masked_trajectories[i,0,:],masked_trajectories[i,1,:])
    #     ax2.plot(masked_trajectories[i, 0, :], masked_trajectories[i, 2, :])
    #     ax3.plot(masked_trajectories[i, 2, :], masked_trajectories[i, 1, :])
    # fig.show()
    return masked_trajectories, col_ind, cost, row_ind

def summative_report_graph_plotter_acc(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False, deltat=0.01):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)
    IMPORTANT: ALL INPUTS MUST HAVE SIZE GREATER THAN 3!!!!!!"""
    cost_matrix = np.zeros((len(list_of_input_trajectories), len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(6.25, 2.5))
        outer_grid = gridspec.GridSpec(1, 18, figure=fig, left=0.1, right=0.975, top=0.99, bottom=0.2)
        outer_grid = outer_grid[0, :18].subgridspec(2, 3, hspace=0.0, wspace=0.45)

        axx = fig.add_subplot(outer_grid[0, 0])
        ax1 = fig.add_subplot(outer_grid[1, 0], sharex=axx)
        axy = fig.add_subplot(outer_grid[0, 1], sharex=axx)
        ax2 = fig.add_subplot(outer_grid[1, 1], sharex=axx)
        axz = fig.add_subplot(outer_grid[0, 2], sharex=axx)
        ax3 = fig.add_subplot(outer_grid[1, 2], sharex=axx)
        axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axx.tick_params(axis="both", labelbottom=False)
        axy.tick_params(axis="both", labelbottom=False)
        axz.tick_params(axis="both", labelbottom=False)

        axx.set_ylabel('X (m)')
        axy.set_ylabel('Y (m)')
        axz.set_ylabel('Z (m)')
        axz.set_xlabel('Time (s)')

        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_input_trajectories))]
        alpha = 0.1
        alpha2 = 1

        for i in range(len(list_of_output_trajectories)):
            output_trajectory_masked = list_of_output_trajectories[
                i]  # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3, -1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.ma.getmaskarray(output_trajectory_masked)[0]
            # times_output_mask = np.zeros_like(times_array, dtype=bool)
            # times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            # times_output_mask[:np.amin(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0]))] = True
            # times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                output_times = output_times[:output_length]

            axx.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[0, :],
                        color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2)
            axy.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[1, :],
                        color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2)
            axz.scatter(output_times[len(output_times) - len(output_trajectory[0, :]):], output_trajectory[2, :],
                        color=colors[i], marker='.', s=15 * (72. / fig.dpi) ** 2)

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3, -1))
        if len(input_trajectory[0, :]) < 3:
            return "", np.roll(
                np.array(range(len([array for array in list_of_input_trajectories if array.mask.all() == True]))),
                1), np.nan
        velocity = differentiater(input_trajectory, timedelta=deltat)
        acceleration = differentiater(velocity, timedelta=deltat)
        X = acceleration[0, :, None]
        Y = acceleration[1, :, None]
        Z = acceleration[2, :, None]
        times_input_mask = input_mask[0, :]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        input_times = input_times[:-2, None]  # CHECK THIS BUT I THINK IT IS RIGHT.
        kernelx = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)  # * GPy.kern.sde_StdPeriodic(1)
        kernely = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)  # * GPy.kern.sde_StdPeriodic(1)
        kernelz = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)  # * GPy.kern.sde_StdPeriodic(1)
        # kernelx = GPy.kern.sde_StdPeriodic(1)
        # kernely = GPy.kern.sde_StdPeriodic(1)
        # kernelz = GPy.kern.sde_StdPeriodic(1)

        gpx = GPy.models.GPRegression(input_times, X, kernelx)
        gpy = GPy.models.GPRegression(input_times, Y, kernely)
        gpz = GPy.models.GPRegression(input_times, Z, kernelz)
        # print(gpx)
        gpx.rbf.lengthscale.unconstrain()
        gpy.rbf.lengthscale.unconstrain()
        gpz.rbf.lengthscale.unconstrain()
        gpx.rbf.lengthscale.constrain_bounded(0.1, 5.)  # 00)
        gpy.rbf.lengthscale.constrain_bounded(0.1, 5.)  # 0)
        gpz.rbf.lengthscale.constrain_bounded(0.1, 5.)  # 00)
        # gpx.mul.rbf.lengthscale.constrain_bounded(1., 5.)#00)
        # gpy.mul.rbf.lengthscale.constrain_bounded(1., 5.)#0)
        # gpz.mul.rbf.lengthscale.constrain_bounded(1., 5.)#00)
        # gpx.mul.std_periodic.period.constrain_bounded(0.3, 1.5)  # 00)
        # gpy.mul.std_periodic.period.constrain_bounded(0.3, 1.5)  # 0)
        # gpz.mul.std_periodic.period.constrain_bounded(0.3, 1.5)  # 00)
        # gpx.mul.std_periodic.lengthscale.constrain_bounded(0.3, 10.)  # 00)
        # gpy.mul.std_periodic.lengthscale.constrain_bounded(0.3, 10.)  # 0)
        # gpz.mul.std_periodic.lengthscale.constrain_bounded(0.3, 10.)  # 00)
        #
        # gpx.Gaussian_noise.variance.unconstrain()
        # gpy.Gaussian_noise.variance.unconstrain()
        # gpz.Gaussian_noise.variance.unconstrain()
        # gpx.Gaussian_noise.variance.constrain_fixed(0.1)
        # gpy.Gaussian_noise.variance.constrain_fixed(0.1)
        # gpz.Gaussian_noise.variance.constrain_fixed(0.1)

        if lengthscales or verbose:
            print(f"\nInput: {i}")
        # gpx.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        # gpy.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        # gpz.optimize_restarts(num_restarts=n_restarts, verbose=verbose)
        gpx.optimize_restarts(num_restarts=3, verbose=verbose)
        gpy.optimize_restarts(num_restarts=3, verbose=verbose)
        gpz.optimize_restarts(num_restarts=3, verbose=verbose)
        #     print(f"Cost: {gp.objective_function()}")
        if lengthscales:
            # print(f"X timescale: {gpx.rbf.lengthscale}")
            # print(f"Y timescale: {gpy.rbf.lengthscale}")
            # print(f"Z timescale: {gpz.rbf.lengthscale}")
            # print(f"X timescale: \n{gpx}")
            # gpx.plot()
            # print(f"Y timescale: \n{gpy}")
            # gpy.plot()
            # print(f"Z timescale: \n{gpz}")
            # gpz.plot()
            print(f"X timescale: \n{gpx}")
            print(f"Y timescale: \n{gpy}")
            print(f"Z timescale: \n{gpz}")
            plot_3outputs_independent(gpx, gpy, gpz, input_times, X, Y, Z)
            # TP.LML_landscape(input_times, X, gpx, noise_lower=-3., noise_upper=3., time_lower=-3., time_upper=3.)
            # TP.LML_landscape(input_times, Y, gpy, noise_lower=-3., noise_upper=3., time_lower=-3., time_upper=3.)
            # TP.LML_landscape(input_times, Z, gpz, noise_lower=0., noise_upper=3., time_lower=-3, time_upper=3, rbf_variance_lower=-10, rbf_variance_upper=1) # time_lower=np.log(0.1), time_upper=np.log(5),

        # FINDING INDIVIDUAL COSTS
        # if plot_cost:
            # ax.plot(input_trajectory[2, :], input_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(input_trajectory[2, :], input_trajectory[0, :], marker='+', color=colors[i])
            # axins.plot(input_trajectory[2, :], input_trajectory[1, :], ls='-', color=colors[i])

            #################################################################

        output_trajectory_masked = max(list_of_output_trajectories, key=latest_unmasked_index)
        output_mask = np.ma.getmask(output_trajectory_masked)
        output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3, -1))
        times_output_mask = np.zeros_like(times_array, dtype=bool)
        x2 = np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))  # Check this !!!
        times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[
            0])) - 1] = True  # I think this is correct because need to predict acceleration for last TWO times of position.
        times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[
            0])) - 1:] = True  # I think this should be the correct size to ensure the acceleration has the correct number of steps and therefore, the position should finish in the right spot.
        times_output_masked = np.ma.masked_array(times_array, times_output_mask)
        output_times = np.array(times_output_masked[~times_output_mask])
        # duration = max(output_times)-min(output_times)
        # output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))
        if plot_cost:
            xlim = [min(output_times+0.21),
                    max(output_times+0.21)]  # (max(output_times)-min(output_times)) * 1.03 + min(output_times)]
            axx.set_xlim(xlim)

        X_reshaped = output_times[:, None]
        array1 = output_trajectory.T[:, 0, None]
        array2 = output_trajectory.T[:, 1, None]
        array3 = output_trajectory.T[:, 2, None]
        # Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) - 1), axis=1)
        # noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
        # Xpred, Xvar = gp.predict(Times_pred_1, Y_metadata=noise_dict1)
        #
        # Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
        # noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
        # Ypred, Yvar = gp.predict(Times_pred_2, Y_metadata=noise_dict2)
        #
        # Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped) + 1), axis=1)
        # noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
        # Zpred, Zvar = gp.predict(Times_pred_3, Y_metadata=noise_dict3)
        Xpred_a, Xvar_a = gpx.predict(X_reshaped)
        Ypred_a, Yvar_a = gpy.predict(X_reshaped)
        Zpred_a, Zvar_a = gpz.predict(X_reshaped)
        ###################################################################
        Xpred, Xvel = evolve(input_trajectory[0, -1], velocity[0, -1], Xpred_a[:, 0], deltat)
        Ypred, Yvel = evolve(input_trajectory[1, -1], velocity[1, -1], Ypred_a[:, 0], deltat)
        Zpred, Zvel = evolve(input_trajectory[2, -1], velocity[2, -1], Zpred_a[:, 0], deltat)
        Xpred_lower, _ = evolve(input_trajectory[0, -1], velocity[0, -1], Xpred_a[:, 0] - Xvar_a[:, 0] ** 0.5, deltat)
        Ypred_lower, _ = evolve(input_trajectory[1, -1], velocity[1, -1], Ypred_a[:, 0] - Yvar_a[:, 0] ** 0.5, deltat)
        Zpred_lower, _ = evolve(input_trajectory[2, -1], velocity[2, -1], Zpred_a[:, 0] - Zvar_a[:, 0] ** 0.5, deltat)
        Xpred_upper, _ = evolve(input_trajectory[0, -1], velocity[0, -1], Xpred_a[:, 0] + Xvar_a[:, 0] ** 0.5, deltat)
        Ypred_upper, _ = evolve(input_trajectory[1, -1], velocity[1, -1], Ypred_a[:, 0] + Yvar_a[:, 0] ** 0.5, deltat)
        Zpred_upper, _ = evolve(input_trajectory[2, -1], velocity[2, -1], Zpred_a[:, 0] + Zvar_a[:, 0] ** 0.5, deltat)
        Xpred, Xvel = Xpred[1:], Xvel[1:]
        Ypred, Yvel = Ypred[1:], Yvel[1:]
        Zpred, Zvel = Zpred[1:], Zvel[1:]
        Xpred_lower = Xpred_lower[1:]
        Ypred_lower = Ypred_lower[1:]
        Zpred_lower = Zpred_lower[1:]
        Xpred_upper = Xpred_upper[1:]
        Ypred_upper = Ypred_upper[1:]
        Zpred_upper = Zpred_upper[1:]

        if plot_cost:
            # ax.plot(Zpred, Xpred, ls='--', color=colors[i], alpha=0.5)
            # axins.plot(Zpred, Ypred, ls='--', color=colors[i], alpha=0.5)
            # axins.set_ylim([142, 148])
            axx.plot(output_times+0.21, Xpred, ls='--', color=colors[i], alpha=0.3)
            axy.plot(output_times+0.21, Ypred, ls='--', color=colors[i], alpha=0.3)
            axz.plot(output_times+0.21, Zpred, ls='--', color=colors[i], alpha=0.3)
            # axx.fill_between(x=output_times, y1=Xpred[:, 0] - Xvar[:, 0] ** 0.5, y2=Xpred[:, 0] + Xvar[:, 0] ** 0.5, color=colors[i], alpha=0.05)
            axx.fill_between(x=output_times+0.21, y1=Xpred_lower, y2=Xpred_upper, color=colors[i], alpha=0.05)
            axy.fill_between(x=output_times+0.21, y1=Ypred_lower, y2=Ypred_upper, color=colors[i], alpha=0.05)
            axz.fill_between(x=output_times+0.21, y1=Zpred_lower, y2=Zpred_upper, color=colors[i], alpha=0.05)
            # plt.tight_layout()
        # final_index = np.amax(np.nonzero(~np.ma.getmaskarray(min(list_of_output_trajectories, key=np.ma.count_masked))[0]))     # Index corresponding to latest position information.
        # first_index = np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))                                      # Index corresponding to last position information of corresponding input.
        final_index = latest_unmasked_index(max(list_of_output_trajectories,
                                                key=latest_unmasked_index))  # Index corresponding to latest position information.
        first_index = latest_unmasked_index(
            input_trajectory_masked) + 1  # Index corresponding to first unknown position information.
        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            prediction_output_mask = output_mask[0, first_index:final_index + 1]
            Xpred_a_masked = np.ma.masked_array(Xpred, prediction_output_mask)
            Ypred_a_masked = np.ma.masked_array(Ypred, prediction_output_mask)
            Zpred_a_masked = np.ma.masked_array(Zpred, prediction_output_mask)
            Xpred_lower_masked = np.ma.masked_array(Xpred_lower, prediction_output_mask)
            Ypred_lower_masked = np.ma.masked_array(Ypred_lower, prediction_output_mask)
            Zpred_lower_masked = np.ma.masked_array(Zpred_lower, prediction_output_mask)
            Xpred_upper_masked = np.ma.masked_array(Xpred_upper, prediction_output_mask)
            Ypred_upper_masked = np.ma.masked_array(Ypred_upper, prediction_output_mask)
            Zpred_upper_masked = np.ma.masked_array(Zpred_upper, prediction_output_mask)
            Xpred_a_prime = np.array(Xpred_a_masked[~prediction_output_mask])
            Ypred_a_prime = np.array(Ypred_a_masked[~prediction_output_mask])
            Zpred_a_prime = np.array(Zpred_a_masked[~prediction_output_mask])
            Xpred_lower_prime = np.array(Xpred_lower_masked[~prediction_output_mask])
            Ypred_lower_prime = np.array(Ypred_lower_masked[~prediction_output_mask])
            Zpred_lower_prime = np.array(Zpred_lower_masked[~prediction_output_mask])
            Xpred_upper_prime = np.array(Xpred_upper_masked[~prediction_output_mask])
            Ypred_upper_prime = np.array(Ypred_upper_masked[~prediction_output_mask])
            Zpred_upper_prime = np.array(Zpred_upper_masked[~prediction_output_mask])
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3, -1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                output_trajectory = output_trajectory[:, :output_length]
                Xpred_a_prime = Xpred_a_prime[:output_length]
                Ypred_a_prime = Ypred_a_prime[:output_length]
                Zpred_a_prime = Zpred_a_prime[:output_length]
                Xpred_upper_prime = Xpred_upper_prime[:output_length]
                Ypred_upper_prime = Ypred_upper_prime[:output_length]
                Zpred_upper_prime = Zpred_upper_prime[:output_length]
                Xpred_lower_prime = Xpred_lower_prime[:output_length]
                Ypred_lower_prime = Ypred_lower_prime[:output_length]
                Zpred_lower_prime = Zpred_lower_prime[:output_length]
            times_output_mask = output_mask[0, :]  #
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                output_times = output_times[:output_length]
            if i==0 and j==0:
                color1=colors[i]
                X_reshaped = output_times[:, None]
                Xpred1 = np.array(Xpred_a_prime)
                Xstd1 = np.array(Xpred_upper_prime-Xpred_a_prime)
                X1 = output_trajectory.T[:, 0, None]
                Ypred1 = np.array(Ypred_a_prime)
                Ystd1 = np.array(Ypred_upper_prime-Ypred_a_prime)
                Y1 = output_trajectory.T[:, 1, None]
                Zpred1 = np.array(Zpred_a_prime)
                Zstd1 = np.array(Zpred_upper_prime-Zpred_a_prime)
                Z1 = output_trajectory.T[:, 2, None]
            if i==1 and j==1:
                color2=colors[i]
                X_reshaped = output_times[:, None]
                Xpred2 = np.array(Xpred_a_prime)
                Xstd2 = np.array(Xpred_upper_prime-Xpred_a_prime)
                X2 = output_trajectory.T[:, 0, None]
                Ypred2 = np.array(Ypred_a_prime)
                Ystd2 = np.array(Ypred_upper_prime-Ypred_a_prime)
                Y2 = output_trajectory.T[:, 1, None]
                Zpred2 = np.array(Zpred_a_prime)
                Zstd2 = np.array(Zpred_upper_prime-Zpred_a_prime)
                Z2 = output_trajectory.T[:, 2, None]
            cost_matrix[i, j] = individual_cost_function_acceleration(Xpred_a_prime, Ypred_a_prime, Zpred_a_prime,
                                                                      Xpred_upper_prime, Ypred_upper_prime,
                                                                      Zpred_upper_prime, output_trajectory,
                                                                      combined_axis_mean, output_times)  # plot_cost)
    if plot_cost:
        axx.xaxis.set_major_locator(MultipleLocator(0.5))
        axx.xaxis.set_minor_locator(AutoMinorLocator(2))
        axx.yaxis.set_minor_locator(AutoMinorLocator(2))
        axy.yaxis.set_major_locator(MultipleLocator(3))
        axy.yaxis.set_minor_locator(AutoMinorLocator(3))
        axz.yaxis.set_minor_locator(AutoMinorLocator(2))
        axx.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axy.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axz.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
        axis_NR_plotter(ax1, output_times, Xpred1, Xpred2, Xstd1, Xstd2, X1, X2, color1=color1, color2=color2,
                        ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
        axis_NR_plotter(ax2, output_times, Ypred1, Ypred2, Ystd1, Ystd2, Y1, Y2, color1=color1, color2=color2,
                        ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)
        axis_NR_plotter(ax3, output_times, Zpred1, Zpred2, Zstd1, Zstd2, Z1, Z2, color1=color1, color2=color2,
                        ylim=None, xlim=None, s=15 * (72. / fig.dpi) ** 2)

    if print_cost_matrix:
        print(cost_matrix)
    row_ind, col_ind, cost = combined_costs(cost_matrix)
    # print(f"The row indices are: {row_ind}")
    # print(f"The column indices are: {col_ind}")
    list_of_masked_times = []
    list_of_connected_trajectories = []
    trajectories = np.zeros((len(list_of_input_trajectories), 3, np.shape(list_of_input_trajectories[0])[-1]))
    large_mask = np.zeros((len(list_of_input_trajectories), 3, np.shape(list_of_input_trajectories[0])[-1]), dtype=bool)
    for i, row_index in enumerate(row_ind):
        col_index = col_ind[i]
        output_mask = np.ma.getmask(list_of_output_trajectories[col_index])
        input_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        trajectories[i, ~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        trajectories[i, ~input_mask] = list_of_input_trajectories[row_index][~input_mask]
        # list_of_input_trajectories[row_index][~output_mask] = list_of_output_trajectories[col_index][~output_mask]
        # new_mask = np.ma.getmask(list_of_input_trajectories[row_index])
        new_mask = (input_mask == True) & (output_mask == True)
        times_new_mask = new_mask[0, :]
        list_of_masked_times.append(np.ma.masked_array(times_array, mask=times_new_mask))
        # list_of_connected_trajectories.append(list_of_input_trajectories[row_index])
        # trajectories[i,:,:] = list_of_input_trajectories[row_index]
        large_mask[i, :, :] = new_mask
    masked_trajectories = np.ma.array(trajectories, mask=large_mask)
    # fig, (ax1, ax2,ax3) = plt.subplots(1, 3)
    # ax1.set_ylabel("Y")
    # ax1.set_xlabel("X")
    # ax2.set_ylabel("Z")
    # ax2.set_xlabel("X")
    # ax3.set_ylabel("Y")
    # ax3.set_xlabel("Z")
    # for i in range(len(list_of_input_trajectories)):
    #     ax1.plot(masked_trajectories[i,0,:],masked_trajectories[i,1,:])
    #     ax2.plot(masked_trajectories[i, 0, :], masked_trajectories[i, 2, :])
    #     ax3.plot(masked_trajectories[i, 2, :], masked_trajectories[i, 1, :])
    # fig.show()
    return masked_trajectories, col_ind, cost, row_ind

def axis_NR_plotter(ax, output_times, Xpred1,Xpred2, Xstd1, Xstd2, X1, X2, color1='r', color2='b', ylim=None, xlim=None, s=None):
    X1 = X1.squeeze()
    X2 = X2.squeeze()
    Xpred1 = Xpred1.squeeze()
    Xpred2 = Xpred2.squeeze()
    Xstd1 = Xstd1.squeeze()
    Xstd2 = Xstd2.squeeze()
    X1_C = (Xpred1-X1)/Xstd1
    X2_C = (Xpred2 - X2) / Xstd2
    X1_I = (Xpred1-X2)/Xstd1
    X2_I = (Xpred2 - X1) / Xstd2
    alpha = 0.6
    # ylower = ax.get_ylim()[0]-0.5
    # yupper = ax.get_ylim()[1]+0.5
    ax.scatter(output_times, X1_C, color=color1, marker='.', alpha=alpha, s=s)
    ax.scatter(output_times, X2_C, color=color2, marker='.', alpha=alpha, s=s)
    ax.scatter(output_times, X1_I, color=color1, marker='x', alpha=alpha, s=s)
    ax.scatter(output_times, X2_I, color=color2, marker='x', alpha=alpha, s=s)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ylim = [ax.get_ylim()[0]-0.1, ax.get_ylim()[1]+0.1]
    ax.set_ylabel('Norm. Res.')
    ax.set_xlabel('Time (s)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis="both", which='both', direction="in", bottom=True, top=True, left=True, right=True)
    return ax


def assignment_problem_plotter(list_of_input_trajectories, list_of_output_trajectories, times_array, n_restarts=3, verbose=False, lengthscales=False, print_cost_matrix=False, adjusting_output_len=False, output_max_length=5, plot_cost=False, switchingIO=False):
    """Takes list of input and output trajectories of the same length with masks in the slots with no data.
    There should be the same number of input and output trajectories.
    shape of each trajectory: (number_of_axis=3, number_of_timesteps in whole vid)"""
    # get list of Xs that line up with inputs and outputs and are limited to them.
    # for each input:
    #     train a GP
    #     compare actual outputs to predicted using defined function for MSLL
    #     store MSLL in array size that is the same size as the inputs and the outputs.
    # For the last line, if you do it for all inputs, can end up with a square array of inputs to outputs
    # Then I need some method of choosing the maximum combination of inputs and outputs. Research this...
    fontsize=12
    fontsize1=10
    cost_matrix = np.zeros((len(list_of_input_trajectories),len(list_of_output_trajectories)))
    if plot_cost:
        fig = plt.figure(figsize=(3.125, 2.75)) # ratio is approx: (6.25, 3.41)
        outer_grid = gridspec.GridSpec(1, 1, figure=fig, left=0.23, right=0.975, top=0.975, wspace=0.3, bottom=0.23)

        left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])

        ax = fig.add_subplot(left_cell[:, :])
        ax.set_xlabel('Z (m)', fontsize=fontsize)
        ax.set_ylabel('X (m)', fontsize=fontsize)

        axins = ax.inset_axes([0.32, 0.6, 0.375, 0.35])#[0.15, 0.5, 0.375, 0.4])   #Bottom left: [0.175, 0.15, 0.375, 0.35]; Top Left: [0.175, 0.6, 0.375, 0.35]
        axins.set_xlabel('Z (m)', fontsize=fontsize1)
        axins.set_ylabel('Y (m)', fontsize=fontsize1)

        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(list_of_input_trajectories))]
        alpha = 0.1
        alpha2 = 1

    for i, input_trajectory_masked in enumerate(list_of_input_trajectories):
        input_mask = np.ma.getmask(input_trajectory_masked)
        input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3,-1))
        X = input_trajectory[0, :, None]
        Y = input_trajectory[1, :, None]
        Z = input_trajectory[2, :, None]
        times_input_mask = input_mask[0,:]
        times_input_masked = np.ma.masked_array(times_array, times_input_mask)
        input_times = np.array(times_input_masked[~times_input_mask])
        input_times = input_times[:, None]

        # FINDING INDIVIDUAL COSTS
        if plot_cost:
            ax.plot(input_trajectory[2, :], input_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(input_trajectory[2, :], input_trajectory[0, :], marker='+', color=colors[i])
            axins.plot(input_trajectory[2, :], input_trajectory[1, :], ls='-', color=colors[i])

            output_trajectory_masked = list_of_output_trajectories[i] # min(list_of_output_trajectories, key=np.ma.count_masked)# list_of_output_trajectories[i]
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]

            ax.plot(output_trajectory[2, :], output_trajectory[0, :], ls='-', color=colors[i])
            # ax.scatter(output_trajectory[2, :], output_trajectory[0, :], marker='1', color=colors[i])
            axins.plot(output_trajectory[2, :], output_trajectory[1, :], ls='-', color=colors[i])

            output_trajectory_masked = min(list_of_output_trajectories, key=np.ma.count_masked)
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            times_output_mask = np.zeros_like(times_array, dtype=bool)
            times_output_mask[:np.amax(np.nonzero(~np.ma.getmaskarray(input_trajectory_masked)[0]))] = True
            times_output_mask[np.amax(np.nonzero(~np.ma.getmaskarray(output_trajectory_masked)[0])) + 1:] = True
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            duration = max(output_times)-min(output_times)
            output_times = np.concatenate((output_times, np.array([duration*1.03+min(output_times)])))

            xlim = [min(output_times), max(output_times)]

            X_reshaped = output_times[:, None]
            array1 = output_trajectory.T[:, 0, None]
            array2 = output_trajectory.T[:, 1, None]
            array3 = output_trajectory.T[:, 2, None]

            axins.set_ylim([142, 148])
            # plt.tight_layout()
        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            if adjusting_output_len:
                output_length = min(output_max_length, np.shape(output_trajectory)[-1])
                if switchingIO:
                    output_trajectory = output_trajectory[:, -output_length:]
                else:
                    output_trajectory = output_trajectory[:, :output_length]
            times_output_mask = output_mask[0,:]
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            if adjusting_output_len:
                if switchingIO:
                    output_times = output_times[-output_length:]
                else:
                    output_times = output_times[:output_length]
    return
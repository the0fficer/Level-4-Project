import matplotlib.pyplot as plt
import numpy as np
from time import time, strftime, localtime
from datetime import timedelta, datetime
# import time
import pickle
import os

total_time = time()


def seconds_to_str(elapsed=None):
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

    for i in range(trajectories.shape[0]):
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

    n_birds, n_parameters, n_time_steps = np.shape(trajectories)

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

    n_birds, n_parameters, n_time_steps = np.shape(trajectories)

    conditional_squared_distance = 3 * min(squared_distance_calculator(
        trajectories[0, :, 1], trajectories[0, :, 2]), squared_distance_calculator(
        trajectories[0, :, 2], trajectories[0, :, 3]), squared_distance_calculator(
        trajectories[0, :, 3], trajectories[0, :, 4]))

    difference_array = trajectories[:, :, 1:] - trajectories[:, :, :-1]
    squared_distance_array = np.sum(difference_array ** 2, axis=1)  # creates array with shape (n_birds, n_time_steps-1)
    splits_array = squared_distance_array > conditional_squared_distance  # Creates boolean array with splits located at True
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


def trajectory_reformatter(input_filename, plot_trajectories=False):
    """Takes in .dat files, corrects trajectories and save them in pickled arrays. Plots trajectories in process."""
    trajectories = read_trajectory(input_filename)
    trajectories = trajectory_error_correcter_improved(trajectories)
    if plot_trajectories:
        trajectory_plotter(trajectories)
    n_birds, n_parameters, n_time_steps = np.shape(trajectories)
    save_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{n_birds}birds{n_time_steps}timesteps{save_time}"
    output_filename = f"trajectories_as_arrays/{filename}"
    with open(output_filename, 'wb') as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def folder_reformatter(folder_name, plot_trajectories=False):
    """Takes folder name as input, and reformats files and saves them into trajectories_as_arrays as numpy arrays."""
    for filename in os.listdir(folder_name):
        start_time = time()
        input_filename = f"{folder_name}\{filename}"
        trajectory_reformatter(input_filename, plot_trajectories)
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
    for filename, counter in enumerate(os.listdir(folder_name)):
        start_time = time()
        input_filename = f"{folder_name}\{filename}"
        trajectories_arrays.append(array_unpacker(input_filename, plot_trajectories))
        print("\n")
        print(f"{filename}")
        print("--- %s ---" % seconds_to_str((time() - start_time)))
    print(f"The number of files is {number_of_files}")
    return trajectories_arrays


# array_unpacker("trajectories_as_arrays1/1birds101timesteps20201114-153312", True)

print("--- %s ---" % seconds_to_str((time() - total_time)))

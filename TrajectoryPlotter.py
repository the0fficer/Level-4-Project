import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from time import time, strftime, localtime
from datetime import timedelta, datetime
import pickle
import os
import itertools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import seaborn as sns
import GPy
import matplotlib.gridspec as gridspec
from itertools import permutations
from operator import mul
from functools import reduce
from math import fsum
# from spermutations import spermutations

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


def trajectory_plotter(trajectories, title = "Trajectories"):
    """Creates a 3D plot of the trajectories."""

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

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
    limit = 510000
    while len(splits_indices[0, :]) != 0 and counter < limit:
        counter += 1
        indices_of_birds_with_same_split = list(np.nonzero(splits_indices[1, :] == splits_indices[1, 0]))[0]
        position_of_first_bird = trajectories[splits_indices[0, 0], :, splits_indices[1, 0]]
        for count, i in enumerate(indices_of_birds_with_same_split):
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
        if counter%4000 == 0:
            print(f"{counter} - Corrections left: {len(splits_indices[0, :])}")
        if counter == limit:
            print("The trajectory correction failed")
            return trajectories, False
            # print(f"The number of corrections left is {len(splits_indices[0, :])}")
        # trajectory_plotter(trajectories)
    return trajectories, True


def trajectory_reformatter(input_filename, folder_save_location= "trajectories_as_arrays", plot_trajectories=False):
    """Takes in .dat files, corrects trajectories and save them in pickled arrays. Plots trajectories in process."""
    trajectories = read_trajectory(input_filename)
    trajectories, success = trajectory_error_correcter_improved(trajectories)
    if plot_trajectories:
        trajectory_plotter(trajectories)
    n_birds, n_parameters, n_time_steps = np.shape(trajectories)
    savetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{n_birds}birds{n_time_steps}timesteps{savetime}"
    if success == False:
        filename = f"ERROR_{filename}"
    output_filename = f"{folder_save_location}/{filename}"
    with open(output_filename, 'wb') as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def folder_reformatter(folder_name, folder_save_location= "trajectories_as_arrays", plot_trajectories=False):
    """Takes folder name as input and reformats files and saves them into trajectories_as_arrays as """
    # file_list = list(reversed(os.listdir(folder_name)))
    file_list = os.listdir(folder_name)
    for filename in file_list:
        start_time = time()
        input_filename = f"{folder_name}\{filename}"
        trajectory_reformatter(input_filename, folder_save_location=folder_save_location, plot_trajectories=plot_trajectories)
        print("\n")
        print(f"{filename}")
        print("--- %s ---" % seconds_to_str((time() - start_time)))
    return


def array_unpacker(file, plot_trajectories=False, title="Trajectories"):
    with open(file, 'rb') as f:
        trajectories = pickle.load(f)
    if plot_trajectories:
        trajectory_plotter(trajectories, title)
    return trajectories


def folder_unpacker(folder_name, plot_trajectories=False):
    number_of_files = len(os.listdir(folder_name))
    trajectories_arrays = []
    for counter, filename in enumerate(sorted(os.listdir(folder_name))):
        start_time = time()
        input_filename = f"{folder_name}\{filename}"
        trajectories_arrays.append(array_unpacker(input_filename, plot_trajectories, title=filename))
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


def coordinate_vs_time_plotter(array, xyz_axis=0, bird=0, axis_of_time_steps=2, start=0., end=1.):
    """Plots chosen 1 dimensional coordinate variation against time. Array should have shape (birds, xyz, timesteps)"""
    y_values = array[bird, xyz_axis, :]
    x_values = get_time_array(array, axis_of_time_steps, start, end)

    fig = plt.figure()
    ax = fig.add_subplot()

    if xyz_axis == 0:
        ax.set_ylabel('X (m)')
    elif xyz_axis == 1:
        ax.set_ylabel('Y (m)')
    elif xyz_axis == 2:
        ax.set_ylabel('Z (m)')
    else:
        print("That is not a valid axis choice. Please choose one of: 0, 1, 2")
    ax.set_xlabel('Time (s)')
    ax.scatter(x_values, y_values)
    return fig.show()


def kernel_function_vector(x, y, sigma_f=1, l=1):
    """Define squared exponential kernel function."""
    kernel = sigma_f * np.exp(- (np.linalg.norm(x - y) ** 2) / (2 * l ** 2))
    return kernel


def kernel_function_1D(x, y, sigma_f=1, l=1):
    """Define squared exponential kernel function."""
    kernel = sigma_f * np.exp(- ((x - y) ** 2) / (2 * l ** 2))
    return kernel


def compute_kernel_matrices(x, x_star, sigma_f=1, l=1):
    """
    Compute components of the covariance matrix of the joint distribution.

    We follow the notation:

        - K = K(X, X)
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    start_time = time()

    n = x.shape[0]
    n_star = x_star.shape[0]

    K = [kernel_function_vector(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x, x)]
    K = np.array(K).reshape(n, n)

    K_star2 = [kernel_function_vector(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x_star)]
    K_star2 = np.array(K_star2).reshape(n_star, n_star)

    K_star = [kernel_function_vector(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x)]
    K_star = np.array(K_star).reshape(n_star, n)

    print("--- %s ---" % seconds_to_str((time() - start_time)))

    return (K, K_star2, K_star)


def numpy_compute_kernel_matrices(x, x_star, sigma_f=1, l=1):
    """
    Compute components of the covariance matrix of the joint distribution.

    We follow the notation:

        - K = K(X, X)
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    start_time = time()


    xx, yy = np.meshgrid(x, x, sparse=True)
    xx_star2, yy_star2 = np.meshgrid(x_star, x_star, sparse=True)
    xx_star, yy_star = np.meshgrid(x, x_star, sparse=True)

    K = kernel_function_1D(xx, yy, sigma_f, l)
    K_star2 = kernel_function_1D(xx_star2, yy_star2, sigma_f, l)
    K_star = kernel_function_1D(xx_star, yy_star, sigma_f, l)

    print("--- %s ---" % seconds_to_str((time() - start_time)))

    return (K, K_star2, K_star)


def get_time_array(array, axis_of_time_steps=2, start=0., end=1.):
    """Produces array of time_steps."""
    number_of_time_steps = np.shape(array)[axis_of_time_steps]
    return np.linspace(start, end, number_of_time_steps)


def matrix_contour_plotter(matrix, cmap="cividis"):
    plt.imshow(matrix, cmap=cmap)
    plt.colorbar()
    return plt.show()


def construct_covariance_matrix(K, K_star2, K_star, sigma_n=0.):
    a = np.concatenate((K + (sigma_n ** 2) * np.eye(np.shape(K)[0]), K_star), axis=0)
    b = np.concatenate((K_star.T, K_star2), axis=0)
    C = np.concatenate((a, b), axis=1)
    return C


def compute_gp_parameters(K, K_star2, K_star, true_values, sigma_n=0.5):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    d = 1 # Dimension
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n ** 2) * np.eye(n)), true_values.reshape([n, d]))) #Prediction Function

    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n ** 2) * np.eye(n)), K_star.T)) # Covariance Function

    return (f_bar_star, cov_f_star)


def GPR(training_set, n_star, length=5., l=100, sigma_f=2, n_samples=10, three_dimensional=False):
    """Gaussian Proccess Regression."""
    n_training_points = np.shape(training_set)[-1]
    training_set = training_set.T
    if three_dimensional:
        assert np.all(np.shape(training_set)) == np.all(np.shape(np.zeros([n_training_points, 3])))
        dimension = 3
        x = np.linspace(start=0., stop=length, num=n_training_points)
        X = x.reshape(n_training_points, 1)
        X = np.hstack((X,X,X))

        x_star = np.linspace(start=0, stop=1.01 * length, num=n_star)
        X_star = x_star.reshape(n_star, 1)
        X_star = np.hstack((X_star, X_star, X_star))

    else:
        assert np.all(np.shape(training_set)) == np.all(np.shape(np.zeros(n_training_points)))
        dimension = 1
        x = np.linspace(start=0., stop=length, num=n_training_points)
        X = x.reshape(n_training_points, dimension)
        x_star = np.linspace(start=0, stop=1.01 * length, num=n_star)
        X_star = x_star.reshape(n_star, dimension)


    # Define kernel object.
    kernel = ConstantKernel(constant_value=sigma_f, constant_value_bounds=(1e-2, 1e10)) \
                * RBF(length_scale=l, length_scale_bounds=(1e-1, 1e2))

    kernel1 = WhiteKernel(noise_level=sigma_f, noise_level_bounds=(1e-10, 1e+1)) \
             + 1.0 * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e3))


    # Define GaussianProcessRegressor object.
    gp = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=10)#, alpha=0.1)


    # Fit to data using Maximum Likelihood Estimation of the parameters.
    gp.fit(X, training_set)
    print(gp.get_params())
    print(f"The log-marginal likelihood is {gp.log_marginal_likelihood()}.")
    print(gp.kernel_)

    # Make the prediction on test set.
    y_pred = gp.predict(X_star)


    # Generate samples from posterior distribution.
    y_hat_samples = gp.sample_y(X_star, n_samples=n_samples)

    # Compute the mean of the sample.
    y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
    # Compute the standard deviation of the sample.
    y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    # Plot training data.
    sns.scatterplot(x=x, y=training_set, label='training data', ax=ax1)

    for i in range(np.shape(y_hat_samples)[1]):
        sns.lineplot(x=x_star, y=y_hat_samples[:, i], color='blue', alpha=0.1, ax=ax1)
    # plt.plot(x_star, y_hat_samples, alpha=0.1)

    # Plot corridor.
    ax1.fill_between(
        x=x_star,
        y1=(y_hat - 2*y_hat_sd),
        y2=(y_hat + 2*y_hat_sd),
        color='green',
        alpha=0.3,
        label='Credible Interval'
    )
    # Plot prediction.
    sns.lineplot(x=x_star, y=y_pred, color='green', label='pred', ax=ax1)
    ax1.set(title=f'Prediction & Credible Interval for {gp.kernel_}.\n The log-marginal likelihood is {gp.log_marginal_likelihood()}', ylabel='Position (m)')
    ax1.legend(loc='lower left')


    sns.lineplot(x=x_star, y=np.zeros(x_star.shape), color='green', label='pred', ax=ax2)
    for i in range(np.shape(y_hat_samples)[1]):
        residuals = y_hat_samples[:, i] - y_pred
        sns.lineplot(x=x_star, y=residuals, color='blue', alpha=0.1, ax=ax2)
    ax2.set(ylabel='Residuals (m) ', xlabel='Time (s)')
    fig.show()


    # Plot LML landscape
    fig1 = plt.figure()
    # length_scale = np.logspace(-2, 2, 49)
    # noise = np.logspace(-2, 5, 50)
    # Length_Scale, Noise = np.meshgrid(length_scale, noise)
    # LML = [[gp.log_marginal_likelihood(np.log([Noise[i, j], Length_Scale[i, j], 3.5]))
    #         for i in range(Length_Scale.shape[0])] for j in range(Length_Scale.shape[1])]

    white_noise = np.logspace(-7, -4, 50)
    amplitude = np.logspace(3, 4, 50)
    White_Noise, Amplitude = np.meshgrid(white_noise, amplitude)
    LML = [[gp.log_marginal_likelihood(np.log([White_Noise[i, j], Amplitude[i, j], 3.5]))
            for i in range(White_Noise.shape[0])] for j in range(White_Noise.shape[1])]

    LML = np.array(LML).T

    vmin, vmax = (-LML).min(), (-LML).max()
    # vmax = 50
    level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)

    # plt.contour(Length_Scale, Noise, -LML,
    #             levels=level, norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.contour(White_Noise, Amplitude, -LML,
                levels=level,)# norm=LogNorm(vmin=vmin, vmax=vmax))

    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Noise")
    plt.ylabel("Amplitude")
    plt.title("Log-marginal-likelihood")
    plt.tight_layout()

    fig1.show()
    return


def GPyGPR(y, input_dim=1, variance=1., lengthscale=1., length=10., view_ratio=1.1):
    """Returns two figures of Gaussian process, one optimised, the other not. Uses GPy"""
    n_training_points = np.shape(y)[-1]
    x = np.linspace(start=0., stop=length, num=n_training_points)
    kernel = GPy.kern.RBF(input_dim=input_dim, variance=variance, lengthscale=lengthscale)
    X = x.reshape(n_training_points, input_dim)
    Y = y.reshape(n_training_points, input_dim)
    gp = GPy.models.GPRegression(X, Y, kernel)
    # print(gp)
    fig = gp.plot(plot_limits=np.array([0., view_ratio*length]))
    # GPy.plotting.show(fig, filename='basic_gp_regression_notebook')
    gp.optimize(messages=False)
    # gp.optimize(messages=True)
    gp.optimize_restarts(num_restarts=3)
    # print(gp)
    fig1 = gp.plot(plot_density=True, plot_limits=np.array([0., view_ratio*length]))
    # GPy.plotting.show(fig1, filename='basic_gp_regression_density_notebook_optimized')
    return


def GPy_get_X(y, keep_length=True, length=10., output_dim=1):
    """Returns X, Y (and length if keep_length=True). Shape of Y: (n_training_points, n_output_dim); Shape of X: (n_training_points, n_input_dimension=1)"""

    input_dim = 1
    n_training_points = np.shape(y)[-1]
    x = np.linspace(start=0., stop=length, num=n_training_points)
    X = x.reshape(n_training_points, input_dim)
    if output_dim == 1:
        Y = y.reshape(n_training_points, output_dim)
    elif output_dim != 1:
        Y = y.T
    if keep_length:
        return X, Y, length
    elif not keep_length:
        return X, Y
    else:
        return print("keep_length must be True or False.")

def GPy_log_marginal_likelihood(X, Y, keep_model=True, plot=False, variance=1., lengthscale=3., input_dim=1, length=10., view_ratio=1.1):
    """Initialises an rbf GP model and returns the log marginal likelihood for the parameters specified. Note that this returns the negative of the scikit learn value."""
    kernel= GPy.kern.RBF(input_dim=input_dim, variance=variance, lengthscale=lengthscale)
    gp = GPy.models.GPRegression(X, Y, kernel)
    # print(gp)
    # print(gp.rbf.lengthscale.values)
    if plot:
        gp.plot(plot_limits=np.array([0., view_ratio*length]))
    if keep_model:
        return gp.log_likelihood(), gp
    elif keep_model==False:
        return gp.log_likelihood()
    else:
        return print("keep_model must be True or False.")


def GPyOptimiser(gp, keep_model=True, plot=True, num_restarts=1, length=10., view_ratio=1.1):
    """Returns the optimised parameters, lengthscale, rbf variance, gaussian noise variance, for an rbf GP."""
    gp.optimize_restarts(num_restarts=num_restarts)
    # print(gp)
    if plot:
        gp.plot(plot_limits=np.array([0., view_ratio * length]))
    if keep_model:
        return gp.rbf.lengthscale, gp.rbf.variance, gp.Gaussian_noise.variance, gp
    elif keep_model == False:
        return gp.rbf.lengthscale, gp.rbf.variance, gp.Gaussian_noise.variance
    else:
        return print("keep_model must be True or False.")


def GPy_compiled(y, length=10., input_dim=1, output_dim=1, plot=True, view_ratio=1.1, variance=1., lengthscale=3., num_restarts=1):
    """Returns optimised GP and plots it."""
    X, Y, length = GPy_get_X(y, length=length, input_dim=input_dim, output_dim=output_dim, keep_length=True)
    log_marginal_likelihood, gp = GPy_log_marginal_likelihood(X, Y, keep_model=True, plot=plot, variance=variance, lengthscale=lengthscale, input_dim=input_dim, length=length, view_ratio=view_ratio)
    gp.rbf.lengthscale, gp.rbf.variance, gp.Gaussian_noise.variance, gp = GPyOptimiser(gp, keep_model=True, plot=plot, num_restarts=num_restarts, length=length, view_ratio=view_ratio)
    return X, Y, gp


def GPy_plotter(X, Y, gp, X_Test, num_samples):
    """Plots Gaussian Process with residuals"""
    n_training_points, input_dim = np.shape(X)
    n_training_points, output_dim = np.shape(Y)

    prediction_mean, prediction_variance = gp.predict(X_Test)
    quantiles = np.array(gp.predict_quantiles(X_Test))
    samples = gp.posterior_samples(X_Test, size=num_samples)


    # print(np.shape(quantiles))
    # print(np.shape(samples))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))
    ax1.plot(X_Test, prediction_mean, label='Prediction', color='green')
    ax1.scatter(X,Y, marker='|', color='k', )
    ax1.fill_between(
        x=X_Test[:, 0],
        y1=quantiles[0, :, 0],
        y2=quantiles[1, :, 0],
        color='green',
        alpha=0.3,
        label='Credible Interval'
    )

    for i in range(num_samples):
        # ax1.plot(X_Test, samples[:, 0, i], color='blue', alpha=0.1)

        Residuals = samples[:, 0, i] - prediction_mean[:, 0]
        ax2.plot(X_Test, Residuals, color='blue', label='Posterior sample',alpha=0.1)
    ax2.plot(X_Test, np.zeros(np.shape(X_Test)), color='green', label='prediction', alpha=0.1)

    ax1.legend(loc='lower left')
    ax1.set_title(
        f'Prediction and Residuals with rbf kernel \n (noise: {gp.Gaussian_noise.variance.values}, amplitude: {gp.rbf.variance.values}, length-scale: {gp.rbf.lengthscale.values}')
    ax1.set_ylabel('Position (m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Residuals (m)')
    return fig.show()


def plot_3outputs(X, gp,xlim):
    """Takes as input an array (n_training_points, n_output_dimensions=3)"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 8))
    slices = GPy.util.multioutput.get_slices([X, X, X])
    #Output 1
    ax1.set_xlim(xlim)
    ax1.set_title('X')
    gp.plot(plot_limits=xlim, ax=ax1, fixed_inputs=[(1,0)], which_data_rows=slices[0])
    # ax1.plot(X1,Y1,'r,',mew=1.5)
    #Output 2
    ax2.set_xlim(xlim)
    ax2.set_title('Y')
    gp.plot(plot_limits=xlim, ax=ax2, fixed_inputs=[(1,1)], which_data_rows=slices[1])
    # ax2.plot(X2,Y2,'r,',mew=1.5)
    # Output 3
    ax3.set_xlim(xlim)
    ax3.set_title('Z')
    gp.plot(plot_limits=xlim, ax=ax3, fixed_inputs=[(1,2)], which_data_rows=slices[2])
    # ax3.plot(X3, Y3, 'r,', mew=1.5)
    return


def three_dimensional_gaussian_plotter(Y, extension_ratio=0.2, length=10.):
    X, Y = GPy_get_X(Y, keep_length=False, length=length, output_dim=3)
    Y1 = Y[:, 0, None]
    Y2 = Y[:, 1, None]
    Y3 = Y[:, 2, None]

    icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=3, kernel=GPy.kern.RBF(1))
    # print(icm)

    gp = GPy.models.GPCoregionalizedRegression([X, X, X], [Y1,Y2,Y3], kernel=icm)
    gp['.*rbf.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
    gp.optimize(messages=False)
    # gp.optimize(messages=True)
    # print(gp)
    xlim=[min(X[:,0]), (max(X[:,0])-min(X[:,0]))*extension_ratio+max(X[:,0])]
    plot_3outputs(X, gp, xlim)
    return X, gp


def prediction_plotter(X_Test, gp):
    """Takes a 1D array and a gp as inputs and plots out the predictions"""
    newX = X_Test[:, None]
    predictions_array = np.zeros((3,2,np.shape(X_Test)[0]))

    for i in [-1, 0, 1]:
        each_axis_test = np.hstack([newX,np.ones_like(newX)+i])
        predictions_array[i, :,:] = np.squeeze(gp.predict(each_axis_test, include_likelihood=False))
    predictions_array = predictions_array[:,0,:]

    corrected_predictions = np.zeros((3, np.shape(X_Test)[0]))
    corrected_predictions[0,:] = predictions_array[2,:]
    corrected_predictions[1, :] = predictions_array[0, :]
    corrected_predictions[2, :] = predictions_array[1, :]

    trajectory_plotter(corrected_predictions[None, :, :])
    return predictions_array


def differentiater(trajectories, time_interval=0.01):
    """Function to differentiate either positions or velocities with respect to time. (n_birds, n_axis, n_time_steps) or (n_axis, n_timesteps)"""
    number_of_array_axis = len(np.shape(trajectories))
    if number_of_array_axis == 2:
        velocities = (trajectories[:, 1:] - trajectories[:, :-1])/time_interval
    elif number_of_array_axis == 3:
        velocities = (trajectories[:, :, 1:] - trajectories[:, :, :-1]) / time_interval
    else:
        raise ValueError("The size of the trajectories array was not correct.")
    return velocities


def plot_6outputs(X, gp,xlim):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(15, 8))
    slices = GPy.util.multioutput.get_slices([X, X, X])
    # Output 1
    ax1.set_xlim(xlim)
    ax1.set_title('X')
    gp.plot(plot_limits=xlim, ax=ax1, fixed_inputs=[(1,0)], which_data_rows=slices[0])
    # Output 2
    ax2.set_xlim(xlim)
    ax2.set_title('Y')
    gp.plot(plot_limits=xlim, ax=ax2, fixed_inputs=[(1,1)], which_data_rows=slices[1])
    # Output 3
    ax3.set_xlim(xlim)
    ax3.set_title('Z')
    gp.plot(plot_limits=xlim, ax=ax3, fixed_inputs=[(1,2)], which_data_rows=slices[2])
    # # Output 4
    # ax4.set_xlim(xlim)
    # ax4.set_title('X_velocity')
    # gp.plot(plot_limits=xlim, ax=ax4, fixed_inputs=[(1,3)], which_data_rows=slices[3])
    # # Output 5
    # ax5.set_xlim(xlim)
    # ax5.set_title('Y_velocity')
    # gp.plot(plot_limits=xlim, ax=ax5, fixed_inputs=[(1,4)], which_data_rows=slices[4])
    # # Output 6
    # ax6.set_xlim(xlim)
    # ax6.set_title('Z')
    # gp.plot(plot_limits=xlim, ax=ax6, fixed_inputs=[(1,5)], which_data_rows=slices[5])
    return


def six_dimensional_gaussian_plotter(Y,V, extension_ratio=0.2, length=10.):
    X, Y = GPy_get_X(Y, keep_length=False, length=length, input_dim=1, output_dim=3)
    Y1 = Y[:, 0, None]
    Y2 = Y[:, 1, None]
    Y3 = Y[:, 2, None]

    icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=3, kernel=GPy.kern.RBF(1))
    # print(icm)

    gp = GPy.models.GPCoregionalizedRegression([X, X, X], [Y1,Y2,Y3], kernel=icm)
    gp['.*rbf.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
    gp.optimize(messages=False)
    # gp.optimize(messages=True)
    # print(gp)
    xlim=[min(X[:,0]), (max(X[:,0])-min(X[:,0]))*extension_ratio+max(X[:,0])]
    plot_3outputs(X, gp, xlim)
    return X, gp


def plot_multi_outputs(X_list, gp,xlim,n_outputs):
    """Plots GPs for variable number of outputs. X is the list of np arrays."""

    slices = GPy.util.multioutput.get_slices(X_list)
    fig = plt.figure(figsize=(15, 8))
    plt.subplots_adjust(hspace=0.)
    for i in range(1, n_outputs+1):
        ax = plt.subplot(n_outputs, 1, i)
        ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis="both", labelbottom=False)
        ax.set_xlim(xlim)
        ax.set_ylabel(f'Output {i}')
        gp.plot(plot_limits=xlim, ax=ax, fixed_inputs=[(1, i-1)], which_data_rows=slices[i-1], legend=False)
    ax.tick_params(axis="both", labelbottom=True)
    plt.show()
    return


def multi_dimensional_gaussian_plotter(Y, extension_ratio=0.2, length=10., n_dimensions=3, fraction=0.1):
    # Then see why all the outputs are going to 0. Should I change to differentiate and then integrate after?
    if n_dimensions == 3:
        pass
    elif n_dimensions == 6:
        Y2 = differentiater(Y)
        X2, Y2 = GPy_get_X(Y2, keep_length=False, length=length, output_dim=3)
        Y2 = Y2[:, :, None]

        X2 = array_fractional_reducer(X2, fraction=fraction, axis=0)
        Y2 = array_fractional_reducer(Y2, fraction=fraction, axis=0)
        X_List2 = [X2, X2, X2]
        Y_List2 = list(np.transpose(Y2, [1, 0, 2]))
    elif n_dimensions ==9:
        Y2 = differentiater(Y)
        Y3 = differentiater(Y2)

        X2, Y2 = GPy_get_X(Y2, keep_length=False, length=length, output_dim=3)
        Y2 = Y2[:, :, None]
        X3, Y3 = GPy_get_X(Y3, keep_length=False, length=length, output_dim=3)
        Y3 = Y3[:, :, None]

        X2 = array_fractional_reducer(X2, fraction=fraction, axis=0)
        Y2 = array_fractional_reducer(Y2, fraction=fraction, axis=0)
        X_List2 = [X2, X2, X2]
        Y_List2 = list(np.transpose(Y2, [1, 0, 2]))
        X3 = array_fractional_reducer(X3, fraction=fraction, axis=0)
        Y3 = array_fractional_reducer(Y3, fraction=fraction, axis=0)
        X_List3 = [X3, X3, X3]
        Y_List3 = list(np.transpose(Y3, [1, 0, 2]))
    else:
        raise ValueError("n_dimensions must be 3, 6, or 9.")
    X1, Y1 = GPy_get_X(Y, keep_length=False, length=length, output_dim=3)
    Y1 = Y1[:, :, None]

    X1 = array_fractional_reducer(X1, fraction=fraction, axis=0)
    Y1 = array_fractional_reducer(Y1, fraction=fraction, axis=0)
    X_List1 = [X1, X1, X1]
    Y_List1 = list(np.transpose(Y1,[1, 0, 2]))

    if n_dimensions==3:
        Y_List = Y_List1
        X_List = X_List1
    elif n_dimensions == 6:
        Y_List = Y_List1+Y_List2
        X_List = X_List1+X_List2
    elif n_dimensions == 9:
        Y_List = Y_List1+Y_List2+Y_List3
        X_List = X_List1+X_List2+X_List3

    icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=n_dimensions, kernel=GPy.kern.RBF(1))
    # print(icm)

    gp = GPy.models.GPCoregionalizedRegression(X_List, Y_List, kernel=icm)
    gp['.*rbf.var'].constrain_fixed(1.) #For this kernel, B.kappa encodes the variance now.
    # gp.optimize(messages=True)
    # gp.optimize(messages=False)
    gp.optimize_restarts(num_restarts=3)
    # print(gp)
    # print(icm.B.B)
    xlim=[min(X1[:,0]), (max(X1[:,0])-min(X1[:,0]))*extension_ratio+max(X1[:,0])]
    # plot_multi_outputs(X_List, gp, xlim, n_dimensions)
    return X1, gp



def trajectory_splitter(trajectory, split_start, split_length):
    """Take trajectory of shape [number of axis, number of timesteps] and return two trajectories of size
    [number_of_axis, split_start] and [number_of_axis, number of timesteps-split_start-split_length]. Also returns array
    with the split part masked."""
    mask_array = np.zeros(np.shape(trajectory), dtype=bool)
    # print(np.shape(mask_array))
    mask_array[:, split_start:split_start+split_length] = 1
    # print(np.ma.masked_array(trajectory, mask=mask_array))
    return trajectory[:, :split_start],trajectory[:, split_start+split_length:], np.ma.masked_array(trajectory, mask=mask_array)


def report_graph_plotter(trajectory1, trajectory2, n_split, n_length, fraction, length):
    """Function to make figure for report."""

    X, Y = GPy_get_X(trajectory1, keep_length=False, output_dim=3, length = length)
    Xprime, Yprime = GPy_get_X(trajectory2, keep_length=False, output_dim=3, length= length)

    # n_split = int(np.floor(np.shape(trajectory1)[-1]/2))
    # n_length = 3

    Y1, Y2, Y_mask = trajectory_splitter(trajectory1, n_split, n_length)
    Y1prime, Y2prime, Yprime_mask = trajectory_splitter(trajectory2, n_split, n_length)
    X1, X2, X_mask = trajectory_splitter(X.T, n_split, n_length)
    X1prime, X2prime, Xprime_mask = trajectory_splitter(X.T, n_split, n_length)
    X1, X2, X_mask = X1.T, X2.T, X_mask.T
    X1prime, X2prime, Xprime_mask = X1prime.T, X2prime.T, Xprime_mask.T

    throwaway1, GP1 = multi_dimensional_gaussian_plotter(Y1, extension_ratio=0., length=n_split*0.01/fraction, n_dimensions=3, fraction=1.)
    throwaway2, GP2 = multi_dimensional_gaussian_plotter(Y1prime, extension_ratio=0., length=n_split*0.01/fraction, n_dimensions=3, fraction=1.)

    # assert((intermediate1 == X1).all())
    # assert((intermediate2 == X1prime).all())

    Y1 = Y1[None, :, :] # treating each trajectory fragment as a separate bird.
    Y2 = Y2[None, :, :]
    Y1prime = Y1[None, :, :]
    Y2prime = Y2prime[None, :, :]
    # Y_mask = Y_mask[None, :, :]
    # Yprime_mask = Yprime_mask[None, :, :]

    # trajectories = np.ma.concatenate((Y_mask,Yprime_mask), axis=0)
    # print(f'The shape of the trajectories before is [2,{np.shape(trajectory1)}]\nThe shape of Y_mask is {np.shape(Y_mask)}\nThe shape of trajectories is {np.shape(trajectories)}')

    fig = plt.figure(figsize=(9.,4.))
    outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975,top=0.975, wspace=0.3)

    left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])

    ax = fig.add_subplot(left_cell[:, :])
    right_cell = outer_grid[1].subgridspec(5, 3, hspace=0.05)
    upper_right_cell = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=right_cell[:3, :], hspace=0.0)
    lower_right_cell = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=right_cell[3:, :], hspace=0.0)
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
    ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
    ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)


    ax.set_xlabel('Z')
    ax.set_ylabel('Y')

    ax.plot(Y_mask[2,:],Y_mask[1,:], 'k-')
    ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b-')


    # inverse_mask = ~np.array(np.ma.getmask(Y_mask), dtype=bool)
    # Y_no_mask = np.ma.masked_array(Y_mask, ~np.ma.getmask(Y_mask))
    # Yprime_no_mask = np.ma.masked_array(Yprime_mask, ~np.ma.getmask(Yprime_mask))
    # ax.plot(Y_no_mask[2,:],Y_no_mask[1,:], 'k--')
    # ax.plot(Yprime_no_mask[2, :], Yprime_no_mask[1, :], 'b--')


    axins = ax.inset_axes([0.175, 0.15, 0.375, 0.35])
    axins.plot(Yprime_mask[2, :], Yprime_mask[0, :], 'b-')
    axins.plot(Y_mask[2,:],Y_mask[0,:], 'k-')
    # axins.plot(Yprime_no_mask[2, :], Yprime_no_mask[0, :], 'b--')
    # axins.plot(Y_no_mask[2,:],Y_no_mask[0,:], 'k--')
    axins.set_xlabel('Z')
    axins.set_ylabel('X')

    Y_mask.mask = np.ma.nomask
    Yprime_mask.mask = np.ma.nomask
    ax.plot(Y_mask[2,:],Y_mask[1,:], 'k:')
    ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b:')
    axins.plot(Yprime_mask[2, :], Yprime_mask[0, :], 'b:')
    axins.plot(Y_mask[2,:],Y_mask[0,:], 'k:')

    ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

    axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="both", labelbottom=False)
    axx.tick_params(axis="both", labelbottom=False)
    axy.tick_params(axis="both", labelbottom=False)
    axz.tick_params(axis="both", labelbottom=False)

    X_list = [X1, X1, X1]
    Xprime_list = [X1prime, X1prime, X1prime]
    slices = GPy.util.multioutput.get_slices(X_list)
    slicesprime = GPy.util.multioutput.get_slices(Xprime_list)
    assert((slices == slicesprime))

    axx.set_ylabel('X')
    axy.set_ylabel('Y')
    axz.set_ylabel('Z')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Incorrect\nMatching')
    ax2.set_ylabel('Correct\nMatching')

    xlim = [(n_split+n_length)*0.01/fraction + 0.1, length] # Can't be bothered to do the maths to work out why the 0.1 works. multiply by 0.01 for the timesteps.divid by fraction for beginning.
    ax2.set_xlim(xlim)
    # ax2.set_ylim(-4,3.5)
    # ax3.set_ylim(-11,15)
    # ax3.set_xlim(xlim)

    # GP1.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)], which_data_rows=slices[0], legend=False, marker='+')
    # GP1.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], which_data_rows=slices[1], legend=False, marker='+')
    # GP1.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], which_data_rows=slices[2], legend=False, marker='+')
    # GP2.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)], which_data_rows=slicesprime[0], legend=False, marker='+')
    # GP2.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], which_data_rows=slicesprime[1], legend=False, marker='+')
    # GP2.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], which_data_rows=slicesprime[2], legend=False, marker='+')


    num_samples = 5

    Xnewx = np.concatenate((X2, np.ones_like(X2)-1), axis=1)
    noise_dict = {'output_index': Xnewx[:, 1:].astype(int)}
    Xpred, Xvar = GP1.predict(Xnewx,Y_metadata=noise_dict)
    Xnewx = np.concatenate((X2, np.ones_like(X2)-1), axis=1)
    noise_dict = {'output_index': Xnewx[:, 1:].astype(int)}
    Xpred_prime, Xprime_var = GP2.predict(Xnewx,Y_metadata=noise_dict)
    Xquantiles = np.array(GP1.predict_quantiles(Xnewx,Y_metadata=noise_dict))
    Xsamples = GP1.posterior_samples(Xnewx, Y_metadata=noise_dict, size=num_samples)
    Xprime_quantiles = np.array(GP2.predict_quantiles(Xnewx,Y_metadata=noise_dict))
    Xprime_samples = GP2.posterior_samples(Xnewx, Y_metadata=noise_dict, size=num_samples)

    Ynewx = np.concatenate((X2, np.ones_like(X2)), axis=1)
    noise_dict = {'output_index': Ynewx[:, 1:].astype(int)}
    Ypred, Yvar = GP1.predict(Ynewx,Y_metadata=noise_dict)
    Ynewx = np.concatenate((X2, np.ones_like(X2)), axis=1)
    noise_dict = {'output_index': Ynewx[:, 1:].astype(int)}
    Ypred_prime, Yprime_var = GP2.predict(Ynewx,Y_metadata=noise_dict)
    Yquantiles = np.array(GP1.predict_quantiles(Ynewx,Y_metadata=noise_dict))
    Ysamples = GP1.posterior_samples(Ynewx, Y_metadata=noise_dict, size=num_samples)
    Yprime_quantiles = np.array(GP2.predict_quantiles(Ynewx,Y_metadata=noise_dict))
    Yprime_samples = GP2.posterior_samples(Ynewx, Y_metadata=noise_dict, size=num_samples)


    Znewx = np.concatenate((X2, np.ones_like(X2)+1), axis=1)
    noise_dict = {'output_index': Znewx[:, 1:].astype(int)}
    Zpred, Zvar = GP1.predict(Znewx,Y_metadata=noise_dict)
    Znewx = np.concatenate((X2, np.ones_like(X2)+1), axis=1)
    noise_dict = {'output_index': Znewx[:, 1:].astype(int)}
    Zpred_prime, Zprime_var = GP2.predict(Znewx,Y_metadata=noise_dict)
    Zquantiles = np.array(GP1.predict_quantiles(Znewx,Y_metadata=noise_dict))
    Zsamples = GP1.posterior_samples(Znewx, Y_metadata=noise_dict, size=num_samples)
    Zprime_quantiles = np.array(GP2.predict_quantiles(Znewx,Y_metadata=noise_dict))
    Zprime_samples = GP2.posterior_samples(Znewx, Y_metadata=noise_dict, size=num_samples)


    # axx.fill_between(x=Xnewx[:, 0], y1=Xquantiles[0, :, 0], y2=Xquantiles[1, :, 0], color='black', alpha=0.05)
    # axx.fill_between(x=Xnewx[:, 0], y1=Xprime_quantiles[0, :, 0], y2=Xprime_quantiles[1, :, 0], color='blue', alpha=0.05)
    # axy.fill_between(x=Xnewx[:, 0], y1=Yquantiles[0, :, 0], y2=Yquantiles[1, :, 0], color='black', alpha=0.05)
    # axy.fill_between(x=Xnewx[:, 0], y1=Yprime_quantiles[0, :, 0], y2=Yprime_quantiles[1, :, 0], color='blue',alpha=0.05)
    # axz.fill_between(x=Xnewx[:, 0], y1=Zquantiles[0, :, 0], y2=Zquantiles[1, :, 0], color='black', alpha=0.05)
    # axz.fill_between(x=Xnewx[:, 0], y1=Zprime_quantiles[0, :, 0], y2=Zprime_quantiles[1, :, 0], color='blue',alpha=0.05)

    axx.fill_between(x=Xnewx[:, 0], y1=Xpred[:,0]-Xvar[:,0]**0.5, y2=Xpred[:,0]+Xvar[:,0]**0.5, color='black', alpha=0.05)
    axx.fill_between(x=Xnewx[:, 0], y1=Xpred_prime[:,0]-Xprime_var[:,0]**0.5, y2=Xpred_prime[:,0]+Xprime_var[:,0]**0.5, color='blue', alpha=0.05)
    axy.fill_between(x=Xnewx[:, 0], y1=Ypred[:,0]-Yvar[:,0]**0.5, y2=Ypred[:,0]+Yvar[:,0]**0.5, color='black', alpha=0.05)
    axy.fill_between(x=Xnewx[:, 0], y1=Ypred_prime[:,0]-Yprime_var[:,0]**0.5, y2=Ypred_prime[:,0]+Yprime_var[:,0]**0.5, color='blue', alpha=0.05)
    axz.fill_between(x=Xnewx[:, 0], y1=Zpred[:,0]-Zvar[:,0]**0.5, y2=Zpred[:,0]+Zvar[:,0]**0.5, color='black', alpha=0.05)
    axz.fill_between(x=Xnewx[:, 0], y1=Zpred_prime[:,0]-Zprime_var[:,0]**0.5, y2=Zpred_prime[:,0]+Zprime_var[:,0]**0.5, color='blue', alpha=0.05)

    # axx.plot(X2, Xpred,  'k--', alpha=0.5)
    # axx.plot(X2, Xpred_prime, 'b--', alpha=0.5)
    # axy.plot(X2, Ypred, 'k--', alpha=0.5)
    # axy.plot(X2, Ypred_prime, 'b--', alpha=0.5)
    # axz.plot(X2, Zpred, 'k--', alpha=0.5)
    # axz.plot(X2, Zpred_prime, 'b--', alpha=0.5)

    # axx.scatter(X2[:, 0], Y2[0, 0, :],  color='k', marker='x', s=50*(72./fig.dpi)**2)
    # axx.scatter(X2[:, 0], Y2prime[0, 0, :], color='b', marker='x', s=50*(72./fig.dpi)**2)
    # axy.scatter(X2[:, 0], Y2[0, 1, :],  color='k', marker='1', s=50*(72./fig.dpi)**2)
    # axy.scatter(X2[:, 0], Y2prime[0, 1, :], color='b', marker='1', s=50*(72./fig.dpi)**2)
    # axz.scatter(X2[:, 0], Y2[0, 2, :],  color='k', marker='+', s=50*(72./fig.dpi)**2)
    # axz.scatter(X2[:, 0], Y2prime[0, 2, :], color='b', marker='+', s=50*(72./fig.dpi)**2)

    axx.plot(X2[:, 0], Y2[0, 0, :],  color='k', linestyle=':', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    axx.plot(X2[:, 0], Y2prime[0, 0, :], color='b', linestyle=':', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    axy.plot(X2[:, 0], Y2[0, 1, :],  color='k', linestyle='--', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    axy.plot(X2[:, 0], Y2prime[0, 1, :], color='b', linestyle='--', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    axz.plot(X2[:, 0], Y2[0, 2, :],  color='k', linestyle='-.', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    axz.plot(X2[:, 0], Y2prime[0, 2, :], color='b', linestyle='-.', alpha=0.5) # , s=50*(72./fig.dpi)**2)

    Xresiduals = (Y2[0, 0, :] - Xpred[:,0])/Xvar[:,0]**0.5
    Yresiduals = (Y2[0, 1, :] - Ypred[:,0])/Yvar[:,0]**0.5
    Zresiduals = (Y2[0, 2, :] - Zpred[:,0])/Zvar[:,0]**0.5
    # ax2.scatter(X2[:, 0], Xresiduals, color='k', marker='x', alpha=0.3)# , s=(72./fig.dpi)**2)
    # ax2.scatter(X2[:, 0], Yresiduals, color='k', marker='1', alpha=0.3)# , s=(72./fig.dpi)**2)
    # ax2.scatter(X2[:, 0], Zresiduals, color='k', marker='+', alpha=0.3)# , s=(72./fig.dpi)**2)
    ax2.plot(X2[:, 0], Xresiduals, color='k', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax2.plot(X2[:, 0], Yresiduals, color='k', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax2.plot(X2[:, 0], Zresiduals, color='k', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)

    Xprime_residuals = (Y2prime[0, 0, :] - Xpred_prime[:, 0])/Xprime_var[:,0]**0.5
    Yprime_residuals = (Y2prime[0, 1, :] - Ypred_prime[:, 0])/Yprime_var[:,0]**0.5
    Zprime_residuals = (Y2prime[0, 2, :] - Zpred_prime[:, 0])/Zprime_var[:,0]**0.5
    # ax2.scatter(X2[:, 0], Xprime_residuals, color='b', marker='x', alpha=0.3)  # , s=(72./fig.dpi)**2)
    # ax2.scatter(X2[:, 0], Yprime_residuals, color='b', marker='1', alpha=0.3)  # , s=(72./fig.dpi)**2)
    # ax2.scatter(X2[:, 0], Zprime_residuals, color='b', marker='+', alpha=0.3)  # , s=(72./fig.dpi)**2)
    ax2.plot(X2[:, 0], Xprime_residuals, color='b', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax2.plot(X2[:, 0], Yprime_residuals, color='b', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax2.plot(X2[:, 0], Zprime_residuals, color='b', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)


    bad_Xresiduals = (Y2prime[0, 0, :] - Xpred[:, 0])/Xvar[:,0]**0.5
    bad_Yresiduals = (Y2prime[0, 1, :] - Ypred[:, 0])/Yvar[:,0]**0.5
    bad_Zresiduals = (Y2prime[0, 2, :] - Zpred[:, 0])/Zvar[:,0]**0.5
    # ax3.scatter(X2[:, 0], bad_Xresiduals, color='k', marker='x', alpha=0.3)# , s=(72./fig.dpi)**2)
    # ax3.scatter(X2[:, 0], bad_Yresiduals, color='k', marker='1', alpha=0.3)# , s=(72./fig.dpi)**2)
    # ax3.scatter(X2[:, 0], bad_Zresiduals, color='k', marker='+', alpha=0.3)# , s=(72./fig.dpi)**2)
    ax3.plot(X2[:, 0], bad_Xresiduals, color='k', linestyle=':', alpha=0.5)# , s=(72./fig.dpi)**2)
    ax3.plot(X2[:, 0], bad_Yresiduals, color='k', linestyle='--', alpha=0.5)# , s=(72./fig.dpi)**2)
    ax3.plot(X2[:, 0], bad_Zresiduals, color='k', linestyle='-.', alpha=0.5)# , s=(72./fig.dpi)**2)


    bad_Xprime_residuals = (Y2[0, 0, :] - Xpred_prime[:,0])/Xprime_var[:,0]**0.5
    bad_Yprime_residuals = (Y2[0, 1, :] - Ypred_prime[:,0])/Yprime_var[:,0]**0.5
    bad_Zprime_residuals = (Y2[0, 2, :] - Zpred_prime[:,0])/Zprime_var[:,0]**0.5
    # ax3.scatter(X2[:, 0], bad_Xprime_residuals, color='b', marker='x', alpha=0.3)  # , s=(72./fig.dpi)**2)
    # ax3.scatter(X2[:, 0], bad_Yprime_residuals, color='b', marker='1', alpha=0.3)  # , s=(72./fig.dpi)**2)
    # ax3.scatter(X2[:, 0], bad_Zprime_residuals, color='b', marker='+', alpha=0.3)  # , s=(72./fig.dpi)**2)
    ax3.plot(X2[:, 0], bad_Xprime_residuals, color='b', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax3.plot(X2[:, 0], bad_Yprime_residuals, color='b', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax3.plot(X2[:, 0], bad_Zprime_residuals, color='b', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ylabels = axz.get_yticklabels()
    # print(ylabels)

    # X2_list = [X2, X2, X2]
    #
    # print(GP3.log_predictive_density())
    return fig.show()




# TP.coordinate_vs_time_plotter(trajectories, xyz_axis=2, end=5.)
# x = TP.get_time_array(trajectories, end=5.)
# x_star = np.linspace(0, 5., 101)
# y = trajectories[0, 2, :]
# K, K_star2, K_star = TP.numpy_compute_kernel_matrices(x, x_star, sigma_f=1., l=0.5)
# OldK, OldK_star2, OldK_star = TP.numpy_compute_kernel_matrices(x, x_star, sigma_f=1., l=1.)
# C = TP.construct_covariance_matrix(K, K_star2, K_star)
# TP.matrix_contour_plotter(C)
# f_bar_star, cov_f_star = TP.compute_gp_parameters(K, K_star2, K_star, true_values=y)
# TP.matrix_contour_plotter(cov_f_star)

# print("--- %s ---" % seconds_to_str((time() - total_time)))

def trajectory_masker(trajectory, split_start, split_length):
    """Take trajectory of shape [number of axis, number of timesteps] and return two trajectories of same size as before
    but one has a mask starting from the split and the other has a mask that finishes at the end of the split."""
    input_mask = np.zeros(np.shape(trajectory), dtype=bool)
    output_mask = np.zeros(np.shape(trajectory), dtype=bool)

    input_mask[:, split_start:] = 1
    output_mask[:, :split_start+split_length] = 1
    # print(np.ma.masked_array(trajectory, mask=mask_array))
    return np.ma.masked_array(trajectory, mask=input_mask), np.ma.masked_array(trajectory, mask=output_mask)


def train_GPs_on_position(list_of_input_trajectories, list_of_output_trajectories, times_array):
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
        # gp.optimize(messages=True)
        gp.optimize(messages=False)

        # FINDING INDIVIDUAL COSTS
        for j, output_trajectory_masked in enumerate(list_of_output_trajectories):#
            output_mask = np.ma.getmask(output_trajectory_masked)
            output_trajectory = np.array(output_trajectory_masked[~output_mask].reshape(3,-1))
            times_output_mask = output_mask[0,:]
            times_output_masked = np.ma.masked_array(times_array, times_output_mask)
            output_times = np.array(times_output_masked[~times_output_mask])
            cost_matrix[i,j] = individual_cost_function(gp, output_trajectory, output_times)


    # ARRAY OF ROW INDICES, ARRRAY OF COLUMN INDICES
    # CALL COMBINED COSTS
    # INPUT ARRAY[OUTPUT ARRAY NO MASK]  = OUTPUT ARRAY[OUTPUT ARRAY NO MASK]

    return

def combined_costs(matrix_MSLL_IO):
    """Choose the optimum combination based on the minimum combined cost. The method for combining the cost has yet to be determined (ie sum or product). Take the MSLL for each input and output and return the best combination of inputs and outputs as well as the probability of that choice."""
    return

def individual_cost_function(gp, output_trajectory, output_times):
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
    X_list = X_reshaped
    array1 = output_trajectory.T[:, 0, None]
    array2 = output_trajectory.T[:, 1, None]
    array3 = output_trajectory.T[:, 2, None]
    Y_list = np.concatenate((array1,array2,array3),axis=1)
    Y_list = array1
    X_list = np.concatenate((X_reshaped,np.zeros_like(X_reshaped)),axis=1)


    Times_pred_1 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)-1), axis=1)
    noise_dict1 = {'output_index': Times_pred_1[:, 1:].astype(int)}
    Xpred, Xvar = gp.predict(Times_pred_1,Y_metadata=noise_dict1)

    Times_pred_2 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)), axis=1)
    noise_dict2 = {'output_index': Times_pred_2[:, 1:].astype(int)}
    Ypred, Yvar = gp.predict(Times_pred_2,Y_metadata=noise_dict2)

    Times_pred_3 = np.concatenate((X_reshaped, np.ones_like(X_reshaped)+1), axis=1)
    noise_dict3 = {'output_index': Times_pred_3[:, 1:].astype(int)}
    Zpred, Zvar = gp.predict(Times_pred_3,Y_metadata=noise_dict3)

    return gp.log_predictive_density(X_list,Y_list) # ,Y_metadata=noise_dict1) # ARRAY OF ROW INDICES, ARRAY OF COLUMN INDICES, COST


def BensDet(A, total=0):
    indices = list(range(len(A)))
    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] + A[1][0] * A[0][1]
        return val
    for fc in indices:
        As = A[:]
        As = As[1:]
        height = len(As)

        for i in range(height):
            As[i] = As[i][0:fc] + As[i][fc + 1:]
        sub_det = BensDet(As)
        total += A[0][fc] * sub_det
    return total


# def prod(lst):
#     return reduce(mul, lst, 1)
#
#
# def perm(a):
#     n = len(a)
#     r = range(n)
#     s = permutations(r)
#     return [prod(a[i][sigma[i]] for i in r) for sigma in s]

def prod(lst):
    return reduce(mul, lst, 1)


def perm(a):
    n = len(a)
    r = range(n)
    s = permutations(r)
    return fsum(prod(a[i][sigma[i]] for i in r) for sigma in s)

# if __name__ == '__main__':
#     from pprint import pprint as pp
#
#     for a in (
#             [
#                 [1, 2],
#                 [3, 4]],
#
#             [
#                 [1, 2, 3, 4],
#                 [4, 5, 6, 7],
#                 [7, 8, 9, 10],
#                 [10, 11, 12, 13]],
#
#             [
#                 [0, 1, 2, 3, 4],
#                 [5, 6, 7, 8, 9],
#                 [10, 11, 12, 13, 14],
#                 [15, 16, 17, 18, 19],
#                 [20, 21, 22, 23, 24]],
#     ):
#         print('')
#         pp(a)
#         print(f'Perm: {perm(a)};')
#         print(f'Length of Perm: {len(perm(a))};')

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

def build_XY(input_list,output_list=None,index=None):
    num_outputs = len(input_list)
    if output_list is not None:
        assert num_outputs == len(output_list)
        Y = np.vstack(output_list)
    else:
        Y = None

    if index is not None:
        assert len(index) == num_outputs
        I = np.hstack( [np.repeat(j,_x.shape[0]) for _x,j in zip(input_list,index)] )
    else:
        I = np.hstack( [np.repeat(j,_x.shape[0]) for _x,j in zip(input_list,range(num_outputs))] )

    X = np.vstack(input_list)
    X = np.hstack([X,I[:,None]])

    return X,Y,I[:,None]#slices


def nearest_neighbour_sq_dist(trajectories):
    n_birds = np.shape(trajectories)[0]
    n_timesteps = np.shape(trajectories)[-1]
    NN_sq_dist_array = np.zeros((n_birds, n_timesteps))
    for t in range(n_timesteps):
        for b in range(n_birds):
            intermediate1_array = (trajectories[b,:,t] - trajectories[:b,:,t])**2
            intermediate2_array = (trajectories[b, :, t] - trajectories[b+1:, :, t]) ** 2
            intermediate_array = np.concatenate((intermediate1_array,intermediate2_array), axis=0)
            NN_sq_dist_array[b,t] = min(np.sum(intermediate_array,axis=1))
    return NN_sq_dist_array


def array_save(array, filename,folder_save_location):
    output_filename = f"{folder_save_location}/{filename}"
    with open(output_filename, 'wb') as f:
        pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


def array_load(absolute_file_path):
    with open(absolute_file_path, 'rb') as f:
        array = pickle.load(f)
    return array


def NN_and_speeds_histogram_plotter(NN_dist_array, speeds):
    fig = plt.figure(figsize=(9., 4.))
    outer_grid = gridspec.GridSpec(2, 1, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.3)

    left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
    right_cell = outer_grid[1].subgridspec(1, 1)

    ax0 = fig.add_subplot(left_cell[:, :])
    ax1 = fig.add_subplot(right_cell[:, :])

    n_bins = 100
    # ax0.hist(NN_dist_array[:, 0], n_bins, histtype='step', fill=False, density=True, label="In the first frame")
    ax0.hist(NN_dist_array.flatten(), n_bins, histtype='step', fill=False, density=True, label="Over all the frames")
    ax0.set_xlabel("Distance to Nearest Neighbour")
    # ax1.hist(speeds[:, 0], n_bins, histtype='step', fill=False, density=True, label="In the first frame")
    ax1.hist(speeds.flatten(), n_bins, histtype='step', fill=False, density=True, label="Over all the frames")
    ax1.set_xlabel("Speed")
    fig.legend()
    plt.show()
    return


def break_up_array(absolute_file_path, number_of_sections):
    trajectories = array_load(absolute_file_path)
    original_n_birds = np.shape(trajectories)[0]
    original_n_axis = np.shape(trajectories)[1]
    original_n_time_steps = np.shape(trajectories)[2]
    list_of_sub_arrays = np.array_split(trajectories, number_of_sections, 2)
    for index, sub_array in enumerate(list_of_sub_arrays):
        n_birds = np.shape(sub_array)[0]
        n_axis = np.shape(sub_array)[1]
        n_time_steps = np.shape(sub_array)[2]
        array_save(sub_array, f"trajectory_{n_birds}b_{n_time_steps}t_{index}", f"trajectories_from_{original_n_birds}b_{original_n_time_steps}t")
    return


def presentation_graph_plotter(trajectory1, trajectory2, n_split, n_length, fraction, length):
    """Function to make figure for report."""

    X, Y = GPy_get_X(trajectory1, keep_length=False, output_dim=3, length = length)
    Xprime, Yprime = GPy_get_X(trajectory2, keep_length=False, output_dim=3, length= length)

    # n_split = int(np.floor(np.shape(trajectory1)[-1]/2))
    # n_length = 3

    Y1, Y2, Y_mask = trajectory_splitter(trajectory1, n_split, n_length)
    Y1prime, Y2prime, Yprime_mask = trajectory_splitter(trajectory2, n_split, n_length)
    # X1, X2, X_mask = trajectory_splitter(X.T, n_split, n_length)
    X1, X2, X_mask = trajectory_splitter(X.T, n_split, 0)
    # X1, X2, X_mask = trajectory_splitter(X.T, n_split, 3*n_length)
    X1prime, X2prime, Xprime_mask = trajectory_splitter(X.T, n_split, n_length)
    X1, X2, X_mask = X1.T, X2.T, X_mask.T
    X2 = np.concatenate((X2, np.array([[length*1.01]])), axis=0)
    X1prime, X2prime, Xprime_mask = X1prime.T, X2prime.T, Xprime_mask.T

    throwaway1, GP1 = multi_dimensional_gaussian_plotter(Y1, extension_ratio=0., length=n_split*0.01/fraction, n_dimensions=3, fraction=1.)
    throwaway2, GP2 = multi_dimensional_gaussian_plotter(Y1prime, extension_ratio=0., length=n_split*0.01/fraction, n_dimensions=3, fraction=1.)

    # assert((intermediate1 == X1).all())
    # assert((intermediate2 == X1prime).all())

    Y1 = Y1[None, :, :] # treating each trajectory fragment as a separate bird.
    Y2 = Y2[None, :, :]
    Y1prime = Y1[None, :, :]
    Y2prime = Y2prime[None, :, :]
    # Y_mask = Y_mask[None, :, :]
    # Yprime_mask = Yprime_mask[None, :, :]

    # trajectories = np.ma.concatenate((Y_mask,Yprime_mask), axis=0)
    # print(f'The shape of the trajectories before is [2,{np.shape(trajectory1)}]\nThe shape of Y_mask is {np.shape(Y_mask)}\nThe shape of trajectories is {np.shape(trajectories)}')
    fig2 = plt.figure(figsize=(5.5, 4.5))
    outer_grid2 = gridspec.GridSpec(2, 1, figure=fig2, left=0.15, right=0.975, top=0.975, wspace=0.3, hspace=0.05)
    ax = fig2.add_subplot(outer_grid2[:,:])
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylabel("Normalised\nResiduals")
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    fig = plt.figure(figsize=(10.,4.5))
    outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975,top=0.975, wspace=0.3)

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

    axx = fig.add_subplot(upper_right_cell[0])
    axy = fig.add_subplot(upper_right_cell[1], sharex=axx)
    axz = fig.add_subplot(upper_right_cell[2], sharex=axx)
    # ax2 = fig.add_subplot(lower_right_cell[0], sharex=axx)
    # ax3 = fig.add_subplot(lower_right_cell[1], sharex=axx)
    ax2 = fig2.add_subplot(outer_grid2[0])
    ax3 = fig2.add_subplot(outer_grid2[1], sharex=ax2)

    ax.set_xlabel('Z (m)')
    ax.set_ylabel('Y (m)')

    ax.plot(Y_mask[2,:],Y_mask[1,:], 'k-')
    ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b-')


    # inverse_mask = ~np.array(np.ma.getmask(Y_mask), dtype=bool)
    # Y_no_mask = np.ma.masked_array(Y_mask, ~np.ma.getmask(Y_mask))
    # Yprime_no_mask = np.ma.masked_array(Yprime_mask, ~np.ma.getmask(Yprime_mask))
    # ax.plot(Y_no_mask[2,:],Y_no_mask[1,:], 'k--')
    # ax.plot(Yprime_no_mask[2, :], Yprime_no_mask[1, :], 'b--')


    axins = ax.inset_axes([0.175, 0.15, 0.375, 0.35])
    axins.plot(Yprime_mask[2, :], Yprime_mask[0, :], 'b-')
    axins.plot(Y_mask[2,:],Y_mask[0,:], 'k-')
    # axins.plot(Yprime_no_mask[2, :], Yprime_no_mask[0, :], 'b--')
    # axins.plot(Y_no_mask[2,:],Y_no_mask[0,:], 'k--')
    axins.set_xlabel('Z (m)')
    axins.set_ylabel('X (m)')

    Y_mask.mask = np.ma.nomask
    Yprime_mask.mask = np.ma.nomask
    ax.plot(Y_mask[2,:],Y_mask[1,:], 'k:')
    ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b:')
    axins.plot(Yprime_mask[2, :], Yprime_mask[0, :], 'b:')
    axins.plot(Y_mask[2,:],Y_mask[0,:], 'k:')

    ax.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    axins.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

    axx.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    axy.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    axz.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="both", labelbottom=False)
    axx.tick_params(axis="both", labelbottom=False)
    axy.tick_params(axis="both", labelbottom=False)
    # axz.tick_params(axis="both", labelbottom=False)

    X_list = [X1, X1, X1]
    Xprime_list = [X1prime, X1prime, X1prime]
    slices = GPy.util.multioutput.get_slices(X_list)
    slicesprime = GPy.util.multioutput.get_slices(Xprime_list)
    assert((slices == slicesprime))

    axx.set_ylabel('X (m)')
    axy.set_ylabel('Y (m)')
    axz.set_ylabel('Z (m)')
    axz.set_xlabel('Time (s)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('\nIncorrect')
    ax2.set_ylabel('\nCorrect')

    xlim = [(n_split) * 0.01 / fraction, length*1.01]
    axx.set_xlim(xlim)
    xlim2 = [(n_split+n_length)*0.01/fraction + 0.1, length*1.01] # Can't be bothered to do the maths to work out why the 0.1 works. multiply by 0.01 for the timesteps.divid by fraction for beginning.
    ax2.set_xlim(xlim2)
    # ax2.set_ylim(-4,3.5)
    # ax3.set_ylim(-11,15)
    # ax3.set_xlim(xlim)

    # GP1.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)], which_data_rows=slices[0], legend=False, marker='+')
    # GP1.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], which_data_rows=slices[1], legend=False, marker='+')
    # GP1.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], which_data_rows=slices[2], legend=False, marker='+')
    # GP2.plot(plot_limits=xlim, ax=axx, fixed_inputs=[(1, 0)], which_data_rows=slicesprime[0], legend=False, marker='+')
    # GP2.plot(plot_limits=xlim, ax=axy, fixed_inputs=[(1, 1)], which_data_rows=slicesprime[1], legend=False, marker='+')
    # GP2.plot(plot_limits=xlim, ax=axz, fixed_inputs=[(1, 2)], which_data_rows=slicesprime[2], legend=False, marker='+')


    num_samples = 5

    Xnewx = np.concatenate((X2, np.ones_like(X2)-1), axis=1)
    noise_dict = {'output_index': Xnewx[:, 1:].astype(int)}
    Xpred, Xvar = GP1.predict(Xnewx,Y_metadata=noise_dict)
    Xnewx = np.concatenate((X2, np.ones_like(X2)-1), axis=1)
    noise_dict = {'output_index': Xnewx[:, 1:].astype(int)}
    Xpred_prime, Xprime_var = GP2.predict(Xnewx,Y_metadata=noise_dict)
    Xquantiles = np.array(GP1.predict_quantiles(Xnewx,Y_metadata=noise_dict))
    Xsamples = GP1.posterior_samples(Xnewx, Y_metadata=noise_dict, size=num_samples)
    Xprime_quantiles = np.array(GP2.predict_quantiles(Xnewx,Y_metadata=noise_dict))
    Xprime_samples = GP2.posterior_samples(Xnewx, Y_metadata=noise_dict, size=num_samples)

    Ynewx = np.concatenate((X2, np.ones_like(X2)), axis=1)
    noise_dict = {'output_index': Ynewx[:, 1:].astype(int)}
    Ypred, Yvar = GP1.predict(Ynewx,Y_metadata=noise_dict)
    Ynewx = np.concatenate((X2, np.ones_like(X2)), axis=1)
    noise_dict = {'output_index': Ynewx[:, 1:].astype(int)}
    Ypred_prime, Yprime_var = GP2.predict(Ynewx,Y_metadata=noise_dict)
    Yquantiles = np.array(GP1.predict_quantiles(Ynewx,Y_metadata=noise_dict))
    Ysamples = GP1.posterior_samples(Ynewx, Y_metadata=noise_dict, size=num_samples)
    Yprime_quantiles = np.array(GP2.predict_quantiles(Ynewx,Y_metadata=noise_dict))
    Yprime_samples = GP2.posterior_samples(Ynewx, Y_metadata=noise_dict, size=num_samples)


    Znewx = np.concatenate((X2, np.ones_like(X2)+1), axis=1)
    noise_dict = {'output_index': Znewx[:, 1:].astype(int)}
    Zpred, Zvar = GP1.predict(Znewx,Y_metadata=noise_dict)
    Znewx = np.concatenate((X2, np.ones_like(X2)+1), axis=1)
    noise_dict = {'output_index': Znewx[:, 1:].astype(int)}
    Zpred_prime, Zprime_var = GP2.predict(Znewx,Y_metadata=noise_dict)
    Zquantiles = np.array(GP1.predict_quantiles(Znewx,Y_metadata=noise_dict))
    Zsamples = GP1.posterior_samples(Znewx, Y_metadata=noise_dict, size=num_samples)
    Zprime_quantiles = np.array(GP2.predict_quantiles(Znewx,Y_metadata=noise_dict))
    Zprime_samples = GP2.posterior_samples(Znewx, Y_metadata=noise_dict, size=num_samples)


    # axx.fill_between(x=Xnewx[:, 0], y1=Xquantiles[0, :, 0], y2=Xquantiles[1, :, 0], color='black', alpha=0.05)
    # axx.fill_between(x=Xnewx[:, 0], y1=Xprime_quantiles[0, :, 0], y2=Xprime_quantiles[1, :, 0], color='blue', alpha=0.05)
    # axy.fill_between(x=Xnewx[:, 0], y1=Yquantiles[0, :, 0], y2=Yquantiles[1, :, 0], color='black', alpha=0.05)
    # axy.fill_between(x=Xnewx[:, 0], y1=Yprime_quantiles[0, :, 0], y2=Yprime_quantiles[1, :, 0], color='blue',alpha=0.05)
    # axz.fill_between(x=Xnewx[:, 0], y1=Zquantiles[0, :, 0], y2=Zquantiles[1, :, 0], color='black', alpha=0.05)
    # axz.fill_between(x=Xnewx[:, 0], y1=Zprime_quantiles[0, :, 0], y2=Zprime_quantiles[1, :, 0], color='blue',alpha=0.05)

    axx.fill_between(x=Xnewx[:, 0], y1=Xpred[:,0]-Xvar[:,0]**0.5, y2=Xpred[:,0]+Xvar[:,0]**0.5, color='black', alpha=0.05)
    axx.fill_between(x=Xnewx[:, 0], y1=Xpred_prime[:,0]-Xprime_var[:,0]**0.5, y2=Xpred_prime[:,0]+Xprime_var[:,0]**0.5, color='blue', alpha=0.05)
    axy.fill_between(x=Xnewx[:, 0], y1=Ypred[:,0]-Yvar[:,0]**0.5, y2=Ypred[:,0]+Yvar[:,0]**0.5, color='black', alpha=0.05)
    axy.fill_between(x=Xnewx[:, 0], y1=Ypred_prime[:,0]-Yprime_var[:,0]**0.5, y2=Ypred_prime[:,0]+Yprime_var[:,0]**0.5, color='blue', alpha=0.05)
    axz.fill_between(x=Xnewx[:, 0], y1=Zpred[:,0]-Zvar[:,0]**0.5, y2=Zpred[:,0]+Zvar[:,0]**0.5, color='black', alpha=0.05)
    axz.fill_between(x=Xnewx[:, 0], y1=Zpred_prime[:,0]-Zprime_var[:,0]**0.5, y2=Zpred_prime[:,0]+Zprime_var[:,0]**0.5, color='blue', alpha=0.05)

    axx.plot(X2, Xpred,  'k--', alpha=0.5)
    axx.plot(X2, Xpred_prime, 'b--', alpha=0.5)
    axy.plot(X2, Ypred, 'k--', alpha=0.5)
    axy.plot(X2, Ypred_prime, 'b--', alpha=0.5)
    axz.plot(X2, Zpred, 'k--', alpha=0.5)
    axz.plot(X2, Zpred_prime, 'b--', alpha=0.5)


    X2 = X2[n_length:-1,:]
    Xpred = Xpred[n_length:-1,:]
    Xvar = Xvar[n_length:-1, :]
    Ypred = Ypred[n_length:-1,:]
    Yvar = Yvar[n_length:-1, :]
    Zpred = Zpred[n_length:-1,:]
    Zvar = Zvar[n_length:-1, :]
    Xpred_prime = Xpred_prime[n_length:-1,:]
    Xprime_var = Xprime_var[n_length:-1, :]
    Ypred_prime = Ypred_prime[n_length:-1,:]
    Yprime_var = Yprime_var[n_length:-1, :]
    Zpred_prime = Zpred_prime[n_length:-1,:]
    Zprime_var = Zprime_var[n_length:-1, :]
    # X2 = X2[n_length:,:]
    # Xpred = Xpred[n_length:,:]
    # Xvar = Xvar[n_length:, :]
    # Ypred = Ypred[n_length:,:]
    # Yvar = Yvar[n_length:, :]
    # Zpred = Zpred[n_length:,:]
    # Zvar = Zvar[n_length:, :]
    # Xpred_prime = Xpred_prime[n_length:,:]
    # Xprime_var = Xprime_var[n_length:, :]
    # Ypred_prime = Ypred_prime[n_length:,:]
    # Yprime_var = Yprime_var[n_length:, :]
    # Zpred_prime = Zpred_prime[n_length:,:]
    # Zprime_var = Zprime_var[n_length:, :]
    axx.scatter(X2[:, 0], Y2[0, 0, :],  color='k', marker='x', s=50*(72./fig.dpi)**2)
    axx.scatter(X2[:, 0], Y2prime[0, 0, :], color='b', marker='x', s=50*(72./fig.dpi)**2)
    axy.scatter(X2[:, 0], Y2[0, 1, :],  color='k', marker='1', s=50*(72./fig.dpi)**2)
    axy.scatter(X2[:, 0], Y2prime[0, 1, :], color='b', marker='1', s=50*(72./fig.dpi)**2)
    axz.scatter(X2[:, 0], Y2[0, 2, :],  color='k', marker='+', s=50*(72./fig.dpi)**2)
    axz.scatter(X2[:, 0], Y2prime[0, 2, :], color='b', marker='+', s=50*(72./fig.dpi)**2)
    # axx.scatter(X2[:-1, 0], Y2[0, 0, :-1],  color='k', marker='x', s=50*(72./fig.dpi)**2)
    # axx.scatter(X2[:-1, 0], Y2prime[0, 0, :-1], color='b', marker='x', s=50*(72./fig.dpi)**2)
    # axy.scatter(X2[:-1, 0], Y2[0, 1, :-1],  color='k', marker='1', s=50*(72./fig.dpi)**2)
    # axy.scatter(X2[:-1, 0], Y2prime[0, 1, :-1], color='b', marker='1', s=50*(72./fig.dpi)**2)
    # axz.scatter(X2[:-1, 0], Y2[0, 2, :-1],  color='k', marker='+', s=50*(72./fig.dpi)**2)
    # axz.scatter(X2[:-1, 0], Y2prime[0, 2, :-1], color='b', marker='+', s=50*(72./fig.dpi)**2)

    # axx.plot(X2[:, 0], Y2[0, 0, :],  color='k', linestyle='-', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    # axx.plot(X2[:, 0], Y2prime[0, 0, :], color='b', linestyle='-', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    # axy.plot(X2[:, 0], Y2[0, 1, :],  color='k', linestyle='-', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    # axy.plot(X2[:, 0], Y2prime[0, 1, :], color='b', linestyle='-', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    # axz.plot(X2[:, 0], Y2[0, 2, :],  color='k', linestyle='-', alpha=0.5) # , s=50*(72./fig.dpi)**2)
    # axz.plot(X2[:, 0], Y2prime[0, 2, :], color='b', linestyle='-', alpha=0.5) # , s=50*(72./fig.dpi)**2)

    deltay = max(axx.get_ylim()[1]-axx.get_ylim()[0],axy.get_ylim()[1]-axy.get_ylim()[0],axz.get_ylim()[1]-axz.get_ylim()[0])
    # axx.set_ylim([axx.get_ylim()[1]-deltay, axx.get_ylim()[1]])
    # axy.set_ylim([axy.get_ylim()[1] - deltay, axy.get_ylim()[1]])
    # axz.set_ylim([axz.get_ylim()[1] - deltay, axz.get_ylim()[1]])

    alpha =0.5
    Xresiduals = (Y2[0, 0, :] - Xpred[:,0])/Xvar[:,0]**0.5
    Yresiduals = (Y2[0, 1, :] - Ypred[:,0])/Yvar[:,0]**0.5
    Zresiduals = (Y2[0, 2, :] - Zpred[:,0])/Zvar[:,0]**0.5
    ax2.scatter(X2[:, 0], Xresiduals, color='k', marker='x', alpha=alpha)# , s=(72./fig.dpi)**2)
    ax2.scatter(X2[:, 0], Yresiduals, color='k', marker='1', alpha=alpha)# , s=(72./fig.dpi)**2)
    ax2.scatter(X2[:, 0], Zresiduals, color='k', marker='+', alpha=alpha)# , s=(72./fig.dpi)**2)
    # ax2.plot(X2[:, 0], Xresiduals, color='k', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ax2.plot(X2[:, 0], Yresiduals, color='k', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ax2.plot(X2[:, 0], Zresiduals, color='k', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)

    Xprime_residuals = (Y2prime[0, 0, :] - Xpred_prime[:, 0])/Xprime_var[:,0]**0.5
    Yprime_residuals = (Y2prime[0, 1, :] - Ypred_prime[:, 0])/Yprime_var[:,0]**0.5
    Zprime_residuals = (Y2prime[0, 2, :] - Zpred_prime[:, 0])/Zprime_var[:,0]**0.5
    ax2.scatter(X2[:, 0], Xprime_residuals, color='b', marker='x', alpha=alpha)  # , s=(72./fig.dpi)**2)
    ax2.scatter(X2[:, 0], Yprime_residuals, color='b', marker='1', alpha=alpha)  # , s=(72./fig.dpi)**2)
    ax2.scatter(X2[:, 0], Zprime_residuals, color='b', marker='+', alpha=alpha)  # , s=(72./fig.dpi)**2)
    # ax2.plot(X2[:, 0], Xprime_residuals, color='b', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ax2.plot(X2[:, 0], Yprime_residuals, color='b', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ax2.plot(X2[:, 0], Zprime_residuals, color='b', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)


    bad_Xresiduals = (Y2prime[0, 0, :] - Xpred[:, 0])/Xvar[:,0]**0.5
    bad_Yresiduals = (Y2prime[0, 1, :] - Ypred[:, 0])/Yvar[:,0]**0.5
    bad_Zresiduals = (Y2prime[0, 2, :] - Zpred[:, 0])/Zvar[:,0]**0.5
    ax3.scatter(X2[:, 0], bad_Xresiduals, color='k', marker='x', alpha=alpha)# , s=(72./fig.dpi)**2)
    ax3.scatter(X2[:, 0], bad_Yresiduals, color='k', marker='1', alpha=alpha)# , s=(72./fig.dpi)**2)
    ax3.scatter(X2[:, 0], bad_Zresiduals, color='k', marker='+', alpha=alpha)# , s=(72./fig.dpi)**2)
    # ax3.plot(X2[:, 0], bad_Xresiduals, color='k', linestyle=':', alpha=0.5)# , s=(72./fig.dpi)**2)
    # ax3.plot(X2[:, 0], bad_Yresiduals, color='k', linestyle='--', alpha=0.5)# , s=(72./fig.dpi)**2)
    # ax3.plot(X2[:, 0], bad_Zresiduals, color='k', linestyle='-.', alpha=0.5)# , s=(72./fig.dpi)**2)


    bad_Xprime_residuals = (Y2[0, 0, :] - Xpred_prime[:,0])/Xprime_var[:,0]**0.5
    bad_Yprime_residuals = (Y2[0, 1, :] - Ypred_prime[:,0])/Yprime_var[:,0]**0.5
    bad_Zprime_residuals = (Y2[0, 2, :] - Zpred_prime[:,0])/Zprime_var[:,0]**0.5
    ax3.scatter(X2[:, 0], bad_Xprime_residuals, color='b', marker='x', alpha=alpha)  # , s=(72./fig.dpi)**2)
    ax3.scatter(X2[:, 0], bad_Yprime_residuals, color='b', marker='1', alpha=alpha)  # , s=(72./fig.dpi)**2)
    ax3.scatter(X2[:, 0], bad_Zprime_residuals, color='b', marker='+', alpha=alpha)  # , s=(72./fig.dpi)**2)
    # ax3.plot(X2[:, 0], bad_Xprime_residuals, color='b', linestyle=':', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ax3.plot(X2[:, 0], bad_Yprime_residuals, color='b', linestyle='--', alpha=0.5)  # , s=(72./fig.dpi)**2)
    # ax3.plot(X2[:, 0], bad_Zprime_residuals, color='b', linestyle='-.', alpha=0.5)  # , s=(72./fig.dpi)**2)
    ax2.plot(xlim2, [0,0], color='grey', alpha=0.3)
    ax3.plot(xlim2, [0, 0], color='grey', alpha=0.3)
    ax2.fill_between(xlim2, [-1, -1], [1,1],color='grey', alpha=0.3)
    ax3.fill_between(xlim2, [-1, -1], [1,1],color='grey', alpha=0.3)

    # ylabels = axz.get_yticklabels()
    # print(ylabels)

    # X2_list = [X2, X2, X2]
    #
    # print(GP3.log_predictive_density())
    return fig.show()


def plot_2d_projections(trajectories, trajectories_altered, bold_trajectories=[0]):
    n_birds, n_axis, n_timesteps = np.shape(trajectories)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig = plt.figure(figsize=(9., 4.))
    outer_grid = gridspec.GridSpec(1, 2, figure=fig, left=0.1, right=0.975, top=0.975, wspace=0.23)

    left_cell = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_grid[0])
    ax2 = fig.add_subplot(left_cell[:, :])
    right_cell = outer_grid[1].subgridspec(2, 1, hspace=0.3)
    ax1 = fig.add_subplot(right_cell[1], sharex=ax2)
    ax3 = fig.add_subplot(right_cell[0], sharey=ax1)# , sharex=ax2)
    ax1.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis="both", direction="in", bottom=True, top=True, left=True, right=True)

    ax1.set_ylabel("Y (m)")
    ax1.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_xlabel("Z (m)")
    alpha = 0.1
    alpha2 = 1
    for i in range(n_birds):
        ax1.plot(trajectories[i, 0, :], trajectories[i, 1, :], alpha=alpha)
        ax2.plot(trajectories[i, 0, :], trajectories[i, 2, :], alpha=alpha)
        ax3.plot(trajectories[i, 2, :], trajectories[i, 1, :], alpha=alpha)
    if len(trajectories_altered)!=0:
        for k in bold_trajectories:
            for j in range(len(trajectories_altered[k])):
                ax1.plot(trajectories[trajectories_altered[k][j], 0, :], trajectories[trajectories_altered[k][j], 1, :],
                         alpha=alpha2)
                ax2.plot(trajectories[trajectories_altered[k][j], 0, :], trajectories[trajectories_altered[k][j], 2, :],
                         alpha=alpha2)
                ax3.plot(trajectories[trajectories_altered[k][j], 2, :], trajectories[trajectories_altered[k][j], 1, :],
                         alpha=alpha2)
    fig.show()
    return


def LML_landscape(input_times, data, gp, noise_lower=-9., noise_upper=9., time_lower=-9., time_upper=9., rbf_variance_lower=-6, rbf_variance_upper=1):
    ''' Plot LML landscape'''

    not_rbf_variance = False
    not_noise = True
    not_time_scale = False


    fig1 = plt.figure()

    optimised_rbf_time_scale = np.log10(gp.rbf.lengthscale[0])
    optimised_gaussian_noise = np.log10(gp.Gaussian_noise.variance[0])
    optimised_rbf_variance = np.log10(gp.rbf.variance[0])
    # gaussian_noise = np.logspace(0.5*np.log(optimised_gaussian_noise)+noise_lower, 0.5*np.log(optimised_gaussian_noise)+noise_upper, 50)
    # time_scale = np.logspace(np.log(rbf_time_scale)+time_lower, np.log(rbf_time_scale)+time_upper, 50)


    if not_rbf_variance:
        rbf_variance = gp.rbf.variance[0]
        gaussian_noise = np.logspace(noise_lower, noise_upper, 50)
        time_scale = np.logspace(time_lower, time_upper, 51)
        LML = np.zeros((len(gaussian_noise), len(time_scale)))
        for i in range(len(gaussian_noise)):
            for j in range(len(time_scale)):
                gp = GPy.models.GPRegression(input_times, data, GPy.kern.RBF(input_dim=1, variance=rbf_variance, lengthscale=time_scale[j]))
                gp.Gaussian_noise.variance = gaussian_noise[i]**2
                LML[i,j] = gp.log_likelihood()
        # LML = np.array(LML).T
        vmin, vmax = (-LML).min(), (-LML).max()
        level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)

        plt.contour(time_scale, gaussian_noise, -LML,
                    levels=level, cmap='jet')  # norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.scatter(optimised_rbf_time_scale, optimised_gaussian_noise, color='k', marker='x')
        plt.xlabel("Time-scale")
        plt.ylabel("Gaussian noise")
        plt.title(f"LML for rbf kernel, variance: {rbf_variance}")
    elif not_noise:
        gaussian_noise = gp.Gaussian_noise.variance[0] ** 0.5
        # time_scale = np.logspace(time_lower, time_upper, 51)
        time_scale = np.logspace(optimised_rbf_time_scale+time_lower, optimised_rbf_time_scale+time_upper, 51)
        # rbf_variance = np.logspace(rbf_variance_lower, rbf_variance_upper, 50)
        rbf_variance = np.logspace(optimised_rbf_variance+rbf_variance_lower, optimised_rbf_variance+rbf_variance_upper, 50)
        LML = np.zeros((len(rbf_variance), len(time_scale)))
        for i in range(len(rbf_variance)):
            for j in range(len(time_scale)):
                gp = GPy.models.GPRegression(input_times, data, GPy.kern.RBF(input_dim=1, variance=rbf_variance[i],
                                                                             lengthscale=time_scale[j]))
                gp.Gaussian_noise.variance = gaussian_noise ** 2
                LML[i, j] = gp.log_likelihood()
        # LML = np.array(LML).T
        vmin, vmax = (-LML).min(), (-LML).max()
        level = np.logspace(np.log10(vmin), np.log10(vmax), 50) # np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)

        plt.contour(time_scale, rbf_variance, -LML,
                    levels=level, cmap='jet')  # norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.xlabel("Time-scale")
        plt.ylabel("RBF Variance")
        plt.scatter(optimised_rbf_time_scale, optimised_rbf_variance, color='k', marker='x')
        plt.title(f"LML for rbf, Gaussian noise: {gaussian_noise}")
    elif not_time_scale:
        time_scale = gp.rbf.lengthscale[0]
        gaussian_noise = np.logspace(noise_lower, noise_upper, 51)
        rbf_variance = np.logspace(rbf_variance_lower, rbf_variance_upper, 50)
        LML = np.zeros((len(rbf_variance), len(gaussian_noise)))
        for i in range(len(rbf_variance)):
            for j in range(len(gaussian_noise)):
                gp = GPy.models.GPRegression(input_times, data, GPy.kern.RBF(input_dim=1, variance=rbf_variance[i],
                                                                             lengthscale=time_scale))
                gp.Gaussian_noise.variance = gaussian_noise[j] ** 2
                LML[i, j] = gp.log_likelihood()
        LML = np.array(LML).T
        vmin, vmax = (-LML).min(), (-LML).max()
        level = np.around(np.logspace(np.log10(vmin), np.log10(vmax), 50), decimals=1)

        plt.contour(rbf_variance, gaussian_noise, -LML,
                    levels=level, cmap='jet')  # norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.xlabel("RBF Variance")
        plt.ylabel("Gaussian Noise")
        plt.scatter(10**optimised_rbf_variance, 10**optimised_gaussian_noise, color='k', marker='x')
        plt.plot(optimised_rbf_variance, optimised_gaussian_noise, 'b+')
        plt.title(f"LML for rbf, Time-scale: {time_scale}")
    plt.colorbar()
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()

    fig1.show()





if __name__ == '__main__':
    # trajectories = array_unpacker(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\no_rounding_trajectories_arrays0\300birds501timesteps20210201-192940")
    # NN_sq_dist_array = nearest_neighbour_sq_dist(trajectories)
    # array_save(NN_sq_dist_array, "NN_sq_dist_300b_501t", "NearestNeighbourSqDistances")
    #####
    # folder_reformatter(r"data", folder_save_location="no_rounding_trajectories_arrays1",plot_trajectories=True)
    #####
    # trajectories = array_unpacker(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\no_rounding_trajectories_arrays1\500birds4000timesteps20210202-004516")
    # NN_sq_dist_array = nearest_neighbour_sq_dist(trajectories)
    # array_save(NN_sq_dist_array, "NN_sq_dist_500b_4001t", "NearestNeighbourSqDistances")
    #####
    # velocities = differentiater(trajectories, time_interval=0.01)
    # array_save(velocities, "velocities300b_501t", "NearestNeighbourSqDistances")
    #####
    # break_up_array(r"C:\Users\Owner\OneDrive - Durham University\Level 4 Project\Code\no_rounding_trajectories_arrays1\250birds8001timesteps20210202-035622", 16)
    folder_unpacker(r"C:\\Users\\Owner\\OneDrive - Durham University\\Level 4 Project\\Code\\Results16traj_12_TD\\trajectories_for_analysing", plot_trajectories=True)
    print("--- %s ---" % seconds_to_str((time() - total_time)))




# Work that I think is now useless:
# """Function to make figure for report."""
#     #TODO I need to find X2, Y2, X2prime, Y2prime, in order to do that I need to find the correct value for n_split
#
#     # X, Y = GPy_get_X(trajectory1, keep_length=False, output_dim=3, length = length)
#     # Xprime, Yprime = GPy_get_X(trajectory2, keep_length=False, output_dim=3, length= length)
#
#     # n_split = int(np.floor(np.shape(trajectory1)[-1]/2))
#     # n_length = 3
#
#     # Y1, Y2, Y_mask = trajectory_splitter(trajectory1, n_split, n_length)
#     # Y1prime, Y2prime, Yprime_mask = trajectory_splitter(trajectory2, n_split, n_length)
#     # X1, X2, X_mask = trajectory_splitter(X.T, n_split, n_length)
#     # X1prime, X2prime, Xprime_mask = trajectory_splitter(X.T, n_split, n_length)
#     # X1, X2, X_mask = X1.T, X2.T, X_mask.T
#     # X1prime, X2prime, Xprime_mask = X1prime.T, X2prime.T, Xprime_mask.T
#
#     input_trajectory_masked = input_list[0]
#     input_mask = np.ma.getmask(input_trajectory_masked)
#     input_trajectory = np.array(input_trajectory_masked[~input_mask].reshape(3, -1))
#     times_input_mask = input_mask[0, :]
#     times_input_masked = np.ma.masked_array(times_array, times_input_mask)
#     input_times = np.array(times_input_masked[~times_input_mask])
#     Y_List = GPy_reformat_3D(input_trajectory)
#     Y1 = Y_List
#     X_List = GPy_reformat_3D(input_times)
#     X1 = X_List
#     input_trajectory_maskedprime = input_list[1]
#     input_maskprime = np.ma.getmask(input_trajectory_maskedprime)
#     input_trajectoryprime = np.array(input_trajectory_maskedprime[~input_maskprime].reshape(3, -1))
#     times_input_maskprime = input_maskprime[0, :]
#     times_input_maskedprime = np.ma.masked_array(times_array, times_input_maskprime)
#     input_timesprime = np.array(times_input_maskedprime[~times_input_maskprime])
#     Y_Listprime = GPy_reformat_3D(input_trajectoryprime)
#     Y1prime = Y_Listprime
#     X_Listprime = GPy_reformat_3D(input_timesprime)
#     X1prime = X_Listprime
#
#     throwaway1, GP1 = multi_dimensional_gaussian_plotter(Y1, extension_ratio=0., length=n_split*0.1, n_dimensions=3, fraction=1.)
#     throwaway2, GP2 = multi_dimensional_gaussian_plotter(Y1prime, extension_ratio=0., length=n_split*0.1, n_dimensions=3, fraction=1.)
#
#     # assert((intermediate1 == X1).all())
#     # assert((intermediate2 == X1prime).all())
#
#     Y1 = Y1[None, :, :] # treating each trajectory fragment as a separate bird.
#     Y2 = Y2[None, :, :]
#     Y1prime = Y1[None, :, :]
#     Y2prime = Y2prime[None, :, :]
#     # Y_mask = Y_mask[None, :, :]
#     # Yprime_mask = Yprime_mask[None, :, :]
#
#     # trajectories = np.ma.concatenate((Y_mask,Yprime_mask), axis=0)
#     # print(f'The shape of the trajectories before is [2,{np.shape(trajectory1)}]\nThe shape of Y_mask is {np.shape(Y_mask)}\nThe shape of trajectories is {np.shape(trajectories)}')
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
#     ax.plot(Y_mask[2,:],Y_mask[1,:], 'k-')
#     ax.plot(Yprime_mask[2, :], Yprime_mask[1, :], 'b-')
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
#     xlim = [(n_split+n_length)*0.1 + 0.1, length]
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
import TrajectoryPlotter as TP
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from time import time, strftime, localtime
from datetime import timedelta, datetime

total_time = time()

# trajectories_arrays = TP.folder_unpacker("trajectories_as_arrays")
# TP.trajectory_plotter(trajectories_arrays[-1])
# TP.folder_reformatter("data2", folder_save_location="trajectories_as_arrays2", plot_trajectories=True)

trajectories = TP.array_unpacker("trajectories_as_arrays2/2birds1000timesteps20201126-111657")

TP.trajectory_plotter(trajectories)
trajectories = TP.array_fractional_reducer(trajectories, 1., 2)

sigma_f = 129**2.
l = 4.2 # lengthscale

# TP.GPR(trajectories[0, 2, :], n_star=500, sigma_f=sigma_f, l=l, n_samples=10, three_dimensional=False, length=10.)

# TP.GPyGPR(trajectories[0, 2, :], length=10., variance=13000., lengthscale=2.9)

# TP.GPy_compiled(trajectories[0, 2, :], length=10., input_dim=1, output_dim=1, plot=True, view_ratio=1.1, variance=13000., lengthscale=2.9, num_restarts=1)


length = 10.
n_test_points = 200
input_dim = 1
num_samples = 5
x_test = np.linspace(start=0., stop=length, num=n_test_points)
X_Test = x_test.reshape(n_test_points, input_dim)
for i in [0,1,2]:
    X, Y, gp = TP.GPy_compiled(trajectories[0, i, :], length=10., input_dim=1, output_dim=1, plot=True, view_ratio=1.1, variance=13000., lengthscale=2.9, num_restarts=1)
    TP.GPy_plotter(X, Y, gp, X_Test, num_samples)
    print("--- %s ---" % TP.seconds_to_str((time() - total_time)))


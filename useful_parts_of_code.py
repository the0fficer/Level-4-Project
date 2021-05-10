# trajectory_unpacker("data/Trajectories_13-11-20_10-48-47.dat")
# with open("trajectories_as_arrays/2birds101timesteps20201114-151347", "rb") as file:
#     trajectories = pickle.load(file)




# trajectories = read_trajectory("data/Trajectories_13-11-20_10-48-47.dat") # 2 birds
# trajectory_plotter(trajectories)
# trajectories[0, :, :], trajectories[1, :, :] = trajectory_switcher(trajectories[0, :, :], trajectories[1, :, :], 1)
# trajectories[0, :, :], trajectories[1, :, :] = trajectory_switcher(trajectories[0, :, :], trajectories[1, :, :], 50)


# trajectories = read_trajectory("data/Trajectories_13-11-20_10-50-56.dat") # 10 birds
# trajectories = read_trajectory("data/Trajectories_13-11-20_20-53-23.dat") # 1000 birds
#trajectories = read_trajectory("data/Trajectories_13-11-20_20-53-11.dat") # 2000 birds

# print("--- %s ---" % secondsToStr((time() - start_time)))

# trajectory_plotter(trajectories)

# trajectories = trajectory_error_correcter_improved(trajectories)

# trajectory_plotter(trajectories)

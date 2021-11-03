import numpy as np
import pickle

dim = 2
ls = 0.01
v_kernel = 1.0
obs_noise = 1e-6

M = 100

s = np.random.multivariate_normal(np.zeros(dim), 1 / (ls**2) * np.identity(dim), M)
b = np.random.uniform(0, 2 * np.pi, M)

random_features = {"M":M, "length_scale":ls, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}
pickle.dump(random_features, open("aux_files/RF_M_" + str(M) + ".pkl", "wb"))

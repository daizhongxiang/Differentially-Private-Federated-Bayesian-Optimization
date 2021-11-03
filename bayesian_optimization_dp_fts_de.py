# -*- coding: utf-8 -*-

import numpy as np
import GPy
from helper_funcs_dp_fts_de import UtilityFunction, acq_max
import pickle
import itertools
import time

class dp_fts_de(object):
    def __init__(self, f, pbounds, gp_opt_schedule, ARD=False, gp_mcmc=False, log_file=None, \
                 M=50, N=50, random_features=None, pt=None, M_target=100, verbose=1, partitions=None, partition_assignment=None):
        """
        """

        self.partitions = partitions
        self.partition_assignment = partition_assignment

        self.M = M
        self.N = N
        self.random_features = random_features
        self.M_target = M_target
        self.pt = pt
        
        self.ARD = ARD    
        self.log_file = log_file
        
        self.pbounds = pbounds
        
        self.incumbent = None
        
        self.keys = list(pbounds.keys())
        self.dim = len(pbounds)
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        
        self.gp_mcmc = gp_mcmc
        self.gp = None
        self.gp_params = None
        self.gp_opt_schedule = gp_opt_schedule

        self.util = None

        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[]}

        self.verbose = verbose


    def init(self, init_points, agent_ind):
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.partitions[self.partition_assignment]]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            y = self.f(x, agent_ind)

            y_init.append(y)
            self.res['all']['init_values'].append(y)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        print("init X: {0}".format(self.X))
        print("init Y: {0}".format(self.Y))

        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y}
        self.res['all']['init'] = init


    def sample_w(self):
        M = self.M

        s = self.random_features["s"]
        b = self.random_features["b"]
        v_kernel = self.random_features["v_kernel"]
        obs_noise = self.random_features["obs_noise"]

        Phi = np.zeros((self.X.shape[0], M))
        for i, x in enumerate(self.X):
            x = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

            features = features / np.sqrt(np.inner(features, features))
            features = np.sqrt(v_kernel) * features

            Phi[i, :] = features

        Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
        Sigma_t_inv = np.linalg.inv(Sigma_t)
        nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

        try:
            w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
        except np.linalg.LinAlgError:
            w_sample = np.random.rand(1, self.M) - 0.5
        

        return w_sample

    def maximize(self, n_iter=25, init_points=5, all_w_t=None, init_flag=False, iter_fed=0, agent_ind=0):

        self.util_ts = UtilityFunction(kind="ts")
        self.util_dp_fts_de = UtilityFunction(kind="dp_fts_de")


        if init_flag:
            self.init(init_points, agent_ind)

        if init_flag:
            self.gp = GPy.models.GPRegression(self.X, self.Y.reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=1.0, variance=0.1, ARD=self.ARD))
            self.gp["Gaussian_noise.variance"][0] = 1e-6

            w_return = self.sample_w()
            return w_return
        
        
        if len(self.X) >= self.gp_opt_schedule and len(self.X) % self.gp_opt_schedule == 0:
            if self.gp_mcmc:
                self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                print("[Running MCMC for GP hyper-parameters]")
                hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin

                gp_samples_mean = np.mean(gp_samples, axis=0)
                print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                self.gp.kern.variance.fix(gp_samples_mean[0])
                self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                self.gp.likelihood.variance.fix(gp_samples_mean[2])

                self.gp_params = self.gp.parameters
            else:
                self.gp.optimize_restarts(num_restarts = 10, messages=False)
                self.gp_params = self.gp.parameters

                gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                print("---Optimized hyper: ", self.gp)

        
        if self.pt is not None:
            print("[pt: {0}]".format(self.pt[len(self.X)-init_points]))

            

        if np.random.random() < self.pt[len(self.X)-init_points]:
            M_target = self.M_target

            ls_target = self.gp["rbf.lengthscale"][0]
            v_kernel = self.gp["rbf.variance"][0]
            obs_noise = self.gp["Gaussian_noise.variance"][0]
            obs_noise = np.max([1e-5, obs_noise])

            try:
                s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim), M_target)
            except np.linalg.LinAlgError:
                s = np.random.rand(M_target, self.dim) - 0.5

            b = np.random.uniform(0, 2 * np.pi, M_target)

            random_features_target = {"M":M_target, "length_scale":ls_target, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

            Phi = np.zeros((self.X.shape[0], M_target))
            for i, x in enumerate(self.X):
                x = np.squeeze(x).reshape(1, -1)
                features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                features = features / np.sqrt(np.inner(features, features))
                features = np.sqrt(v_kernel) * features

                Phi[i, :] = features

            Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
            Sigma_t_inv = np.linalg.inv(Sigma_t)
            nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

            try:
                w_sample_1 = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
            except np.linalg.LinAlgError:
                w_sample_1 = np.random.rand(1, self.M) - 0.5

            x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, \
                        w_sample=w_sample_1, bounds=self.bounds, partitions=None)
        else:
            w_samples = all_w_t
            x_max = acq_max(ac=self.util_dp_fts_de.utility, M=self.M, random_features=self.random_features, \
                        w_sample=w_samples, bounds=self.bounds, partitions=self.partitions)

        # check if x is a repeated query
        if not self.X.shape[0] == 0:
            if np.any(np.all(self.X - x_max == 0, axis=1)):
                x_max = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=self.bounds.shape[0])

        y = self.f(x_max, agent_ind)

        self.Y = np.append(self.Y, y)
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        self.gp.set_XY(X=self.X, Y=self.Y.reshape(-1, 1))

        print("Agent {0}, iter {1} ------ x_t: {2}, y_t: {3}".format(agent_ind, len(self.X)-init_points, x_max, y))

        x_max_param = self.X[self.Y.argmax(), :-1]
        self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
        self.res['all']['values'].append(self.Y[-1])
        self.res['all']['params'].append(self.X[-1])

        if self.log_file is not None:
            pickle.dump(self.res, open(self.log_file, "wb"))

        w_return = self.sample_w()
        return w_return


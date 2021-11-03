# -*- coding: utf-8 -*-

import numpy as np
import GPy
from helper_funcs_ts import UtilityFunction, acq_max
import pickle
import itertools
import time

class TS(object):
    def __init__(self, f, pbounds, gp_opt_schedule, ARD=False, \
                 gp_mcmc=False, log_file=None, M_target=100, verbose=1):
        """
        """

        self.M_target = M_target
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
        
        self.i = 0

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
        
        
    def init(self, init_points):
        l = [np.random.uniform(x[0], x[1], size=init_points)
             for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            y = self.f(x)
            y_init.append(y)
            self.res['all']['init_values'].append(y)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        self.initialized = True

        init = {"X":self.X, "Y":self.Y}
        self.res['all']['init'] = init

    def maximize(self, n_iter=25, init_points=5):
        self.util_ts = UtilityFunction()

        if not self.initialized:
            self.init(init_points)

        self.gp = GPy.models.GPRegression(self.X, self.Y.reshape(-1, 1), \
                GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=1.0, variance=0.1, ARD=self.ARD))
        self.gp["Gaussian_noise.variance"][0] = 1e-4

        
        # optimize GP hypers
        if init_points > 1:
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



        # optimize acquisition function
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
            w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
        except np.linalg.LinAlgError:
            w_sample = np.random.rand(1, self.M) - 0.5
            
        x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, w_sample=w_sample, bounds=self.bounds)


        # main loop
        for i in range(n_iter):
            if not self.X.shape[0] == 0:
                if np.any(np.all(self.X - x_max == 0, axis=1)):
                    x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])

            y = self.f(x_max)

            self.Y = np.append(self.Y, y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))

            self.gp.set_XY(X=self.X, Y=self.Y.reshape(-1, 1))

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
                else:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params = self.gp.parameters
                    gp_samples = None # set this flag variable to None, to indicate that MCMC is not used
                    print("---Optimized hyper: ", self.gp)

            # optimize acquisition function
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
                w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)
            except np.linalg.LinAlgError:
                w_sample = np.random.rand(1, self.M) - 0.5
            x_max = acq_max(ac=self.util_ts.utility, M=M_target, random_features=random_features_target, \
                            w_sample=w_sample, bounds=self.bounds)

            print("iter {0} ------ x_t: {1}, y_t: {2}".format(self.i+1, x_max, y))

            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))


#from __future__ import print_function
#from __future__ import division
import numpy as np
from datetime import datetime
from scipy.optimize import minimize

def acq_max(ac, M, random_features, w_sample, bounds, partitions=None):
    para_dict={"M":M, "random_features":random_features, "w_sample":w_sample, "partitions":partitions}

    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(1000, bounds.shape[0]))

    ys = []
    for x in x_tries:
        ys.append(ac(x.reshape(1, -1), para_dict))
    ys = np.array(ys)

    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(20, bounds.shape[0]))
    for x_try in x_seeds:
        res = minimize(lambda x: -ac(x.reshape(1, -1), para_dict),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun
    
    return x_max

class UtilityFunction(object):
    def __init__(self, kind):
        self.kind = kind

    def utility(self, x, para_dict):
        M, random_features, w_sample, partitions = \
                para_dict["M"], para_dict["random_features"], para_dict["w_sample"], para_dict["partitions"]

        if self.kind == 'ts':
            return self._ts(x, M, random_features, w_sample)
        elif self.kind == 'dp_fts_de':
            return self._dp_fts_de(x, M, random_features, w_sample, partitions)

    @staticmethod
    def _dp_fts_de(x, M, random_features, w_sample, partitions):
        '''
        w_sample is now a list of sampled w's, each corresponding to a partition
        '''
        d = x.shape[1]

        x = np.clip(x, 0, 1)


        # below finds the index of the sub-region x belongs to
        partitions_np = np.array(partitions)
        N_partitions = len(partitions)

        tmp = np.tile(x, N_partitions).reshape(N_partitions, d, 1)
        tmp_2 = np.concatenate((tmp, tmp), axis=2)

        tmp_3 = tmp_2 - partitions_np
        flag_left = tmp_3[:, :, 0]
        flag_right = tmp_3[:, :, 1]
        flag_left_count = np.nonzero(flag_left >= 0)[0]
        flag_right_count = np.nonzero(flag_right <= 0)[0]

        (indices_left, counts_left) = np.unique(flag_left_count, return_counts=True)
        indices_left_correct = indices_left[counts_left == d]
        (indices_right, counts_right) = np.unique(flag_right_count, return_counts=True)
        indices_right_correct = indices_right[counts_right == d]
        
        part_ind = np.intersect1d(indices_left_correct, indices_right_correct)[0]
        
        
        w_sample_p = w_sample[part_ind]

        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features

        f_value = np.squeeze(np.dot(w_sample_p, features))

        return f_value

    @staticmethod
    def _ts(x, M, random_features, w_sample):
        d = x.shape[1]
        
        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments

        f_value = np.squeeze(np.dot(w_sample, features))

        return f_value

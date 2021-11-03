import GPy
from bayesian_optimization_ts import TS
import pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score

from multiprocessing.dummy import Pool as ThreadPool

max_iter = 60

N = 29

landmine_data = pickle.load(open("aux_files/landmine_formated_data.pkl", "rb"))
all_X_train, all_Y_train, all_X_test, all_Y_test = landmine_data["all_X_train"], landmine_data["all_Y_train"], \
        landmine_data["all_X_test"], landmine_data["all_Y_test"]

mine_list = np.arange(0, 29)
run_list = np.arange(0, 100)

pool = ThreadPool(10)  # Number of threads

for l in mine_list:
    X_train = all_X_train[l]
    Y_train = np.squeeze(all_Y_train[l])
    X_test = all_X_test[l]
    Y_test = np.squeeze(all_Y_test[l])

    def obj_func_landmine(param):
        parameter_range = [[1e-4, 10.0], [1e-2, 10.0]]
        C_ = param[0]
        C = C_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
        gam_ = param[1]
        gam = gam_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]

        clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
        clf.fit(X_train, Y_train)
        pred = clf.predict_proba(X_test)
        score = roc_auc_score(Y_test, pred[:, 1])

        return score

    def parallel_runs(itr):
        log_file_name = "results_ts/field_" + str(l) + "_iter_" + str(itr) + ".p"

        bo_ts = TS(f=obj_func_landmine, pbounds={'x1':(0, 1), 'x2':(0, 1)}, gp_opt_schedule=5, log_file=log_file_name, M_target=100)
        bo_ts.maximize(n_iter=max_iter, init_points=10)

    pool.map(parallel_runs, run_list)

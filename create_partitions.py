'''
This script creates the partitions information to be used for distributed exploration (DE).
To run FTS (without DE), simply set N_partitions_single_dim=1, a=1, T_const=100, T_max_decay=100.
'''
import numpy as np
import pickle

N = 29

Domain = np.array([0, 1])
Domain_width = Domain[1] - Domain[0]

N_partitions_single_dim = 2 # we partition every dimension into "N_partitions_single_dim" regions

d = 2 # input dimension
partitions = []
for n in range(N_partitions_single_dim):
    for m in range(N_partitions_single_dim):
        partitions.append(np.array([[n * Domain_width / N_partitions_single_dim, (n+1) * Domain_width / N_partitions_single_dim], 
                            [m * Domain_width / N_partitions_single_dim, (m+1) * Domain_width / N_partitions_single_dim]]))

N_partitions = len(partitions)
partition_assignment = np.arange(N) % N_partitions

print("N_partitions: ", N_partitions)
print("partitions: ", partitions)
print("partition_assignment: ", partition_assignment)



T_max = 200
a, b = 16, 1

# these two variables are used to set the adaptive weights: a remains constant for "T_const" iterations, and then a decays linearly to b in the next "T_max_decay" iterations
T_const = 10
T_max_decay = 30

a_s = np.linspace(a, b, T_max_decay)
a_s = np.append(a_s, np.ones(T_max - T_max_decay))

all_weights_all = []
for t in range(T_const):
    all_weights = []
    for n in range(N_partitions):
        weights = np.zeros(N)
        for i in range(N):
            if partition_assignment[i] == n:
                weights[i]= np.exp(a)
            else:
                weights[i]= np.exp(b)
        weights = weights / np.sum(weights)
        all_weights.append(weights)
    all_weights_all.append(all_weights)

for t in range(100):
    all_weights = []
    for n in range(N_partitions):
        weights = np.zeros(N)
        for i in range(N):
            if partition_assignment[i] == n:
                weights[i]= np.exp(a_s[t])
            else:
                weights[i]= np.exp(b)
        weights = weights / np.sum(weights)
        all_weights.append(weights)
    all_weights_all.append(all_weights)


partition_info = {"partitions":partitions, "partition_assignment":partition_assignment, "all_weights":all_weights_all}
pickle.dump(partition_info, open("aux_files/partitions_info_N_part_" + str(N_partitions) + "_a_" + str(a) + "_b_" + str(b) + \
                                 "_T_decay_" + str(T_max_decay) + "_T_const_" + str(T_const) +  "_adaptive.pkl", "wb"))


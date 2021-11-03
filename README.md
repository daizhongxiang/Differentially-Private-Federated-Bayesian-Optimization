# Implementations for the paper: "Differentially Private Federated Bayesian Optimization with Distributed Exploration"
This directory contains the code for the landmine detection experiment in the paper "Differentially Private Federated Bayesian Optimization with Distributed Exploration", which is submitted to NeurIPS 2021.

The implementations here include standard Thompson sampling (TS) and DP-FTS-DE. The implementation of DP-FTS-DE also subsumes its different variants including FTS (without DP and DE), FTS-DE (without DP) and DP-FTS (without DE).

## Requirement:
- sklearn, numpy, GPy (https://github.com/SheffieldML/GPy)

## Preprocessing:
- preprocess.py: preprocesses the Landmine Detection dataset
- create_random_features.py: generates the random features for DP-FTS-DE, which is shared among all agents as a common ground for collaboration
- create_partitions.py: creates the information to be used by distributed exploration (DE)

## Instructions to run:
- landmine_ts: runs the TS algorithm using the landmine detection experiment
- landmine_dp_fts_de: runs the DP-FTS-DE algorithm using the landmine detection experiment

## Analysis of results:
- analyze.ipynb


## Decription of scripts:
- bayesian_optimization_ts.py, helper_funcs_ts.py: standard BO (Thompson sampling or TS) and its helper functions
- bayesian_optimization_dp_fts_de.py, helper_funcs_dp_fts_de.py: DP-FTS-DE and its helper functions

## Decription of directories:
- results_ts: saves the results for TS (only results for 10 of the 100 random runs are included in this directory due to size constraint for uploading to the openreview system)
- results_dp_fts_de: saves the results for DP-FTS-DE (only results for 10 of the 100 random runs are included in this directory due to size constraint for uploading to the openreview system)
- aux_files: saves auxiliary files used by the algorithm, including the shared random features, the partition information, and the landmine detection data reformated for more convenient use


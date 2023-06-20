import gym
import numpy as np
import math
import os
import submitit
import datetime

from training import train
#rom procgen import ProcgenEnv
#import update
#import entropy_update
#from update import *
#from entropy_update import *
import sys
import os
#sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("results", "pong")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "outputs")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(results_path, run_timestamp)
os.mkdir(run_path)

executor = submitit.AutoExecutor(folder="results/pong/outputs")

executor.update_parameters(timeout_min = 1000, mem_gb = 2, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 5, slurm_partition = "gpu")

jobs = []

with executor.batch():
	job = executor.submit(train, episode=2000, reward = 2, tmax = 100, experiment_path=run_path, folder_name = 'pong_clean')
	jobs.append(job)

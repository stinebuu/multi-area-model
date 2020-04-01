import numpy as np
import os
import sys
import shutil
import json
import nest

from run_benchmark_createParams import create_params
from multiarea_model import MultiAreaModel, MultiAreaModel_3
from config import base_path, data_path

"""
Create parameters.

Run as

mpirun/srun python run_benchmark.py N_scaling num_processes t_sim K_scaling nest_version

nest_version can be 2 or 3
"""

N_scaling = float(sys.argv[1])
num_processes = int(sys.argv[2])
t_sim = float(sys.argv[3])
K_scaling = float(sys.argv[4])
NEST_version = int(sys.argv[5])

print("create parameters")

net_label, sim_label = create_params(N_scaling, num_processes, t_sim, K_scaling, NEST_version)

label = sim_label
network_label = net_label

print("load simulation parameters\n")

# Load simulation parameters
fn = os.path.join(data_path, label, '_'.join(('custom_params', label)))
with open(fn, 'r') as f:
    custom_params = json.load(f)

print("load parameters\n")

# Copy custom param file for each MPI process
for i in range(custom_params['sim_params']['num_processes']):
    shutil.copy(fn, '_'.join((fn, str(i))))

fn = os.path.join(data_path,
                  label,
                  '_'.join(('custom_params',
                            label,
                            str(nest.Rank()))))
with open(fn, 'r') as f:
    custom_params = json.load(f)

os.remove(fn)

if NEST_version == 2:
    M = MultiAreaModel(network_label,
                       simulation=True,
                       sim_spec=custom_params['sim_params'])
elif NEST_version == 3:
    M = MultiAreaModel_3(network_label,
                        simulation=True,
                        sim_spec=custom_params['sim_params'])
print("simulate\n")
M.simulation.simulate()

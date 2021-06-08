import numpy as np
import os
import sys
import shutil
import json
import nest

from multiarea_model import MultiAreaModel, MultiAreaModel_3
from multiarea_model.multiarea_helpers import write_out_timer_data
from config import base_path, data_path

"""
Create parameters.

Run as

mpirun/srun python run_benchmark.py N_scaling num_processes t_sim K_scaling nest_version

nest_version can be 2, 3 or rng
"""

N_scaling = float(sys.argv[1])
num_processes = int(sys.argv[2])
t_sim = float(sys.argv[3])
K_scaling = float(sys.argv[4])
NEST_version = sys.argv[5]


print("load simulation and network labels\n")

if N_scaling < 1:
    N_scaling = N_scaling*1000
N_scaling = int(N_scaling)

# Load simulation and network labels
labels_fn = os.path.join(base_path, 'label_files/labels_{}_{}_{}_{}.json'.format(N_scaling, num_processes, int(t_sim), NEST_version))

print(labels_fn)
with open(labels_fn, 'r') as f:
    labels = json.load(f)

label = labels['simulation_label']
network_label = labels['network_label']

print("load simulation parameters\n")

# Load simulation parameters
fn = os.path.join(data_path, label, '_'.join(('custom_params', label)))
with open(fn, 'r') as f:
    custom_params = json.load(f)

print("Create network and simulate\n")

if NEST_version == '2':
    M = MultiAreaModel(network_label,
                       simulation=True,
                       sim_spec=custom_params['sim_params'])
elif NEST_version == '3':
    M = MultiAreaModel_3(network_label,
                        simulation=True,
                        sim_spec=custom_params['sim_params'])
elif NEST_version == 'rng':
    M = MultiAreaModel_rng(network_label,
                           simulation=True,
                           sim_spec=custom_params['sim_params'])
print("simulate\n")
M.simulation.simulate()

data_dir = M.simulation.data_dir
label = M.simulation.label

# Write out timer data
write_out_timer_data(data_dir, label)

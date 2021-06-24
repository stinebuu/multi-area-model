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

data_path = sys.argv[1]
data_folder_hash = sys.argv[2]
NEST_version = sys.argv[3]
mam_state = sys.argv[4]  # Fig3: corresponds to figure 3 in schmidt et al. 2018: Groundstate
                         # Fig5: corresponds to figure 5 in schmidt et al. 2018: Metastable

print("load simulation parameters\n")

# Load simulation parameters
fn = os.path.join(data_path,
                  data_folder_hash,
                  '_'.join(('custom_params', str(nest.Rank()))))
with open(fn, 'r') as f:
    custom_params = json.load(f)

print("Create network and simulate\n")

if NEST_version == '2':
    M = MultiAreaModel('benchmark',
                       simulation=True,
                       sim_spec=custom_params['sim_params'],
                       data_path=data_path,
                       data_folder_hash=data_folder_hash)
elif NEST_version == '3':
    M = MultiAreaModel_3('benchmark',
                         simulation=True,
                         sim_spec=custom_params['sim_params'],
                         data_path=data_path,
                         data_folder_hash=data_folder_hash)
elif NEST_version == 'rng':
    M = MultiAreaModel_rng('benchmark',
                           simulation=True,
                           sim_spec=custom_params['sim_params'],
                           data_path=data_path,
                           data_folder_hash=data_folder_hash)
print("simulate\n")
M.simulation.simulate()

data_dir = M.simulation.data_dir
label = M.simulation.label

# Write out timer data
write_out_timer_data(data_dir, label)

import numpy as np
import os
import sys
import json

from multiarea_model import MultiAreaModel, MultiAreaModel_3
from config import base_path
from figures.Schmidt2018_dyn.network_simulations import NEW_SIM_PARAMS

N_scaling = float(sys.argv[1])
num_processes = int(sys.argv[2])
t_sim = float(sys.argv[3])
K_scaling = float(sys.argv[4])
NEST_version = sys.argv[5]
mam_state = sys.argv[6]  # Fig3: corresponds to figure 3 in schmidt et al. 2018: Groundstate
                         # Fig5: corresponds to figure 5 in schmidt et al. 2018: Metastable

network_params, _ = NEW_SIM_PARAMS[mam_state][0]

network_params['connection_params']['K_stable'] = os.path.join(base_path, 'K_stable.npy')

sim_params = {'t_sim': t_sim,
              'num_processes': num_processes,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False},
              'master_seed': 1}

theory_params = {'dt': 0.1}

if NEST_version == '2':
    print("NEST version 2.x\n")
    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params)
elif NEST_version == '3':
    print("NEST version 3.0\n")
    M = MultiAreaModel_3(network_params, simulation=True,
                         sim_spec=sim_params,
                         theory=True,
                         theory_spec=theory_params)
elif NEST_version == 'rng':
    print("NEST version rng\n")
    M = MultiAreaModel_rng(network_params, simulation=True,
                           sim_spec=sim_params,
                           theory=True,
                           theory_spec=theory_params)

print(M.label)
print(M.simulation.label)

p, r = M.theory.integrate_siegert()

print("dump parameters\n")

if N_scaling < 1:
    N_scaling = N_scaling*1000
N_scaling = int(N_scaling)

labels_fn = os.path.join(base_path, f'label_files/labels_{N_scaling}_{num_processes}_{int(t_sim)}}_{NEST_version}_{mam_state}.json')
labels_dict = {'network_label': M.label,
               'simulation_label': M.simulation.label}
print(labels_fn)

with open(labels_fn, 'w') as f:
    json.dump(labels_dict, f)

import numpy as np
import os
import sys
import shutil
import json
import nest

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

d = {}
conn_params = {#'replace_non_simulated_areas': 'het_poisson_stat',
               'g': -11.,
               'K_stable': os.path.join(base_path, 'K_stable.npy'),
               'fac_nu_ext_TH': 1.2,
               'fac_nu_ext_5E': 1.125,
               'fac_nu_ext_6E': 1.41666667,
               'av_indegree_V1': 3950.}
input_params = {'rate_ext': 10.}
neuron_params = {'V0_mean': -150.,
                 'V0_sd': 50.}
network_params = {'N_scaling': N_scaling,
                  'K_scaling': K_scaling,
                  'fullscale_rates': os.path.join(base_path, 'tests/fullscale_rates.json'),
                  'input_params': input_params,
                  'connection_params': conn_params,
                  'neuron_params': neuron_params}

sim_params = {'t_sim': t_sim,
              'num_processes': num_processes,
              'local_num_threads': 1,
              'recording_dict': {'record_vm': False}}

theory_params = {'dt': 0.1}

if NEST_version == 2:
    print("NEST version 2.x\n")
    M = MultiAreaModel(network_params, simulation=True,
                       sim_spec=sim_params,
                       theory=True,
                       theory_spec=theory_params)
elif NEST_version == 3:
    print("NEST version 3.0\n")
    M = MultiAreaModel_3(network_params, simulation=True,
                         sim_spec=sim_params,
                         theory=True,
                         theory_spec=theory_params)

print(M.label)
print(M.simulation.label)

p, r = M.theory.integrate_siegert()

#M.simulation.simulate()

print("dump parameters\n")

labels_fn = os.path.join(base_path, 'label_files/labels_{}_{}.json'.format(N_scaling, num_processes))
labels_dict = {'network_label': M.label,
               'simulation_label': M.simulation.label}
print(labels_fn)

with open(labels_fn, 'w') as f:
    json.dump(labels_dict, f)


print("load simulation and network labels\n")

# Load simulation and network labels
labels_fn = os.path.join(base_path, 'label_files/labels_{}_{}.json'.format(N_scaling, num_processes))
with open(labels_fn, 'r') as f:
    labels = json.load(f)

label = labels['simulation_label']
network_label = labels['network_label']

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

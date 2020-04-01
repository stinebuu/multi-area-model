import numpy as np
import os
import json

from multiarea_model import MultiAreaModel, MultiAreaModel_3
from config import base_path

def create_params(N_scaling, num_processes, t_sim, K_scaling, NEST_version):

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

    print("return labels")

    return M.label, M.simulation.label

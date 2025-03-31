import measles_efficiency

from mpi4py import MPI
import numpy as np
import pandas as pd
import math
import glob


# Thanks Remy! This is a good trick...
def get_tx_vax_levels(TX_FILENAME):

    df_tx = pd.read_csv(TX_FILENAME)
    vax_levels_list = df_tx['MMR Vaccination Rate'].values.tolist()

    vax_levels_list = [
        np.round(x, 1)
        for x in vax_levels_list
    ]
    vax_levels_list = list(set(vax_levels_list))
    return vax_levels_list


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total_num_processors = comm.Get_size()

NUM_REPS = 2500

vax_rates_grid = np.hstack((np.arange(80, 100, 0.2), np.arange(60, 80, 0.5), np.arange(0, 60, 1)))
school_sizes_grid = [10, 25, 100, 250, 500, 750, 1000, 1500, 2500, 5000]

base_seed = 216014948987466572985971385191991148824
base_seed_sequence = np.random.SeedSequence(base_seed).spawn(total_num_processors)

tx_vax_levels = get_tx_vax_levels("TX_MMR_vax_rate.csv")
tx_vax_to_add = [x for x in tx_vax_levels if not any(math.isclose(x, y, rel_tol=1e-9) for y in vax_rates_grid)]

experiment_csvs = glob.glob("*worker*.csv")

for vax_prop in np.append(np.asarray(tx_vax_to_add), vax_rates_grid):
    for population in school_sizes_grid:

        if f"2500reps_worker{rank}_vax{np.round(vax_prop, 1):.1f}_pop{population}.csv" in experiment_csvs:
            continue

        measles_efficiency.DEFAULT_MSP_EFFICIENCY_PARAMS["vax_prop"] = vax_prop * 0.01
        measles_efficiency.DEFAULT_MSP_EFFICIENCY_PARAMS["population"] = population

        measles_efficiency.DEFAULT_MSP_EFFICIENCY_PARAMS["simulation_seed"] = base_seed_sequence[rank]

        np.savetxt(f"2500reps_worker{rank}_vax{np.round(vax_prop, 1):.1f}_pop{population}.csv",
                   measles_efficiency.compute_new_infections(measles_efficiency.DEFAULT_MSP_EFFICIENCY_PARAMS, NUM_REPS, True),
                   delimiter=",")




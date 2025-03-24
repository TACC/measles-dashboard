# Quick and dirty script on top of old measles_single_population.py file
#   (before Remy vaccination updates) to answer CDC question of whether
#   changing "outbreak conditional on N new infections" should be changed
#   from 20 to 10
# No breakthrough infections
import copy

import numpy as np
import pandas as pd

import measles_single_population as msp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Texas MMR vaccination rate
# Mean: 94
# Median: 97
# 75th quantile: 100 (not relevant)
# 25th quantile: 92.6
# --> Going to test median and 25th quantile (97 and 92.6) and also default 85%

# Testing school sizes: 500 (default), 1000
# --> with the above 3 vaccination rates, this creates 6 combos/distributions

# Using default dashboard parameters otherwise

params_dict = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
params_dict["population"] = [500]
params_dict["simulation_seed"] = 147125098488

# prob_10_plus_list = []
# prob_20_plus_list = []
#
# for vax_prop in np.append(np.arange(40, 95, 2)/100, np.arange(95,101)/100):
#     params_dict["vax_prop"] = [vax_prop]
#
#     n_sim = 10000
#     stochastic_sim = msp.StochasticSimulations(params_dict, n_sim)
#     stochastic_sim.run_stochastic_model(track_infected=True)
#     df = stochastic_sim.df_new_infected_empirical_dist
#
#     prob_10_plus_new_cases = df.loc[df['new_infected'] >= 10, 'probability'].sum()
#     prob_20_plus_new_cases = df.loc[df['new_infected'] >= 20, 'probability'].sum()
#
#     prob_10_plus_list.append(prob_10_plus_new_cases)
#     prob_20_plus_list.append(prob_20_plus_new_cases)
#
#     print("1 done")
#
# np.savetxt("prob_10_plus_new_10k_every2_v2.csv", prob_10_plus_list, delimiter=",")
# np.savetxt("prob_20_plus_new_10k_every2_v2.csv", prob_20_plus_list, delimiter=",")

breakpoint()

def percent_format(x, pos):
    return f"{int(x * 100)}%"  # Converts to integer + percentage sign

prob_list = np.genfromtxt("prob_10_plus_new_10k_every2_v2.csv")

x_positions = [0.4, 0.6, 0.8, 1.0]
y_positions = [0.0, 0.25, 0.5, 0.75]

plt.figure(figsize=(8, 5))
plt.plot(np.append(np.arange(40, 95, 2)/100, np.arange(95, 101)/100), prob_list)
plt.xticks(x_positions)
plt.yticks(y_positions)
plt.gca().xaxis.set_major_formatter(FuncFormatter(percent_format))
plt.gca().yaxis.set_major_formatter(FuncFormatter(percent_format))
plt.xlabel("MMR Vaccination Rate")
plt.ylabel("Probability of at Least 10 New Infections")

plt.savefig("new_CDC_vaccine_graph_10_plus_new_infections.png", dpi=1200)

breakpoint()
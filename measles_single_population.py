#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rfp437

>(^_^)> ~~~~
"""

# %% Imports and parameters
###########################
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy

from typing import TypedDict, Type
from numpy.lib.stride_tricks import sliding_window_view
from abc import ABC, abstractmethod

# %% Print options
##################
pd.set_option('display.precision', 1)
pd.set_option('display.width', 185)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.4}'.format

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=185)


# %% Measles Parameters
#######################
class MeaslesParameters(TypedDict):
    R0: float
    incubation_period: float
    infectious_period: float
    school_contacts: float
    other_contacts: float
    population: list[int]
    I0: list[int]
    vax_prop: list[float]
    threshold_values: list[int]
    sim_duration_days: int
    time_step_days: float
    is_stochastic: bool
    simulation_seed: int


# Set parameters
# Natural history parameters
# https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html
# Incubation: 11.5 days, so first symptoms 11.5 days since t0
# Rash starts: 3 days after first symptoms, so 14.5 since t0
# Infectious: 4 days before rash (so 10.5 days since t0), 4 days after (18.5 days since t0)
DEFAULT_MSP_PARAMS: MeaslesParameters = \
    {
        'R0': 15.0,  # transmission rate
        'incubation_period': 10.5,
        'infectious_period': 5,
        'school_contacts': 5.63424,
        'other_contacts': 2.2823,
        'population': [500],
        'I0': [1],
        'vax_prop': [0.85],  # number between 0 and 1
        'threshold_values': [10],  # outbreak threshold, only first value used for now
        'sim_duration_days': 250,
        'time_step_days': 0.25,
        'is_stochastic': True,  # False for deterministic,
        "simulation_seed": 147125098488
    }


# %% Functions
##############

def transform_matrix_to_long_df(
        np_array: np.ndarray,
        colnames: list[str] = None,
        id_col: str = 'time_idx',
        id_values: list[str] = None,
        var_name: str = 'idx',
        value_name: str = 'value') -> pd.DataFrame:
    """
    Takes a 2D numpy array where each column represents a given simulation
    and each row an observation (ie, a date), and transforms the data in
    a pandas dataframe in a long format.

    Long format: each row represents a single observation,
    and each column represents a single variable. Easier to plot.
    """

    if colnames is None:
        colnames = list(range(np_array.shape[1]))

    if id_values is None:
        id_values = list(range(np_array.shape[0]))

    df_wide = pd.DataFrame(np_array, columns=colnames)
    df_wide[id_col] = id_values
    df_long = pd.melt(
        df_wide,
        id_vars=id_col,
        var_name=var_name,
        value_name=value_name,
        value_vars=colnames
    )

    df_long.apply(pd.to_numeric, errors="coerce")

    return df_long


def calculate_np_moving_average(
        results_array: np.ndarray,
        window: int,
        shorter_window_beginning: bool = True) -> np.ndarray:
    """
    This takes a numpy 2-dimensional array and computes moving averages along
    the columns: each row represents a different time series.

    Row index corresponds to simulation replication
    Column index corresponds to simulation day

    Parameters:
        results_array (np.ndarray):
            2D array over which to take moving average.
        window (int):
            Length of moving average window.
        shorter_window_beginning (Optional[bool]):
            If True, we calculate the moving average for the first few observations
            (length window - 1) using the shorter window of available data. If
            False, values are NAs.

    Returns:
        np.ndarray:
            numpy 2D array of same size as input.
    """

    # Metapopulation formatting issues...
    results_array = np.atleast_2d(results_array)

    array_window = sliding_window_view(results_array, window_shape=(window,), axis=1)
    array_ma = array_window.mean(axis=-1)

    if shorter_window_beginning:
        pre_pad = np.cumsum(results_array[:, :window - 1], axis=1) / np.arange(1, window)
    else:
        pre_pad = np.full((results_array.shape[0], window - 1), np.nan)

    return np.column_stack((pre_pad, array_ma))


# %% Model
##########
class MetapopSEIR:

    def __init__(self,
                 params: dict):
        """
        Initialize metapopulation SEPIR model

        Parameters:
        params: dictionary with disease parameters and movement rates
        N: array of population sizes
        I0: array of initial infectious cases
        T: total simulation time
        dt: time step size
        """

        self.n_pop = len(params['population'])  # number of populations
        self.N = np.array(params['population'])  # population size
        self.steps_per_day = int(1.0 / params['time_step_days'])
        self.dt = params['time_step_days']
        self.n_steps = 1 + int(
            params['sim_duration_days'] / self.dt)  # number of time steps

        # Stochastic simulations
        self.is_stochastic = params['is_stochastic']
        if self.is_stochastic:
            self.RNG = None

        # Number of contacts
        self.school_contacts = params['school_contacts']
        self.other_contacts_base = params['other_contacts']
        self.total_contacts = self.school_contacts + self.other_contacts_base

        # Disease parameters
        beta = params['R0'] / (params["infectious_period"] * self.total_contacts)

        self.beta = beta  # transmission rate
        self.sigma = 1.0 / params["incubation_period"]  # rate of progression from E to I
        self.gamma = 1.0 / params["infectious_period"]  # recovery rate

        # Time array
        self.t = np.linspace(0, params['sim_duration_days'], self.n_steps)

        self.params = params

        # Initialize compartments as 2D arrays [time_step, population]
        self.S = np.zeros((self.n_steps, self.n_pop))
        self.E = np.zeros((self.n_steps, self.n_pop))
        self.I = np.zeros((self.n_steps, self.n_pop))
        self.R = np.zeros((self.n_steps, self.n_pop))

        # Add storage of transition variables
        # Used to compute incidence
        # Looks like we are currently not using P so skipping that
        self.S_to_E = np.zeros((self.n_steps, self.n_pop))
        self.E_to_I = np.zeros((self.n_steps, self.n_pop))
        self.I_to_R = np.zeros((self.n_steps, self.n_pop))

        # Set initial conditions
        # 03052025 1:1 with Lauren -- Lauren does not want
        #   us to change the user input in the backend -- she wants
        #   to have guard rails on the user inputs

        params = self.params
        N = self.N

        self.I[0] = np.array(params['I0'])
        self.R[0] = np.minimum((N * np.array(params['vax_prop'])).astype(int), N - self.I[0])
        self.S[0] = N - self.I[0] - self.R[0]

        # Current step tracker
        self.current_step = 0

    def get_compartment_transition(self,
                                   rate: float,
                                   compartment_count: int) -> float:
        """
        For a given compartment and a given rate, this calculates the number
        of individuals transitioning out of the compartment.
        The calculations is either deterministic or stochastic.

        Parameters
        ----------
        rate : double
            Force of infection for new infected, rate out of compartment
            otherwise.
        compartment_count : int
            Number of individuals in compartment.

        Returns
        -------
        Number of individuals leaving compartment.

        """
        total_rate = rate * compartment_count

        if self.is_stochastic:
            delta = self.RNG.poisson(total_rate)
        else:
            delta = total_rate

        delta = min(delta, compartment_count)

        return delta

    def calculate_compartment_updates(self,
                                      ix_pop: int) -> dict:
        """
        Calculates changes in compartments for population ix_pop.

        Parameters
        ----------
        ix_pop : int
            Index of population.

        Returns
        -------
        dict:
            Keys are: "dS, dE, dI, dR, dS_out, dE_out, dI_out",
            Values are the quantities at time `self.current_step`.

        """
        t = self.current_step
        dt = self.dt

        force_of_infection = self.beta * self.total_contacts * dt * \
                             self.I[t, ix_pop] / self.N[ix_pop]

        dS_out = self.get_compartment_transition(force_of_infection, self.S[t, ix_pop])
        dE_out = self.get_compartment_transition(self.sigma * dt, self.E[t, ix_pop])
        dI_out = self.get_compartment_transition(self.gamma * dt, self.I[t, ix_pop])

        dS = -dS_out
        dE = dS_out - dE_out
        dI = dE_out - dI_out
        dR = dI_out

        updates_dict = {"dS": dS, "dE": dE, "dI": dI, "dR": dR,
                        "dS_out": dS_out, "dE_out": dE_out, "dI_out": dI_out}

        return updates_dict

    def step(self):
        """Calculate one time step using Euler's method"""
        if self.current_step >= self.n_steps - 1:
            return False

        # Loop through populations
        for ix_pop in range(self.n_pop):
            updates = self.calculate_compartment_updates(ix_pop)
            current_step = self.current_step

            # Creates view of same array in memory, NOT a copy
            S, E, I, R = self.S, self.E, self.I, self.R
            S_to_E, E_to_I, I_to_R = self.S_to_E, self.E_to_I, self.I_to_R

            # Update compartments using Euler's method
            S[current_step + 1, ix_pop] = S[current_step, ix_pop] + updates["dS"]
            E[current_step + 1, ix_pop] = E[current_step, ix_pop] + updates["dE"]
            I[current_step + 1, ix_pop] = I[current_step, ix_pop] + updates["dI"]
            R[current_step + 1, ix_pop] = R[current_step, ix_pop] + updates["dR"]

            # Also update transition variables history
            S_to_E[current_step + 1, ix_pop] = updates["dS_out"]
            E_to_I[current_step + 1, ix_pop] = updates["dE_out"]
            I_to_R[current_step + 1, ix_pop] = updates["dI_out"]

        self.current_step += 1
        return True

    def simulate(self):
        """Run simulation for all time steps"""
        while self.step():
            pass

    def clear(self):

        self.S = np.zeros((self.n_steps, self.n_pop))
        self.E = np.zeros((self.n_steps, self.n_pop))
        self.I = np.zeros((self.n_steps, self.n_pop))
        self.R = np.zeros((self.n_steps, self.n_pop))

        self.S_to_E = np.zeros((self.n_steps, self.n_pop))
        self.E_to_I = np.zeros((self.n_steps, self.n_pop))
        self.I_to_R = np.zeros((self.n_steps, self.n_pop))

        params = self.params
        N = self.N

        self.I[0] = np.array(params['I0'])
        self.R[0] = np.minimum((N * np.array(params['vax_prop'])).astype(int), N - self.I[0])
        self.S[0] = N - self.I[0] - self.R[0]

        self.current_step = 0

    def plot_results(self):
        """Plot the results for all populations"""
        fig, axes = plt.subplots(self.n_pop, 1, figsize=(10, 6 * self.n_pop))
        if self.n_pop == 1:
            axes = [axes]

        for pop in range(self.n_pop):
            ax = axes[pop]
            ax.plot(self.t, self.S[:, pop], label='Susceptible')
            ax.plot(self.t, self.E[:, pop], label='Exposed')
            ax.plot(self.t, self.I[:, pop], label='Infectious')
            ax.plot(self.t, self.R[:, pop], label='Recovered')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Number of individuals')
            ax.set_title(f'Population {pop + 1} SEPIR Dynamics')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()


# %% Statistic Collectors
#########################

class AcrossRepStat:

    def __init__(self):
        self.data = np.array([])

    def clear(self):
        self.data[:] = 0


class AcrossRepSamplePath(AcrossRepStat):

    def __init__(self,
                 num_reps: int,
                 num_days: int):
        self.num_reps = num_reps
        self.num_days = num_days

        self.data = np.zeros(shape=(num_reps, num_days + 1))
        self.df_simple_spaghetti = None

    def create_df_simple_spaghetti(self):
        self.df_simple_spaghetti = \
            transform_matrix_to_long_df(self.data.T,
                                        colnames=list(range(self.num_reps)),
                                        id_col="day",
                                        var_name="simulation_idx",
                                        id_values=np.arange(1, 2 + self.num_days),
                                        value_name="result")

    def show_simple_spaghetti(self,
                              ylabel_str=""):
        self.create_df_simple_spaghetti()

        plt.figure(figsize=(10, 6))

        sns.lineplot(
            data=self.df_spaghetti,
            x="day",
            y="result",
            units="simulation_idx",
            alpha=0.1,
            estimator=None
        )

        plt.xlabel("Days")
        plt.ylabel(ylabel_str)

        plt.show()


class AcrossRepPoint(AcrossRepStat):

    def __init__(self,
                 num_reps: int):
        self.num_reps = num_reps
        self.data = np.zeros(shape=(num_reps,))

    def get_exceedance_probability(self,
                                   threshold: int):
        return np.mean(self.data >= threshold)

    def get_conditional_mean_on_exceedance(self,
                                           threshold: int):
        data_subset = self.data[self.data >= threshold]

        if data_subset.size == 0:
            return -1
        else:
            return data_subset.mean()

    def get_quantiles_conditional_on_exceedance(self,
                                                quantiles_list: list[float],
                                                threshold: int):
        data_subset = self.data[self.data >= threshold]
        return [np.percentile(data_subset, q) for q in quantiles_list]

    def get_index_sim_median(self):
        data = self.data
        idx_median = np.argmin(np.abs(data - np.median(data)))

        return idx_median

    def show_histogram(self,
                       xlabel_str: str = ""):
        plt.figure(figsize=(10, 6))

        sns.histplot(
            data=self.data,
            stat="percent"
        )

        plt.xlabel(xlabel_str)
        plt.ylabel("Probability (%)")

        plt.show()

    def get_dashboard_results_strs(self,
                                   init_infected: int,
                                   threshold: int,
                                   lb_quantile: float = 2.5,
                                   ub_quantile: float = 97.5):
        """
        Returns 2 strings to populate the written text portion of the dashboard
        - 1st string corresponds to probability of exceeding X new infections,
          where X is the chosen outbreak threshold_value
        - 2nd corresponds to likely (expected) outbreak size if there are X+ new infections

        """

        exceedance_prob = self.get_exceedance_probability(threshold)

        if exceedance_prob < 0.01:
            exceedance_prob_str = "< 1%"
        elif exceedance_prob > 0.99:
            exceedance_prob_str = "> 99%"
        else:
            exceedance_prob_str = '{:.0%}'.format(exceedance_prob)

        new_cases_conditional_mean = self.get_conditional_mean_on_exceedance(threshold)

        if new_cases_conditional_mean == -1:
            all_cases_conditional_quantiles_str = "Fewer than {} new infections".format(int(threshold))

        else:

            lb_new, ub_new = \
                self.get_quantiles_conditional_on_exceedance([lb_quantile,
                                                              ub_quantile], threshold)

            lb_total, ub_total = init_infected + lb_new, init_infected + ub_new

            all_cases_conditional_quantiles_str = str(int(lb_total)) + ' - ' + str(int(ub_total)) + " total cases"

        return exceedance_prob_str, all_cases_conditional_quantiles_str

# %% Experiment Classes
#######################


class Experiment(ABC):

    def __init__(self,
                 params: MeaslesParameters,
                 num_reps: int):
        self.params = copy.deepcopy(params)
        self.num_reps = num_reps

        self.model = MetapopSEIR(self.params)
        self.model.RNG = np.random.Generator(np.random.MT19937(seed=params["simulation_seed"]))

        self.run()

    @abstractmethod
    def run(self):
        pass


class DashboardExperiment(Experiment):

    def __init__(self,
                 params: MeaslesParameters,
                 num_reps: int):
        self.ma7_num_infected_school_1 = AcrossRepSamplePath(num_reps=num_reps,
                                                             num_days=params['sim_duration_days'])
        self.total_new_cases_school_1 = AcrossRepPoint(num_reps=num_reps)

        super().__init__(params, num_reps)

    def run(self):

        ma7_num_infected = self.ma7_num_infected_school_1.data
        total_new_cases = self.total_new_cases_school_1.data

        model = self.model

        for rep in range(self.num_reps):
            model.simulate()

            ma7_num_infected[rep] = \
                calculate_np_moving_average(
                    np.add(model.I[::model.steps_per_day, 0], model.E[::model.steps_per_day, 0]), 7)

            total_new_cases[rep] = np.sum(model.S_to_E[:, 0])

            model.clear()


def run_deterministic_model(params):
    params_deterministic = copy.deepcopy(params)
    params_deterministic['is_stochastic'] = False
    model_deterministic = MetapopSEIR(params_deterministic)
    model_deterministic.simulate()
    model_deterministic.plot_results()


# %% Example usage
##################

if __name__ == "__main__":

    start = time.time()

    demo = DashboardExperiment(DEFAULT_MSP_PARAMS, 200)
    demo.ma7_num_infected_school_1.create_df_simple_spaghetti()

    prob_threshold_plus_new_str, cases_expected_over_threshold_str = \
        demo.total_new_cases_school_1.get_dashboard_results_strs(DEFAULT_MSP_PARAMS["I0"][0],
                                                                 DEFAULT_MSP_PARAMS["threshold_values"][0],
                                                                 2.5,
                                                                 97.5)

    print(time.time() - start)

    print(prob_threshold_plus_new_str, cases_expected_over_threshold_str)
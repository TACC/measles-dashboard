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

import plotly.express as px
from enum import Enum

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


# %% Constants
###############
class OUTBREAK_SIZE_UNCERTAINTY_OPTIONS(Enum):
    NINETY = '90'
    NINETY_FIVE = '95'
    RANGE = 'range'
    IQR = 'IQR'


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
        np_array: np.ndarray,
        window: int,
        shorter_window_beginning: bool = True) -> np.ndarray:
    """
    This takes a numpy 2-dimensional array and computes moving averages along
    the columns: each row represents a different time series.

    Parameters
    ----------
    np_array : numpy 2D array
        DESCRIPTION.
    window : int
        Length of moving average window.
    shorter_window_beginning : boolean, optional
        If True, we calculate the moving average for the first few observations
        (length window - 1) using the shorter window of available data. If 
        False, values are NAs.

    Returns
    -------
    numpy 2D array of same size as input.

    """

    np_array_ma = np_array.copy()

    if shorter_window_beginning:
        starting_col = 0
    else:
        starting_col = window - 1
    for i in range(starting_col, np_array_ma.shape[1]):
        window_i = min(window, i + 1)
        np_array_ma[:, i] = np_array[:, i - window_i + 1:i + 1].mean(axis=1)

    return np_array_ma


def get_percentile_from_list(
        values_list: list[float],
        percentile_value: float,
        error_value=0.0) -> float:
    if len(values_list) > 0:
        return np.percentile(values_list, q=percentile_value)
    else:
        return error_value


# %% Model
##########
class MetapopulationSEPIR:

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

        # Initialize compartments as 2D arrays [time_step, population]
        self.S = np.zeros((self.n_steps, self.n_pop))
        self.E = np.zeros((self.n_steps, self.n_pop))
        self.P = np.zeros((self.n_steps, self.n_pop))
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

        self.I[0] = np.array(params['I0'])
        self.R[0] = [
            int(self.N[i_pop] * params['vax_prop'][i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.R[0] = [
            min(self.R[0, i_pop], self.N[i_pop] - self.I[0, i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S[0] = self.N - self.I[0] - self.R[0]

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

            # Update compartments using Euler's method
            self.S[self.current_step + 1, ix_pop] = self.S[self.current_step, ix_pop] + updates["dS"]
            self.E[self.current_step + 1, ix_pop] = self.E[self.current_step, ix_pop] + updates["dE"]
            self.I[self.current_step + 1, ix_pop] = self.I[self.current_step, ix_pop] + updates["dI"]
            self.R[self.current_step + 1, ix_pop] = self.R[self.current_step, ix_pop] + updates["dR"]

            # Also update transition variables history
            self.S_to_E[self.current_step + 1, ix_pop] = updates["dS_out"]
            self.E_to_I[self.current_step + 1, ix_pop] = updates["dE_out"]
            self.I_to_R[self.current_step + 1, ix_pop] = updates["dI_out"]

        self.current_step += 1
        return True

    def simulate(self):
        """Run simulation for all time steps"""
        while self.step():
            pass

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


class StochasticSimulations:
    """
    Runs stochastic simulations based on passed parameters, and then calculates
    summary statistics and creates plots.
    """

    def __init__(self, params, n_sim, print_summary_stats=False,
                 show_plots=True):
        self.params = copy.deepcopy(params)
        self.params['sigma'] = 1.0 / self.params['incubation_period']
        self.params['gamma'] = 1.0 / self.params['infectious_period']

        self.RNG = np.random.Generator(np.random.MT19937(seed=params["simulation_seed"]))

        self.n_sim = n_sim
        self.print_summary_stats = print_summary_stats
        self.show_plots = show_plots
        self.steps_per_day = int(np.round(1.0 / params['time_step_days'], 0))

        self.new_infected_school1 = np.zeros(shape=(n_sim))
        self.infectious_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.infected_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.incidence_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))

        self.mean_outbreak_given_20_new_infections = 'NA'
        self.mean_outbreak_given_20_new_infections_min = 'NA'
        self.mean_outbreak_given_20_new_infections_max = 'NA'

        self.run_stochastic_model()
        self.calculate_summary_statistics_stochastic_runs()
        self.create_plots()

    def run_stochastic_model(self):

        for i_sim in range(self.n_sim):
            # For reproducibility, create a FIXED starting point
            #   for the random number generator. Simulations and
            #   random variables are still RANDOM (pseudorandom,
            #   as all computer code is), but the random starting
            #   point is the same. Therefore, the pseudorandomness
            #   does NOT change after refreshing or re-running an experiment,
            #   and results are truly reproducible.

            # NOTE: creating a new random number generator for every
            #   replication is actually REALLY slow. Here, we just use
            #   ONE random number generator rather than a new jumped one
            #   for each replication. Still fixes reproducibility.
            #   Still draws independent random numbers (recall that
            #   RNGs move in-place after spitting out random numbers --
            #   so their next sample starts where things last left off.)

            # Create and run model
            model = MetapopulationSEPIR(self.params)
            model.RNG = self.RNG
            model.simulate()

            # Update so I0 is not param input, but what I0 was really used
            self.cumulative_new_infected_pop_1 = model.R[:, 0] - model.R[0, 0] - model.I[0, 0]  # params['I0'][0]
            self.new_infected_school1[i_sim] = self.cumulative_new_infected_pop_1[-1]

            self.infectious_school_1[i_sim, :] = (
                model.I[::self.steps_per_day, 0]
            )
            self.infected_school_1[i_sim, :] = (
                    model.I[::self.steps_per_day, 0] + model.E[::self.steps_per_day, 0]
            )

            # At Lauren's request -- from Slack 03042025 --
            # "just the number of new transitions from S to E"
            # Note -- because incidence is S_to_E at a given time (and not cumulative),
            #   we have to SUM all the people that move from S_to_E within a day --
            #   i.e. summing all the people that move from S to E over self.steps_per_day --
            #   this is DIFFERENT than checking I every day (every self.steps_per_day)

            S_to_E_school_1 = model.S_to_E[:, 0]
            total_steps = len(S_to_E_school_1)

            self.incidence_school_1[i_sim, :] = \
                np.add.reduceat(S_to_E_school_1, np.arange(0, total_steps, self.steps_per_day))

            # assert np.sum(self.incidence_school_1[i_sim, :]) == \
            #       self.cumulative_new_infected_pop_1[-1]

            self.incidence_school_1_7day_ma = calculate_np_moving_average(self.incidence_school_1, 7)

            self.infected_school_1_7day_ma = calculate_np_moving_average(self.infected_school_1, 7)

            self.model = model

        return

    def calculate_summary_statistics_stochastic_runs(self):

        unique, counts = np.unique(self.new_infected_school1, return_counts=True)
        df_infected_1 = pd.DataFrame({
            'new_infected': unique,
            'num_simulations': counts
        })
        df_infected_1['probability'] = df_infected_1['num_simulations'] / self.n_sim

        median_new_infected = np.median(self.new_infected_school1)
        self.index_sim_closest_median = min(
            range(len(self.new_infected_school1)),
            key=lambda i: abs(self.new_infected_school1[i] - median_new_infected)
        )

        self.probability_5_plus_cases = df_infected_1.loc[
            df_infected_1['new_infected'] >= 5, 'probability'].sum()
        self.probability_10_plus_cases = df_infected_1.loc[
            df_infected_1['new_infected'] >= 10, 'probability'].sum()
        self.probability_20_plus_cases = df_infected_1.loc[
            df_infected_1['new_infected'] >= 20, 'probability'].sum()

        p_5_pct = '{:.0%}'.format(self.probability_5_plus_cases)
        p_10_pct = '{:.0%}'.format(self.probability_10_plus_cases)
        p_20_pct = '{:.0%}'.format(self.probability_20_plus_cases)

        self.expected_infections_all_sim = self.new_infected_school1.mean()
        df_over_20 = df_infected_1.loc[
            df_infected_1['new_infected'] >= 20]  # , 'new_infected']
        if len(df_over_20) > 0:
            # change param input infections to what was used by the model
            self.mean_outbreak_given_20_new_infections = self.model.I[0, 0] + \
                                                         (df_over_20['new_infected'] * df_over_20[
                                                             'probability']).sum() / \
                                                         df_over_20['probability'].sum()
            cases_over_20 = self.new_infected_school1[
                self.new_infected_school1 >= 20]
            quantile_list = [0, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 100]
            self.expected_outbreak_quantiles = {
                q: get_percentile_from_list(cases_over_20, q)
                for q in quantile_list
            }
        # else: expected outbreak attributes remain NA

        self.df_infected_1 = df_infected_1

        if self.print_summary_stats:
            print('Probability of 5 or more cases in outbreak:', p_5_pct)
            print('Probability of 10 or more cases in outbreak:', p_10_pct)
            print('Probability of 20 or more cases in outbreak:', p_20_pct)

            print('Expected number of infections across all simulations:',
                  int(self.expected_infections_all_sim), 'cases')

            if self.mean_outbreak_given_20_new_infections == 'NA':
                mean_outbreak_given_20_new_infections_print = self.mean_outbreak_given_20_new_infections
            else:
                mean_outbreak_given_20_new_infections_print = int(self.mean_outbreak_given_20_new_infections)
            print('Expected number of infections across outbreaks of size 20 or more:',
                  mean_outbreak_given_20_new_infections_print, 'cases')

        return

    def create_plots(self):

        plot_data_dict = {
            "df_spaghetti_infected": (self.infected_school_1, "number_infected"),
            "df_spaghetti_infected_ma": (self.infected_school_1_7day_ma, "number_infected_7_day_ma"),
            "df_spaghetti_infectious": (self.infectious_school_1, "number_infectious"),
            "df_spaghetti_incidence": (self.incidence_school_1, "number_incidence")
        }

        id_values = list(range(1, 2 + MSP_PARAMS['sim_duration_days']))

        for df_name, (matrix, value_name) in plot_data_dict.items():
            setattr(self, df_name, transform_matrix_to_long_df(
                matrix.T,
                colnames=list(range(self.n_sim)),
                id_col="day",
                var_name="simulation_idx",
                id_values=id_values,
                value_name=value_name
            ))

        if self.show_plots:

            nb_plots = 4
            fig, axs = plt.subplots(nb_plots, 1, figsize=(10, 6 * nb_plots))

            # Histogram
            sns.histplot(
                data=self.new_infected_school1,
                stat="percent",
                ax=axs[0]
            )
            axs[0].set_xlabel("Total infections")
            axs[0].set_ylabel("Probability (%)")

            # Line charts
            plot_data = [
                (self.df_spaghetti_infected, "number_infected", "Infected individuals"),
                (self.df_spaghetti_infectious, "number_infectious", "Infectious individuals"),
                (self.df_spaghetti_incidence, "number_incidence", "Newly exposed individuals"),
            ]

            for ax, (data, y_col, ylabel) in zip(axs[1:], plot_data):
                sns.lineplot(
                    data=data,
                    x="day",
                    y=y_col,
                    units="simulation_idx",
                    alpha=0.1,
                    estimator=None,
                    ax=ax
                )
                ax.set_xlabel("Days since outbreak start")
                ax.set_ylabel(ylabel)

            plt.show()


def run_deterministic_model(params):
    # # Create and run model
    params_deterministic = copy.deepcopy(params)
    params_deterministic['is_stochastic'] = False
    model_deterministic = MetapopulationSEPIR(params_deterministic)
    model_deterministic.simulate()
    model_deterministic.plot_results()


def create_strs_20plus_new_and_outbreak(sim: StochasticSimulations,
                                        outbreak_size_uncertainty_displayed: OUTBREAK_SIZE_UNCERTAINTY_OPTIONS):
    """
    Sorry for this UGLY function name :)

    Returns 2 strings to populate the written text portion of the dashboard
    - 1st string corresponds to probability of exceeding 20 new infections
    - 2nd corresponds to likely (expected) outbreak size if there are 20+ new infections

    TODO: in the future, we can write this function and also related functions
        in the StochasticSims class to take in an arbitrary new infection cut-off,
        not just hardcoded 20.
    """

    prob_20plus_new_str = '{:.0%}'.format(sim.probability_20_plus_cases)

    if sim.mean_outbreak_given_20_new_infections == 'NA':
        cases_expected_over_20_str = "Fewer than 20 new infections"

    else:

        if outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY:
            quantile_lb, quantile_ub, range_name = 5, 95, '90% CI'
        elif outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY_FIVE:
            quantile_lb, quantile_ub, range_name = 2.5, 97.5, '95% CI'
        elif outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.RANGE:
            quantile_lb, quantile_ub, range_name = 0, 100, 'range'
        elif outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.IQR:
            quantile_lb, quantile_ub, range_name = 25, 75, 'IQR'

        uncertainty_outbreak_size_str = str(int(sim.expected_outbreak_quantiles[quantile_lb])) + ' - ' + \
                                        str(int(sim.expected_outbreak_quantiles[quantile_ub]))

        cases_expected_over_20_str = uncertainty_outbreak_size_str + " total cases"

    return prob_20plus_new_str, cases_expected_over_20_str


# Set parameters
# Natural history parameters
# https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html
# Incubation: 11.5 days, so first symptoms 11.5 days since t0
# Rash starts: 3 days after first symptoms, so 14.5 since t0
# Infectious: 4 days before rash (so 10.5 days since t0), 4 days after (18.5 days since t0)

MSP_PARAMS = {
    'R0': 15.0,  # transmission rate
    # 'sigma': 1/10.5,    # 10.5 days average latent period
    # 'rho': 1/1,         # 1 days average pre-symptomatic period
    # 'gamma': 1/8,       # 7 days average infectious period
    'incubation_period': 10.5,
    'infectious_period': 8.0,
    'school_contacts': 5.63424,
    'other_contacts': 2.2823,
    'population': [500],
    'I0': [1],
    'vax_prop': [0.9],  # number between 0 and 1
    'sim_duration_days': 250,
    'time_step_days': 0.25,
    'is_stochastic': True,  # False for deterministic,
    "simulation_seed": 147125098488
}

# %% Main
##########
# Example usage
if __name__ == "__main__":
    run_deterministic_model(MSP_PARAMS)

    # Stochastic runs
    # n_sim = 200
    n_sim = 20
    stochastic_sim = StochasticSimulations(
        MSP_PARAMS, n_sim, print_summary_stats=True, show_plots=True)
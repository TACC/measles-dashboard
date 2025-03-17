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
        self.sigma_vax = 1.0 / params["incubation_period_vaccinated"]  # rate of progression from E to I for vaccinated
        self.gamma_vax = 1.0 / params["infectious_period_vaccinated"]  # recovery rate for vaccinated
        
        self.beta_vax = params['relative_infectiousness_vaccinated'] * params['R0'] / \
            (params["incubation_period_vaccinated"] * self.total_contacts)
        self.skip_vax_to_R = params['skip_vaccinated_to_recovered'] # Go from V to R_V, no E_V and I_V steps
        self.vaccine_efficacy = params['vaccine_efficacy']

        # Time array
        self.t = np.linspace(0, params['sim_duration_days'], self.n_steps)

        self.params = params

        # Initialize compartments as 2D arrays [time_step, population]
        self.S = np.zeros((self.n_steps, self.n_pop))
        self.E = np.zeros((self.n_steps, self.n_pop))
        self.I = np.zeros((self.n_steps, self.n_pop))
        self.R = np.zeros((self.n_steps, self.n_pop))
        self.V = np.zeros((self.n_steps, self.n_pop)) # vaccinated
        self.E_V = np.zeros((self.n_steps, self.n_pop)) # vaccinated breakthrough infections
        self.I_V = np.zeros((self.n_steps, self.n_pop)) # vaccinated breakthrough infections
        self.R_V = np.zeros((self.n_steps, self.n_pop)) # vaccinated breakthrough infections

        # Add storage of transition variables
        # Used to compute incidence
        # Looks like we are currently not using P so skipping that
        self.S_to_E = np.zeros((self.n_steps, self.n_pop))
        self.E_to_I = np.zeros((self.n_steps, self.n_pop))
        self.I_to_R = np.zeros((self.n_steps, self.n_pop))
        self.V_to_E_V = np.zeros((self.n_steps, self.n_pop))
        self.E_V_to_I_V = np.zeros((self.n_steps, self.n_pop))
        self.I_V_to_R_V = np.zeros((self.n_steps, self.n_pop))

        # Set initial conditions
        # 03052025 1:1 with Lauren -- Lauren does not want
        #   us to change the user input in the backend -- she wants
        #   to have guard rails on the user inputs

        self.I[0] = np.array(params['I0'])
        self.V[0] = [
            int(self.N[i_pop] * params['vax_prop'][i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.V[0] = [
            min(self.V[0, i_pop], self.N[i_pop] - self.I[0, i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S[0] = self.N - self.I[0] - self.V[0]

        # Current step tracker
        self.current_step = 0

        self.df_spaghetti_infectious = pd.DataFrame([])
        self.df_spaghetti_infected = pd.DataFrame([])
        self.df_spaghetti_infected_ma = pd.DataFrame([])
        self.df_spaghetti_incidence = pd.DataFrame([])

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

        if total_rate > 0.0:
            if self.is_stochastic:
                delta = self.RNG.poisson(total_rate)
            else:
                delta = total_rate
                
            delta = min(delta, compartment_count)
        else:
            delta = 0.0
        
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

        force_of_infection = self.total_contacts * dt * \
            (self.beta * self.I[t, ix_pop] + self.beta_vax * self.I_V[t, ix_pop]) / self.N[ix_pop]
        force_of_infection_vaccinated =  \
            force_of_infection * (1 - self.vaccine_efficacy)

        dS_out = self.get_compartment_transition(force_of_infection, self.S[t, ix_pop])
        dE_out = self.get_compartment_transition(self.sigma * dt, self.E[t, ix_pop])
        dI_out = self.get_compartment_transition(self.gamma * dt, self.I[t, ix_pop])
        
        dV_out = self.get_compartment_transition(force_of_infection_vaccinated, self.V[t, ix_pop])
        if not(self.skip_vax_to_R):
            dE_V_out = self.get_compartment_transition(self.sigma_vax * dt, self.E_V[t, ix_pop])
            dI_V_out = self.get_compartment_transition(self.gamma_vax * dt, self.I_V[t, ix_pop])
        else:
            dE_V_out = 0
            dI_V_out = 0

        dS = -dS_out
        dE = dS_out - dE_out
        dI = dE_out - dI_out
        dR = dI_out
        
        dV = -dV_out
        if not(self.skip_vax_to_R):
            dE_V = dV_out - dE_V_out
            dI_V = dE_V_out - dI_V_out
            dR_V = dI_V_out
        else:
            dE_V = 0
            dI_V = 0
            dR_V = dV_out

        updates_dict = {
            "dS": dS, "dE": dE, "dI": dI, "dR": dR,
            "dV": dV, "dE_V": dE_V, "dI_V": dI_V, "dR_V": dR_V,
            "dS_out": dS_out, "dE_out": dE_out, "dI_out": dI_out,
            "dV_out": dV_out, "dE_V_out": dE_V_out, "dI_V_out": dI_V_out,
            }

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
            
            self.V[self.current_step + 1, ix_pop] = self.V[self.current_step, ix_pop] + updates["dV"]
            self.E_V[self.current_step + 1, ix_pop] = self.E_V[self.current_step, ix_pop] + updates["dE_V"]
            self.I_V[self.current_step + 1, ix_pop] = self.I_V[self.current_step, ix_pop] + updates["dI_V"]
            self.R_V[self.current_step + 1, ix_pop] = self.R_V[self.current_step, ix_pop] + updates["dR_V"]

            # Also update transition variables history
            self.S_to_E[self.current_step + 1, ix_pop] = updates["dS_out"]
            self.E_to_I[self.current_step + 1, ix_pop] = updates["dE_out"]
            self.I_to_R[self.current_step + 1, ix_pop] = updates["dI_out"]
            
            self.V_to_E_V[self.current_step + 1, ix_pop] = updates["dV_out"]
            self.E_V_to_I_V[self.current_step + 1, ix_pop] = updates["dE_V_out"]
            self.I_V_to_R_V[self.current_step + 1, ix_pop] = updates["dI_V_out"]

        self.current_step += 1
        return True

    def simulate(self):
        """Run simulation for all time steps"""
        while self.step():
            pass

    def clear(self):
        # Reset stored values to defaults / empty arrays

        self.S = np.zeros((self.n_steps, self.n_pop))
        self.E = np.zeros((self.n_steps, self.n_pop))
        self.I = np.zeros((self.n_steps, self.n_pop))
        self.R = np.zeros((self.n_steps, self.n_pop))
        self.V = np.zeros((self.n_steps, self.n_pop)) # vaccinated
        self.E_V = np.zeros((self.n_steps, self.n_pop)) # vaccinated breakthrough infections
        self.I_V = np.zeros((self.n_steps, self.n_pop)) # vaccinated breakthrough infections
        self.R_V = np.zeros((self.n_steps, self.n_pop)) # vaccinated breakthrough infections

        self.S_to_E = np.zeros((self.n_steps, self.n_pop))
        self.E_to_I = np.zeros((self.n_steps, self.n_pop))
        self.I_to_R = np.zeros((self.n_steps, self.n_pop))
        self.V_to_E_V = np.zeros((self.n_steps, self.n_pop))
        self.E_V_to_I_V = np.zeros((self.n_steps, self.n_pop))
        self.I_V_to_R_V = np.zeros((self.n_steps, self.n_pop))

        params = self.params

        self.I[0] = np.array(params['I0'])
        self.V[0] = [
            int(self.N[i_pop] * params['vax_prop'][i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.V[0] = [
            min(self.V[0, i_pop], self.N[i_pop] - self.I[0, i_pop])
            for i_pop in range(self.n_pop)
        ]
        self.S[0] = self.N - self.I[0] - self.V[0]

        self.current_step = 0

    def plot_results(self):
        """Plot the results for all populations"""
        fig, axes = plt.subplots(self.n_pop, 1, figsize=(10, 6 * self.n_pop))
        if self.n_pop == 1:
            axes = [axes]

        for pop in range(self.n_pop):
            ax = axes[pop]
            vaccinated_linestyle = 'dashed'
            ax.plot(self.t, self.S[:, pop], label='Susceptible', color='blue')
            ax.plot(self.t, self.E[:, pop], label='Exposed', color='orange')
            ax.plot(self.t, self.I[:, pop], label='Infectious', color='red')
            ax.plot(self.t, self.R[:, pop], label='Recovered', color='green')
            ax.plot(self.t, self.V[:, pop], label='Vaccinated', color='blue', linestyle=vaccinated_linestyle)
            ax.plot(self.t, self.E_V[:, pop], label='Vaccinated Exposed', color='orange', linestyle=vaccinated_linestyle)
            ax.plot(self.t, self.I_V[:, pop], label='Vaccinated Infectious', color='red', linestyle=vaccinated_linestyle)
            ax.plot(self.t, self.R_V[:, pop], label='Vaccinated Recovered', color='green', linestyle=vaccinated_linestyle)
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

    def __init__(self, params, n_sim):
        self.params = copy.deepcopy(params)
        # self.params['sigma'] = 1.0 / self.params['incubation_period']
        # self.params['gamma'] = 1.0 / self.params['infectious_period']

        self.RNG = np.random.Generator(np.random.MT19937(seed=params["simulation_seed"]))

        self.n_sim = n_sim
        self.steps_per_day = int(np.round(1.0 / params['time_step_days'], 0))

        self.new_infected_school1 = np.zeros(shape=(n_sim))
        self.new_infected_vaccinated_school1 = np.zeros(shape=(n_sim))
        self.infectious_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.infectious_vaccinated_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.infected_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.infected_vaccinated_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.incidence_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))
        self.incidence_vaccinated_school_1 = np.zeros(shape=(n_sim, params['sim_duration_days'] + 1))

        self.threshold_values_list = params.get('threshold_values', [20]) # to be passed as input if user chosen
        self.mean_outbreak_given_threshold_new_infections = {
            x: {'mean': 'NA', 
                'min': 'NA', 
                'max': 'NA',}
            for x in self.threshold_values_list
        }
        self.mean_outbreak_vaccinated_given_threshold_new_infections = copy.deepcopy(
            self.mean_outbreak_given_threshold_new_infections
        )
        
        self.index_sim_closest_median = None
        self.outbreak_given_threshold_new_infections_quantiles = {
            x: {}
            for x in self.threshold_values_list
        }
        self.outbreak_vaccinated_given_threshold_new_infections_quantiles = copy.deepcopy(
            self.outbreak_given_threshold_new_infections_quantiles
        )
        self.outbreak_over_threshold_infections_str = {
            x: {'probability': 'NA', 'range': 'NA'}
            for x in self.threshold_values_list
        }
        self.outbreak_vaccinated_over_threshold_infections_str = copy.deepcopy(
            self.outbreak_over_threshold_infections_str)
        # self.outbreak_given_20_new_infections_quantiles = {}
        self.df_new_infected_empirical_dist = pd.DataFrame([])
        self.df_new_infected_vaccinated_empirical_dist = pd.DataFrame([])

        self.probability_threshold_plus_cases = {}
        self.probability_threshold_plus_cases_vaccinated = {}
        
        # self.probability_5_plus_new_cases = None
        # self.probability_10_plus_new_cases = None
        # self.probability_20_plus_new_cases = None

        self.incidence_school_1_7day_ma = np.array([])
        self.incidence_vaccinated_school_1_7day_ma = np.array([])
        self.infected_school_1_7day_ma = np.array([])
        self.infected_vaccinated_school_1_7day_ma = np.array([])

        self.model = MetapopulationSEPIR(self.params)
        self.model.RNG = self.RNG

    def run_stochastic_model(self,
                             track_infectious=False,
                             track_infected=False,
                             track_incidence=False,
                             combine_vax_curves=False,
                             combine_vax_stats=False,
                             print_summary_stats=False,
                             show_plots=False):
        """_summary_

        Parameters
        ----------
        combine_vax_curves : bool, optional
            If True the vaccinated breakthrough compartments are combined
            with the non-vaccinated for variables used to create curbes. 
            Otherwise vaccinated specific variables are populated.
            Default: False.
        combine_vax_stats : bool, optional
            Same as combine_vax_curves, but for summary statistics variables.
            Default: False.
        """
        self.combine_vax_curves = combine_vax_curves
        self.combine_vax_stats = combine_vax_stats

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

            model = self.model

            model.simulate()

            if self.combine_vax_stats:
                self.cumulative_new_infected_pop_1 = model.R[:, 0] - model.R[0, 0] - self.params['I0'][0] +\
                    model.R_V[:, 0] - model.R_V[0, 0]
                self.new_infected_school1[i_sim] = self.cumulative_new_infected_pop_1[-1]
            else:
                self.cumulative_new_infected_pop_1 = model.R[:, 0] - model.R[0, 0] - self.params['I0'][0]
                self.new_infected_school1[i_sim] = self.cumulative_new_infected_pop_1[-1]
                
                self.cumulative_new_infected_vaccinated_pop_1 = model.R_V[:, 0] - model.R_V[0, 0]
                self.new_infected_vaccinated_school1[i_sim] = self.cumulative_new_infected_vaccinated_pop_1[-1]

            if track_infectious:
                if self.combine_vax_curves:
                    self.infectious_school_1[i_sim, :] = (
                        model.I[::self.steps_per_day, 0] + model.I_V[::self.steps_per_day, 0]
                    )
                else:
                    self.infectious_school_1[i_sim, :] = (
                        model.I[::self.steps_per_day, 0]
                    )
                    self.infectious_vaccinated_school_1[i_sim, :] = (
                        model.I_V[::self.steps_per_day, 0]
                    )

            if track_infected:
                if self.combine_vax_curves:
                    self.infected_school_1[i_sim, :] = (
                            model.I[::self.steps_per_day, 0] + model.E[::self.steps_per_day, 0] +\
                            model.I_V[::self.steps_per_day, 0] + model.E_V[::self.steps_per_day, 0]
                    )
                else:
                    self.infected_school_1[i_sim, :] = (
                        model.I[::self.steps_per_day, 0] + model.E[::self.steps_per_day, 0]
                    )
                    self.infected_vaccinated_school_1[i_sim, :] = (
                        model.I_V[::self.steps_per_day, 0] + model.E_V[::self.steps_per_day, 0]
                    )

            if track_incidence:
                # At Lauren's request -- from Slack 03042025 --
                # "just the number of new transitions from S to E"
                # Note -- because incidence is S_to_E at a given time (and not cumulative),
                #   we have to SUM all the people that move from S_to_E within a day --
                #   i.e. summing all the people that move from S to E over self.steps_per_day --
                #   this is DIFFERENT than checking I every day (every self.steps_per_day)

                if self.combine_vax_curves:
                    S_to_E_school_1 = model.S_to_E[:, 0] + model.V_to_E_V[:, 0]
                    total_steps = len(S_to_E_school_1)

                    self.incidence_school_1[i_sim, :] = \
                        np.add.reduceat(S_to_E_school_1, np.arange(0, total_steps, self.steps_per_day))
                else:
                    S_to_E_school_1 = model.S_to_E[:, 0]
                    V_to_E_V_school_1 = model.V_to_E_V[:, 0]
                    total_steps = len(S_to_E_school_1)

                    self.incidence_school_1[i_sim, :] = \
                        np.add.reduceat(S_to_E_school_1, np.arange(0, total_steps, self.steps_per_day))
                    self.incidence_vaccinated_school_1[i_sim, :] = \
                        np.add.reduceat(V_to_E_V_school_1, np.arange(0, total_steps, self.steps_per_day))

                # assert np.sum(self.incidence_school_1[i_sim, :]) == \
                #       self.cumulative_new_infected_pop_1[-1]

            model.clear()

        self.process_across_rep_output()

        if print_summary_stats:
            self.print_summary_stats()
        if show_plots:
            self.create_plots()

        return

    def prep_across_rep_plot_data(self,
                                  include_infectious=False,
                                  include_infected=False,
                                  include_infected_7day_ma=True,
                                  include_incidence=False,
                                  include_incidence_7day_ma=False):

        plot_data_dict = {}

        if include_infectious:
            plot_data_dict["df_spaghetti_infectious"] = (self.infectious_school_1, "number_infectious")
            
            if not(self.combine_vax_curves):
                plot_data_dict["df_spaghetti_infectious_vaccinated"] = (
                    self.infectious_vaccinated_school_1, "number_infectious_vaccinated")

        if include_infected:
            plot_data_dict["df_spaghetti_infected"] = (self.infected_school_1, "number_infected")
            
            if not(self.combine_vax_curves):
                plot_data_dict["df_spaghetti_infected_vaccinated"] = (self.infected_vaccinated_school_1, "number_infected_vaccinated")

        if include_infected_7day_ma:
            self.infected_school_1_7day_ma = calculate_np_moving_average(self.infected_school_1, 7)
            plot_data_dict["df_spaghetti_infected_ma"] = (self.infected_school_1_7day_ma, "number_infected_7_day_ma")
            
            if not(self.combine_vax_curves):
                self.infected_vaccinated_school_1_7day_ma = calculate_np_moving_average(self.infected_vaccinated_school_1, 7)
                plot_data_dict["df_spaghetti_infected_vaccinated_ma"] = (
                    self.infected_vaccinated_school_1_7day_ma, "number_infected_vaccinated_7_day_ma")

        if include_incidence:
            plot_data_dict["df_spaghetti_incidence"] = (self.incidence_school_1, "number_incidence")
            
            if not(self.combine_vax_curves):
                plot_data_dict["df_spaghetti_incidence_vaccinated"] = (
                    self.incidence_vaccinated_school_1, "number_incidence_vaccinated")
        
        if include_incidence_7day_ma:
            self.incidence_school_1_7day_ma = calculate_np_moving_average(self.incidence_school_1, 7)
            plot_data_dict["df_spaghetti_incidence_ma"] = (self.incidence_school_1_7day_ma, "number_incidence_7_day_ma")
            
            if not(self.combine_vax_curves):
                self.incidence_vaccinated_school_1_7day_ma = calculate_np_moving_average(self.incidence_vaccinated_school_1, 7)
                plot_data_dict["df_spaghetti_incidence_vaccinated_ma"] = (
                    self.incidence_vaccinated_school_1_7day_ma, "number_incidence_vaccinated_7_day_ma")

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

    def process_across_rep_output(self):

        self.compute_index_sim_closest_median()
        self.compute_new_infected_empirical_dist()

    def compute_index_sim_closest_median(self):
        median_new_infected = np.median(self.new_infected_school1)
        self.index_sim_closest_median = min(
            range(len(self.new_infected_school1)),
            key=lambda i: abs(self.new_infected_school1[i] - median_new_infected)
        )
        
    def compute_new_infected_empirical_dist(self):
        
        unique, counts = np.unique(self.new_infected_school1, return_counts=True)
        self.df_new_infected_empirical_dist = pd.DataFrame({
            'new_infected': unique,
            'num_simulations': counts,
            'probability': counts / self.n_sim
        })
        
        if not(self.combine_vax_stats):
            unique, counts = np.unique(self.new_infected_vaccinated_school1, return_counts=True)
            self.df_new_infected_vaccinated_empirical_dist = pd.DataFrame({
                'new_infected': unique,
                'num_simulations': counts,
                'probability': counts / self.n_sim
            })

    def calculate_threshold_statistis(
        self,
        outbreak_size_uncertainty_displayed: OUTBREAK_SIZE_UNCERTAINTY_OPTIONS):
        
        for threshold_value in self.threshold_values_list:
            self.calculate_threshold_plus_new_cases_statistics(
                threshold_value)
            self.create_strs_threshold_plus_new_and_outbreak(
                outbreak_size_uncertainty_displayed, threshold_value)
            
            if not(self.combine_vax_stats):
                self.calculate_threshold_plus_new_cases_statistics(
                    threshold_value, subgroup='vaccinated')
                self.create_strs_threshold_plus_new_and_outbreak(
                    outbreak_size_uncertainty_displayed, threshold_value,
                    subgroup='vaccinated')
        
    def calculate_threshold_plus_new_cases_statistics(
        self, 
        threshold_value: int,
        subgroup: str = 'all'):
        
        if subgroup == 'vaccinated':
            df_new_infected_empirical_dist = self.df_new_infected_vaccinated_empirical_dist
            df_new_infected = self.new_infected_vaccinated_school1
            probability_threshold_plus_attr_name = 'probability_threshold_plus_cases_vaccinated'
            mean_outbreak_stats_attr_name = 'mean_outbreak_vaccinated_given_threshold_new_infections'
            outbreak_quantiles_attr_name = 'outbreak_vaccinated_given_threshold_new_infections_quantiles'
        else:
            df_new_infected_empirical_dist = self.df_new_infected_empirical_dist
            df_new_infected = self.new_infected_school1
            probability_threshold_plus_attr_name = 'probability_threshold_plus_cases'
            mean_outbreak_stats_attr_name = 'mean_outbreak_given_threshold_new_infections'
            outbreak_quantiles_attr_name = 'outbreak_given_threshold_new_infections_quantiles'


        probability_above_threshold = df_new_infected_empirical_dist.loc[
            df_new_infected_empirical_dist['new_infected'] >= threshold_value, 'probability'].sum()
        getattr(self, probability_threshold_plus_attr_name)[threshold_value] = probability_above_threshold

        df_over_threshold = df_new_infected_empirical_dist.loc[
            df_new_infected_empirical_dist['new_infected'] >= threshold_value]

        if len(df_over_threshold) > 0:
            getattr(self, mean_outbreak_stats_attr_name)[threshold_value]['mean'] = \
                self.params["I0"][0] + (df_over_threshold['new_infected'] *
                    df_over_threshold['probability']).sum() / probability_above_threshold
            cases_over_threshold = df_new_infected[df_new_infected >= threshold_value]

            quantile_list = [0, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 100]
            getattr(self, outbreak_quantiles_attr_name)[threshold_value] = {
                q: get_percentile_from_list(cases_over_threshold, q)
                for q in quantile_list
            }
        # else: expected outbreak attributes remain NA

    def create_strs_threshold_plus_new_and_outbreak(
        self,
        outbreak_size_uncertainty_displayed: OUTBREAK_SIZE_UNCERTAINTY_OPTIONS,
        threshold_value: int,
        subgroup: str = 'all'):
        """
        Sorry for this UGLY function name :)

        Returns 2 strings to populate the written text portion of the dashboard
        - 1st string corresponds to probability of exceeding 20 new infections
        - 2nd corresponds to likely (expected) outbreak size if there are 20+ new infections

        TODO: in the future, we can write this function and also related functions
            in the StochasticSims class to take in an arbitrary new infection cut-off,
            not just hardcoded 20.
        """
        
        if subgroup == 'vaccinated':
            probability_threshold_plus_new_cases = self.probability_threshold_plus_cases_vaccinated[threshold_value]
            mean_outbreak_size_over = self.mean_outbreak_vaccinated_given_threshold_new_infections[threshold_value]['mean']
            outbreak_size_quantiles = self.outbreak_vaccinated_given_threshold_new_infections_quantiles[threshold_value]
            outbreak_infections_str_attr_name = 'outbreak_vaccinated_over_threshold_infections_str'
        else:
            probability_threshold_plus_new_cases = self.probability_threshold_plus_cases[threshold_value]
            mean_outbreak_size_over = self.mean_outbreak_given_threshold_new_infections[threshold_value]['mean']
            outbreak_size_quantiles = self.outbreak_given_threshold_new_infections_quantiles[threshold_value]
            outbreak_infections_str_attr_name = 'outbreak_over_threshold_infections_str'
            

        # probability_20_plus_new_cases = self.probability_20_plus_new_cases
        # probability_threshold_plus_new_cases = self.probability_threshold_plus_cases[threshold_value]

        if probability_threshold_plus_new_cases < 0.01:
            prob_threshold_plus_new_str = "< 1%"
        elif probability_threshold_plus_new_cases > 0.99:
            prob_threshold_plus_new_str = "> 99%"
        else:
            prob_threshold_plus_new_str = '{:.0%}'.format(probability_threshold_plus_new_cases)

        # breakpoint()

        # if self.mean_outbreak_given_20_new_infections == 'NA':
        if mean_outbreak_size_over == 'NA':
            cases_expected_over_threshold_str = "Fewer than {} new infections".format(int(threshold_value))

        else:

            if outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY:
                quantile_lb, quantile_ub, range_name = 5, 95, '90% CI'
            elif outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY_FIVE:
                quantile_lb, quantile_ub, range_name = 2.5, 97.5, '95% CI'
            elif outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.RANGE:
                quantile_lb, quantile_ub, range_name = 0, 100, 'range'
            elif outbreak_size_uncertainty_displayed == OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.IQR:
                quantile_lb, quantile_ub, range_name = 25, 75, 'IQR'

            uncertainty_outbreak_size_str = str(int(outbreak_size_quantiles[quantile_lb])) + ' - ' + \
                    str(int(outbreak_size_quantiles[quantile_ub]))

            cases_expected_over_threshold_str = uncertainty_outbreak_size_str + " total cases"
            getattr(self, outbreak_infections_str_attr_name)[threshold_value] = {
                'probability': prob_threshold_plus_new_str,
                'range': cases_expected_over_threshold_str,
            }

        # return prob_threshold_plus_new_str, cases_expected_over_threshold_str
    
    def print_summary_stats(self):

        df_new_infected_empirical_dist = self.df_new_infected_empirical_dist
        
        sample_thresholds_list = [5, 10, 20]
        for sample_threshold in sample_thresholds_list:
            self.probability_n_plus_new_cases = df_new_infected_empirical_dist.loc[
                df_new_infected_empirical_dist['new_infected'] >= sample_threshold, 'probability'].sum()
            p_n_pct = '{:.0%}'.format(self.probability_n_plus_new_cases)
            print('Probability of', sample_threshold,'or more cases in outbreak:', p_n_pct)

        for threshold_value in self.threshold_values_list:
            mean_outbreak_size = self.mean_outbreak_given_threshold_new_infections[threshold_value]['mean']
            if mean_outbreak_size == 'NA':
                mean_outbreak_size_print = mean_outbreak_size
            else:
                mean_outbreak_size_print = int(mean_outbreak_size)
            print('Expected number of infections across outbreaks of size', str(threshold_value), 'or more:',
                mean_outbreak_size_print, 'cases')

    def create_plots(self):

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

MSP_PARAMS = {
    'R0': 15.0,  # transmission rate
    # 'sigma': 1/10.5,    # 10.5 days average latent period
    # 'rho': 1/1,         # 1 days average pre-symptomatic period
    # 'gamma': 1/8,       # 7 days average infectious period
    'incubation_period': 10.5,
    'infectious_period': 5,
    'incubation_period_vaccinated': 12,
    'infectious_period_vaccinated': 9,
    'relative_infectiousness_vaccinated': 0.05, #
    'vaccine_efficacy': 0.97, # 1.0 means perfect protection
    'skip_vaccinated_to_recovered': False, # if True vaccinated infected go straight to R
    'school_contacts': 5.63424,
    'other_contacts': 2.2823,
    'population': [500],
    'I0': [1],
    'vax_prop': [0.85],  # number between 0 and 1
    'threshold_values': [20], # outbreak threshold
    'sim_duration_days': 250,
    'time_step_days': 0.25,
    'is_stochastic': True,  # False for deterministic,
    "simulation_seed": 147125098488
}

# %% Main
##########
# Example usage
if __name__ == "__main__":
    # run_deterministic_model(MSP_PARAMS)

    # Stochastic runs
    # n_sim = 200
    import time
    start = time.time()

    n_sim = 200 # 2.8811910152435303 to # 1.0790390968322754 for population 500, init infected 1
    stochastic_sim = StochasticSimulations(MSP_PARAMS, n_sim)
    
    stochastic_sim.run_stochastic_model(
        track_infected=True,
        combine_vax_curves=False,
        combine_vax_stats=True)
    stochastic_sim.prep_across_rep_plot_data(include_infected_7day_ma=True)
    
    stochastic_sim.calculate_threshold_statistis(OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY_FIVE)
    for threshold_value in stochastic_sim.threshold_values_list:
        print('\n\nThreshold value:', threshold_value)
        prob_20plus_new_str = stochastic_sim.outbreak_over_threshold_infections_str[threshold_value]['probability']
        range_20plus_new_str = stochastic_sim.outbreak_over_threshold_infections_str[threshold_value]['range']
        prob_20plus_vaccinated_new_str = stochastic_sim.outbreak_vaccinated_over_threshold_infections_str[threshold_value]['probability']
        range_20plus_vaccinated_new_str = stochastic_sim.outbreak_vaccinated_over_threshold_infections_str[threshold_value]['range']
        print('Outbreak threshold:', threshold_value)
        print(prob_20plus_new_str, ' probability, range:', range_20plus_new_str)
        print('\nVaccinated:\n', prob_20plus_vaccinated_new_str, 'probability, range:', range_20plus_vaccinated_new_str)
    # stochastic_sim.calculate_threshold_plus_new_cases_statistics()
    # stochastic_sim.create_strs_20plus_new_and_outbreak(OUTBREAK_SIZE_UNCERTAINTY_OPTIONS.NINETY_FIVE)

    print(time.time() - start)
    
    run_deterministic_model(MSP_PARAMS)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:19:50 2025

@author: rfp437
"""
# %% Imports and parameters
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
# import math

pd.set_option('display.precision', 1)
np.set_printoptions(precision=4)
np.set_printoptions(linewidth=185)
pd.set_option('display.width', 185)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.4}'.format
np.set_printoptions(threshold=sys.maxsize)

# %% Functions 
def transform_matrix_to_long_df(
    np_array, colnames=None, id_col='time_idx', id_values=None, 
    var_name='idx', value_name='value'):
    """
    Takes a 2D numpy array where each column represents a given simulation
    and each row an observation (ie, a date), and transforms the data in
    a pandas dataframe in a long format.
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
    
    return df_long

def calculate_np_moving_average(
    np_array, window, shorter_window_beginning=True):
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
        starting_col = window-1
    for i in range(starting_col, np_array_ma.shape[1]):
        
        window_i = min(window, i+1)
        np_array_ma[:, i] = np_array[:, i-window_i+1:i+1].mean(axis=1)
    
    return np_array_ma
    
def get_percentile_from_list(
    values_list, percentile_value, error_value=0.0):
    
    if len(values_list) > 0:
        return np.percentile(values_list, q=percentile_value)
    else:
        return error_value 

# %% Model

class MetapopulationSEPIR:
    def __init__(self, params):
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
        self.N = np.array(params['population']) # population size
        self.dt = params['time_step_days']
        self.n_steps = 1 + int(
            params['sim_duration_days'] / self.dt)  # number of time steps
        
        
        # Stochastic simulations
        self.is_stochastic = params['is_stochastic']
        if self.is_stochastic:
            self._bit_generator = np.random.MT19937(seed=params['RNG_seed'])
            self.RNG = np.random.Generator(self._bit_generator)
        
        # Number contacts
        self.school_contacts = params['school_contacts']
        self.other_contacts_base = params['other_contacts']
        self.total_contacts = self.school_contacts + self.other_contacts_base
        
        
        # Disease parameters
        total_infectious_duration = 1.0 / params['gamma']
        beta = params['R0'] / (total_infectious_duration * self.total_contacts)
        
        self.beta = beta                    # transmission rate
        self.sigma = params['sigma']        # rate of progression from E to I
        self.gamma = params['gamma']        # recovery rate
        
        # Time array
        self.t = np.linspace(0, params['sim_duration_days'], self.n_steps)
        
        # Initialize compartments as 2D arrays [time_step, population]
        self.S = np.zeros((self.n_steps, self.n_pop))
        self.E = np.zeros((self.n_steps, self.n_pop))
        self.P = np.zeros((self.n_steps, self.n_pop))
        self.I = np.zeros((self.n_steps, self.n_pop))
        self.R = np.zeros((self.n_steps, self.n_pop))
        
        # Set initial conditions
        self.I[0] = np.array(params['I0'])
        self.R[0] = [
            int(self.N[i_pop] * params['vaccinated_percent'][i_pop])
            for i_pop in range(self.n_pop)
            ]
        self.R[0] = [
            min(self.R[0, i_pop], self.N[i_pop] - self.I[0, i_pop])
            for i_pop in range(self.n_pop)
            ]
        self.S[0] = self.N - self.I[0] - self.R[0]
        
        # Current step tracker
        self.current_step = 0
        
    
    def get_compartment_transition(self, rate, compartment_count):
        """
        For a given compartment and a given rate, this calculates the number
        of individuals transitioning out of the compartment.
        The calculations is either deterministic or stochastic.

        Parameters
        ----------
        rate : double
            Force of infection for new infected, rate out of compartment
            otherwise.
        compartment_count : double
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
            
    
    def calculate_compartment_updates(self, i_pop):
        """
        Calculates changes in compartments for population i_pop.

        Parameters
        ----------
        i_pop : int
            Index of population.

        Returns
        -------
        Changes in the number of individuals in each compartment.

        """
        t = self.current_step
        
        force_of_infection = self.beta * self.total_contacts * self.dt * \
             self.I[t, i_pop] / self.N[i_pop]
        
        dS_out = self.get_compartment_transition(force_of_infection, self.S[t, i_pop])
        dE_out = self.get_compartment_transition(self.sigma * self.dt, self.E[t, i_pop])
        dI_out = self.get_compartment_transition(self.gamma * self.dt, self.I[t, i_pop])
        
        dS = -dS_out
        dE = dS_out - dE_out
        dI = dE_out - dI_out
        dR = dI_out
        
        return dS, dE, dI, dR
    
    def step(self):
        """Calculate one time step using Euler's method"""
        if self.current_step >= self.n_steps - 1:
            return False
        
        # Loop through populations
        for i_pop in range(self.n_pop):
            
            dS, dE, dI, dR = self.calculate_compartment_updates(i_pop)
            
            # Update compartments using Euler's method
            self.S[self.current_step + 1, i_pop] = self.S[self.current_step, i_pop] + dS
            self.E[self.current_step + 1, i_pop] = self.E[self.current_step, i_pop] + dE
            self.I[self.current_step + 1, i_pop] = self.I[self.current_step, i_pop] + dI
            self.R[self.current_step + 1, i_pop] = self.R[self.current_step, i_pop] + dR
        
        
        self.current_step += 1
        return True
    
    def simulate(self):
        """Run simulation for all time steps"""
        while self.step():
            pass
    
    def plot_results(self):
        """Plot the results for all populations"""
        fig, axes = plt.subplots(self.n_pop, 1, figsize=(10, 6*self.n_pop))
        if self.n_pop == 1:
            axes = [axes]
        
        for pop in range(self.n_pop):
            ax = axes[pop]
            ax.plot(self.t, self.S[:,pop], label='Susceptible')
            ax.plot(self.t, self.E[:,pop], label='Exposed')
            ax.plot(self.t, self.I[:,pop], label='Infectious')
            ax.plot(self.t, self.R[:,pop], label='Recovered')
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
        
        self.n_sim = n_sim
        self.print_summary_stats = print_summary_stats
        self.show_plots = show_plots
        self.steps_per_day = int(np.round(1.0 / params['time_step_days'],0))
        
        self.run_stochastic_model()
        self.calculate_summary_statistics_stochastic_runs()
        self.create_plots()
        
        
    def run_stochastic_model(self):
        
        # model_list = []
        self.nb_infected_school1 = np.zeros(shape=(self.n_sim))
        self.infectious_school_1 = np.zeros(shape=(self.n_sim, self.params['sim_duration_days'] + 1))
        self.infected_school_1 = np.zeros(shape=(self.n_sim, self.params['sim_duration_days'] + 1))
        
        seed_base = int(time.time() % 1 * 1000000)
        for i_sim in range(self.n_sim):
            self.params['RNG_seed'] = i_sim + seed_base
            
            # Create and run model
            model = MetapopulationSEPIR(self.params)
            model.simulate()
            # model.plot_results()
            # model_list.append(model)
            
            self.new_infected_pop_1 = model.R[:,0] - model.R[0,0] - params['I0'][0]
            self.nb_infected_school1[i_sim] = self.new_infected_pop_1[-1]
            
            #self.infectious_school_1[i_sim, :] = model.I[self.steps_per_day::self.steps_per_day, 0]
            #self.infected_school_1[i_sim, :] = model.I[self.steps_per_day::self.steps_per_day, 0] +\
            #    model.E[self.steps_per_day::self.steps_per_day, 0]
            self.infectious_school_1[i_sim, :] = (
                model.I[::self.steps_per_day, 0]
            )
            self.infected_school_1[i_sim, :] = (
                    model.I[::self.steps_per_day, 0] + model.E[::self.steps_per_day, 0]
            )

            self.infected_school_1_7day_ma = calculate_np_moving_average(
                self.infected_school_1, 7, shorter_window_beginning=True)
            self.model = model
        
        return
    
    
    def calculate_summary_statistics_stochastic_runs(self):
        
        unique, counts = np.unique(self.nb_infected_school1, return_counts=True)
        df_infected_1 = pd.DataFrame({
            'nb_infected': unique,
            'nb_simulation': counts
            })
        df_infected_1['probability'] = df_infected_1['nb_simulation'] / self.n_sim

        median_nb_infected = np.median(self.nb_infected_school1)
        self.index_sim_closest_median = min(
            range(len(self.nb_infected_school1)), 
            key=lambda i: abs(self.nb_infected_school1[i]-median_nb_infected)
            )
        
        self.probability_5_plus_cases = df_infected_1.loc[
            df_infected_1['nb_infected'] >= 5, 'probability'].sum()
        self.probability_10_plus_cases = df_infected_1.loc[
            df_infected_1['nb_infected'] >= 10, 'probability'].sum()
        self.probability_20_plus_cases = df_infected_1.loc[
            df_infected_1['nb_infected'] >= 20, 'probability'].sum()
        
        p_5_pct = '{:.0%}'.format(self.probability_5_plus_cases)
        p_10_pct = '{:.0%}'.format(self.probability_10_plus_cases)
        p_20_pct = '{:.0%}'.format(self.probability_20_plus_cases)
        
        self.expected_infections_all_sim = self.nb_infected_school1.mean()
        df_over_20 = df_infected_1.loc[
            df_infected_1['nb_infected'] >= 20]#, 'nb_infected']
        if len(df_over_20) > 0:
            self.expected_outbreak_size = params['I0'][0] + \
                (df_over_20['nb_infected'] * df_over_20['probability']).sum() /\
                df_over_20['probability'].sum()
            cases_over_20 = self.nb_infected_school1[
                self.nb_infected_school1 >= 20]
            quantile_list = [0, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 100]
            self.expected_outbreak_quantiles = {
                q: get_percentile_from_list(cases_over_20, q)
                for q in quantile_list
                }
            
            # self.expected_outbreak_size = df_infected_1.loc[
            #     df_infected_1['nb_infected'] >= 20, 'nb_infected'].mean()
        else:
            self.expected_outbreak_size = 'NA'
            self.expected_outbreak_size_min = 'NA'
            self.expected_outbreak_size_max = 'NA'
        self.df_infected_1 = df_infected_1
        
        if self.print_summary_stats:   
            print('Probability of 5 or more cases in outbreak:', p_5_pct)
            print('Probability of 10 or more cases in outbreak:', p_10_pct)
            print('Probability of 20 or more cases in outbreak:', p_20_pct)
            
            print('Expected number of infections across all simulations:', 
                  int(self.expected_infections_all_sim), 'cases')
            
            if self.expected_outbreak_size == 'NA':
                expected_outbreak_size_print = self.expected_outbreak_size
            else:
                expected_outbreak_size_print = int(self.expected_outbreak_size)
            print('Expected number of infections across outbreaks of size 20 or more:', 
                  expected_outbreak_size_print, 'cases')    
        
        return 
    
    def create_plots(self):
        self.df_spaghetti_infected = transform_matrix_to_long_df(
            self.infected_school_1.T,
            colnames = list(range(self.n_sim)),
            id_col = 'day',
            id_values = list(range(1,2+params['sim_duration_days'])),
            var_name = 'simulation_idx',
            value_name='number_infected'
            )
        self.df_spaghetti_infected_ma = transform_matrix_to_long_df(
            self.infected_school_1_7day_ma.T,
            colnames = list(range(self.n_sim)),
            id_col = 'day',
            id_values = list(range(1,2+params['sim_duration_days'])),
            var_name = 'simulation_idx',
            value_name='number_infected_7_day_ma'
            )
        self.df_spaghetti_infectious = transform_matrix_to_long_df(
            self.infectious_school_1.T,
            colnames = list(range(self.n_sim)),
            id_col = 'day',
            id_values = list(range(1,2+params['sim_duration_days'])),
            var_name = 'simulation_idx',
            value_name='number_infectious'
            )
        
        if self.show_plots:
            nb_plots = 3
            fig, axs = plt.subplots(nb_plots, 1, figsize=(10,6*nb_plots))
            i_plot = 0
            
            ax = axs[i_plot]
            sns.histplot(
                data=self.nb_infected_school1,
                stat='percent',
                # binwidth=5,
                ax=ax)
            ax.set_xlabel('Total infections')
            ax.set_xlabel('Probability (%)')
            
            i_plot += 1
            ax=axs[i_plot]
            sns.lineplot(
                self.df_spaghetti_infected,
                x='day',
                y='number_infected',
                units='simulation_idx',
                alpha=0.1,
                estimator=None,
                ax=ax
                )
            ax.set_xlabel('Number of days since beginning of outbreak')
            ax.set_ylabel('Number of infected individuals')
            
            i_plot += 1
            ax=axs[i_plot]
            sns.lineplot(
                self.df_spaghetti_infectious,
                x='day',
                y='number_infectious',
                units='simulation_idx',
                alpha=0.1,
                estimator=None,
                ax=ax
                )
            ax.set_xlabel('Number of days since beginning of outbreak')
            ax.set_ylabel('Number of infectious individuals')
            
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

params = {
    'R0': 15.0,        # transmission rate
    # 'sigma': 1/10.5,    # 10.5 days average latent period
    # 'rho': 1/1,         # 1 days average pre-symptomatic period
    # 'gamma': 1/8,       # 7 days average infectious period
    'incubation_period': 10.5,
    'infectious_period': 8.0,
    'school_contacts': 5.63424,
    'other_contacts': 2.2823,
    'population': [500],
    'I0': [1],
    'vaccinated_percent': [0.9], # number between 0 and 1
    'sim_duration_days': 250,
    'time_step_days': 0.25,
    'is_stochastic': True, # False for deterministic
    'RNG_seed': int(time.time()*1000), #2025,
    }
n_sim = 200

# %% Main
# Example usage
if __name__ == "__main__":
    
    run_deterministic_model(params)
    
    # Stochastic runs
    n_sim = 200
    stochastic_sim = StochasticSimulations( 
        params, n_sim, print_summary_stats=True, show_plots=True)


# %%
# if False:
#     nn=100
#     params['vaccinated_percent'] = [0.8367]
#     params['population'] = [500]
#     params['I0'] = [490]
#     s = StochasticSimulations( 
#         params, nn, print_summary_stats=True, show_plots=False)
#     s.new_infected_pop_1
import pandas as pd
import numpy as np
import copy

import measles_single_population as msp

import pytest

deterministic_params = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
deterministic_params["is_stochastic"] = False


def test_det_exceedance_monotonic_in_vax_rate():
    # Deterministically, there is no probability -- get_exceedance_probability just
    #   returns 0/1 whether there were more than X new cases for the deterministic simulation

    exceedance_prob_list = []

    for i in np.arange(100, step=10) / 100:
        deterministic_params["vax_prop"] = [i]

        experiment = msp.DashboardExperiment(deterministic_params, 1)
        exceedance_prob_list.append(experiment.total_new_cases_school_1.get_exceedance_probability(10))

    assert (np.diff(exceedance_prob_list) >= 0).all()


def test_det_new_cases_monotonic_in_vax_rate():
    """
    As vaccination rate increases, new cases should decrease
    """

    new_cases_list = []

    for i in np.arange(100, step=10) / 100:
        deterministic_params["vax_prop"] = [i]

        experiment = msp.DashboardExperiment(deterministic_params, 1)
        new_cases_list.append(experiment.total_new_cases_school_1.data[0])

    assert (np.diff(new_cases_list) <= 0).all()


def test_det_new_cases_monotonic_in_R0():
    """
    As R0 increases, new cases should increase
    """

    new_cases_list = []

    for i in np.arange(20):
        deterministic_params["R0"] = i

        experiment = msp.DashboardExperiment(deterministic_params, 1)
        new_cases_list.append(experiment.total_new_cases_school_1.data[0])

    assert (np.diff(new_cases_list) >= 0).all()


def test_det_new_cases_monotonic_in_infectious_period():
    """
    Yo! Important kind of counterintuitive note!
    For fixed value of R0, increasing the infectious period
        actually decreases the number of cases deterministically
        because it decreases the value of beta!
    In the code...
        `beta = params['R0'] / (params["infectious_period"] * self.total_contacts)`
    """

    new_cases_list = []

    for i in np.arange(1, 20):
        deterministic_params["infectious_period"] = i

        experiment = msp.DashboardExperiment(deterministic_params, 1)
        new_cases_list.append(experiment.total_new_cases_school_1.data[0])

    assert (np.diff(new_cases_list) <= 0).all()


def test_recovered_monotonic():
    """
    Will probably have to adapt this test because Recovered
        is not necessarily monotonic with leaky vaccination.
        But this works for now.
    """

    model = msp.MetapopSEIR(msp.DEFAULT_MSP_PARAMS)
    model.simulate()

    assert (np.diff(model.R.sum(axis=1)) >= 0).all()


def test_population_is_constant():
    """
    At each timepoint, the total population is the same
    """

    model = msp.MetapopSEIR(msp.DEFAULT_MSP_PARAMS)

    for i in range(10):

        model.simulate()

        sum_of_compartments = model.S + model.E + model.I + model.R

        assert (sum_of_compartments.sum(axis=1) == model.N).all()

        model.clear()


# Increasing incubation period delays peak infections?
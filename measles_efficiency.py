# This is an efficiency run (functional,
#   NOT object-oriented) for computing only
#   the number of new infections for a single
#   subpopulation -- we don't save sample paths
#   -- we only

# Note that this can be compiled in Cython
#   for an even faster speed-up -- but
#   would need to work out the details for Cython
#   and compiling on Docker / on the actual online
#   (non-local) dashboard

# %% Imports and parameters
###########################
import numpy as np
from randomgen import PCG64
from typing import TypedDict


# %% Measles Parameters
#######################

# Compare with MeaslesParameters --
#   no lists, only scalars (only 1 subpopulation allowed)
class MeaslesEfficiencyParameters(TypedDict):
    R0: float
    incubation_period: float
    infectious_period: float
    school_contacts: float
    other_contacts: float
    population: int
    I0: int
    vax_prop: float
    threshold_values: int
    sim_duration_days: int
    time_step_days: float
    simulation_seed: int


# Set parameters
# Natural history parameters
# https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html
# Incubation: 11.5 days, so first symptoms 11.5 days since t0
# Rash starts: 3 days after first symptoms, so 14.5 since t0
# Infectious: 4 days before rash (so 10.5 days since t0), 4 days after (18.5 days since t0)
DEFAULT_MSP_EFFICIENCY_PARAMS: MeaslesEfficiencyParameters = \
    {
        'R0': 15.0,  # transmission rate
        'incubation_period': 10.5,
        'infectious_period': 5,
        'school_contacts': 5.63424,
        'other_contacts': 2.2823,
        'population': 500,
        'I0': 1,
        'vax_prop': 0.85,  # number between 0 and 1
        'threshold_values': 10,  # outbreak threshold, only first value used for now
        'sim_duration_days': 250,
        'time_step_days': 0.25,
        "simulation_seed": 147125098488
    }


# %% Model
##########

def get_transition(rate: float,
                   compartment_count: int,
                   RNG: np.random.Generator) -> float:
    total_rate = rate * compartment_count
    return min(RNG.poisson(total_rate), compartment_count)


def compute_new_infections(params: MeaslesEfficiencyParameters,
                           num_reps: int,
                           use_adaptive_step_size: bool = False):

    # Unpack everything from dictionary
    dt_init = params['time_step_days']
    total_num_steps = 1 + int(params['sim_duration_days'] / dt_init)
    N = params['population']
    total_contacts = params['school_contacts'] + params['other_contacts']
    beta = params['R0'] / (params["infectious_period"] * total_contacts)

    # This is a faster and more modern RNG
    RNG = np.random.Generator(PCG64(params["simulation_seed"]))

    # Pre-compute "base rate" for transitions
    #   (these will get multiplied by dt and force_of_infection_coefficient
    #   will also get multiplied by I, which is time-dependent)
    force_of_infection_coefficient = beta * total_contacts / N
    E_out_rate_coefficient = (1.0 / params["incubation_period"])
    I_out_rate_coefficient = (1.0 / params["infectious_period"])

    # Initial values
    I_init = int(params["I0"])
    R_init = int(min(N * params['vax_prop'], N - I_init))
    S_init = int(N - I_init - R_init)

    total_S_to_E_array = np.zeros(num_reps)

    for rep in range(num_reps):

        I = I_init
        R = R_init
        S = S_init
        E = 0

        total_S_to_E = 0

        step_counter = 0

        dt = dt_init

        dE = 0
        dI = 0

        while step_counter < total_num_steps:

            if use_adaptive_step_size:
                if dE == 0 and dI == 0 and I <= 2:
                    dt = 0.5
                    step_counter += 2
                else:
                    dt = dt_init
                    step_counter += 1

            # If the Poisson rate would be 0 anyway, avoid
            #   computing the random variable to save time! Good trick
            #   from Remy
            if I == 0:
                dS_out = 0
            else:
                dS_out = get_transition(force_of_infection_coefficient * I * dt, S, RNG)

            if E == 0:
                dE_out = 0
            else:
                dE_out = get_transition(E_out_rate_coefficient * dt, E, RNG)

            if I == 0:
                dI_out = 0
            else:
                dI_out = get_transition(I_out_rate_coefficient * dt, I, RNG)

            dS = -dS_out
            dE = dS_out - dE_out
            dI = dE_out - dI_out
            dR = dI_out

            S += dS
            E += dE
            I += dI
            R += dR

            total_S_to_E += dS_out

            if S == 0 or E + I == 0:
                break

        total_S_to_E_array[rep] = total_S_to_E

    return total_S_to_E_array

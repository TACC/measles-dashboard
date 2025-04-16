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
from typing import TypedDict, Callable


# %% Measles Parameters
#######################

# Compare with MeaslesParameters --
#   no lists, only scalars (only 1 subpopulation allowed)
class MeaslesEfficiencyParameters(TypedDict):
    R0: float
    incubation_period: float
    infectious_period: float
    school_contacts: float
    incubation_period_vaccinated: float
    infectious_period_vaccinated: float
    relative_infectiousness_vaccinated: float
    vaccine_efficacy: float
    other_contacts: float
    population: int
    I0: int
    vax_prop: float
    threshold_values: int
    sim_duration_days: int
    time_step_days: float
    simulation_seed: int


DEFAULT_MSP_EFFICIENCY_PARAMS: MeaslesEfficiencyParameters = \
    {
        'R0': 15.0,  # transmission rate
        'incubation_period': 10.5,
        'infectious_period': 5,
        'incubation_period_vaccinated': 10.5,
        'infectious_period_vaccinated': 5,
        'relative_infectiousness_vaccinated': 0.05, # 0 means no infection from vaccinated
        'vaccine_efficacy': 0.997, # 1.0 means perfect protection
        'school_contacts': 5.63424,
        'other_contacts': 2.2823,
        'population': 500,
        'I0': 1,
        'vax_prop': 0.85,
        'threshold_values': 10,
        'sim_duration_days': 250,
        'time_step_days': 0.25,
        'is_stochastic': True,  # False for deterministic
        "simulation_seed": 147125098488
    }


# %% Model
##########

def build_transition_sampler(RNG: np.random.Generator):

    def transition_sampler(rate: float,
                       compartment_count: int) -> float:
        total_rate = rate * compartment_count
        if total_rate == 0:
            return 0
        else:
            return min(RNG.poisson(total_rate), compartment_count)

    return transition_sampler


def compute_new_infections(params: MeaslesEfficiencyParameters,
                           transition_sampler: Callable,
                           num_reps: int,
                           use_adaptive_step_size: bool = False):

    # Hacked this to account for lists (ouch owie) in
    #   MeaslesParameters -- will try to find a way to get a more
    #   elegant workaround later...
    if isinstance(params["population"], list):
        N = params["population"][0]
    else:
        N = params["population"]
    if isinstance(params["I0"], list):
        I_unvax_init = params["I0"][0]
    else:
        I_unvax_init = params["I0"]
    if isinstance(params["vax_prop"], list):
        vax_prop = params["vax_prop"][0]
    else:
        vax_prop = params["vax_prop"]

    # Unpack from dictionary
    dt_init = params['time_step_days']
    total_num_steps = 1 + int(params['sim_duration_days'] / dt_init)
    total_contacts = params['school_contacts'] + params['other_contacts']

    R0 = params["R0"]
    infectious_period = params["infectious_period"]
    incubation_period = params["incubation_period"]
    infectious_period_vaccinated = params["infectious_period_vaccinated"]
    incubation_period_vaccinated = params["incubation_period_vaccinated"]
    relative_infectiousness_vaccinated = \
        params["relative_infectiousness_vaccinated"]
    vaccine_efficacy = params["vaccine_efficacy"]

    beta = R0 / (infectious_period * total_contacts)
    beta_vax = relative_infectiousness_vaccinated * R0 / \
               (infectious_period_vaccinated * total_contacts)

    E_unvax_out_rate_coefficient = (1.0 / incubation_period)
    I_unvax_out_rate_coefficient = (1.0 / infectious_period)

    E_vax_out_rate_coefficient = (1.0 / incubation_period_vaccinated)
    I_vax_out_rate_coefficient = (1.0 / infectious_period_vaccinated)

    total_contacts_to_N_ratio = total_contacts / N

    # Other initial values -- I_unvax_init is above
    S_vax_init = int(N * vax_prop)
    S_vax_init = min(S_vax_init, N - I_unvax_init)
    S_unvax_init = int(N - I_unvax_init - S_vax_init)

    total_S_unvax_to_E_unvax_array = np.zeros(num_reps)
    total_S_vax_to_E_vax_array = np.zeros(num_reps)

    for rep in range(num_reps):

        I_unvax = I_unvax_init
        S_vax = S_vax_init
        S_unvax = S_unvax_init

        I_vax = E_vax = R_vax = E_unvax = R_unvax = 0

        total_S_unvax_to_E_unvax = 0
        total_S_vax_to_E_vax = 0

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

            force_of_infection_unvax = total_contacts_to_N_ratio * \
                                       (beta * I_unvax + beta_vax * I_vax)
            force_of_infection_vax = force_of_infection_unvax * \
                                     (1 - vaccine_efficacy)

            # If the Poisson rate would be 0 anyway, avoid
            #   computing the random variable to save time! Good trick
            #   from Remy

            # UNVACCINATED
            ##############
            dS_unvax_out = transition_sampler(force_of_infection_unvax * dt, S_unvax)
            dE_unvax_out = transition_sampler(E_unvax_out_rate_coefficient * dt, E_unvax)
            dI_unvax_out = transition_sampler(I_unvax_out_rate_coefficient * dt, I_unvax)

            # VACCINATED
            ############
            dS_vax_out = transition_sampler(force_of_infection_vax * dt, S_vax)
            dE_vax_out = transition_sampler(E_vax_out_rate_coefficient * dt, E_vax)
            dI_vax_out = transition_sampler(I_vax_out_rate_coefficient * dt, I_vax)

            # UNVACCINATED
            ##############
            dS_unvax = -dS_unvax_out
            dE_unvax = dS_unvax_out - dE_unvax_out
            dI_unvax = dE_unvax_out - dI_unvax_out
            dR_unvax = dI_unvax_out

            # VACCINATED
            ############
            dS_vax = -dS_vax_out
            dE_vax = dS_vax_out - dE_vax_out
            dI_vax = dE_vax_out - dI_vax_out
            dR_vax = dI_vax_out

            # UNVACCINATED
            ##############
            S_unvax += dS_unvax
            E_unvax += dE_unvax
            I_unvax += dI_unvax
            R_unvax += dR_unvax

            # VACCINATED
            ############
            S_vax += dS_vax
            E_vax += dE_vax
            I_vax += dI_vax
            R_vax += dR_vax

            total_S_vax_to_E_vax += dS_vax_out
            total_S_unvax_to_E_unvax += dS_unvax_out

            step_counter += 1

            if S_unvax + S_vax == 0 or E_unvax + E_vax + I_unvax + I_vax == 0:
                break

        total_S_unvax_to_E_unvax_array[rep] = total_S_unvax_to_E_unvax
        total_S_vax_to_E_vax_array[rep] = total_S_vax_to_E_vax

    return total_S_unvax_to_E_unvax_array, total_S_vax_to_E_vax_array

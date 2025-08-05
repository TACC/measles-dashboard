# sensitivity_analysis.py

from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd
import numpy as np
import copy
import dash_bootstrap_components as dbc
import measles_single_population as msp
import measles_efficiency
import subprocess

from randomgen import PCG64

from app_computation_functions import dashboard_exceedance_prob_str, dashboard_percentiles_str
from app_selectors import SELECTOR_DEFAULTS

DASHBOARD_CONFIG = {
    'num_simulations_graph': 20,
    'num_simulations_results': 1000,
    'simulation_seed': 147125098488,
    'spaghetti_curve_selection_seed': 12345,
}

msp.DEFAULT_MSP_PARAMS["simulation_seed"] = DASHBOARD_CONFIG["simulation_seed"]

def sensitivity_analysis_layout():
    return dbc.Container([
        html.H1("Sensitivity Analysis", className="text-center mb-4"),
        html.P("Compare different scenarios side by side", className="text-center mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scenario 1"),
                    dbc.CardBody([
                        html.Label("School Size:"),
                        dbc.Input(id="scenario1_school_size", type="number", value=500, className="mb-2"),
                        
                        html.Label("Vaccination Rate (%):"),
                        dbc.Input(id="scenario1_vax", type="number", value=85, className="mb-2"),
                        
                        html.Label("Initially Infected:"),
                        dbc.Input(id="scenario1_infected", type="number", value=1, className="mb-3"),
                        
                        html.Hr(),
                        html.H5("Results:"),
                        html.H6("Chance of Exceeding 10 New Infections:"),
                        html.Div(id="scenario1_prob_result", className="mb-2"),
                        html.H6("Likely of outbreak size if exceeds 10 new infections:"),
                        html.Div(id="scenario1_unvax_result", className="mb-2"),
                        html.Div(id="scenario1_vax_result", className="mb-2")
                    ])
                ])
            ], md=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Scenario 2"),
                    dbc.CardBody([
                        html.Label("School Size:"),
                        dbc.Input(id="scenario2_school_size", type="number", value=500, className="mb-2"),
                        
                        html.Label("Vaccination Rate (%):"),
                        dbc.Input(id="scenario2_vax", type="number", value=75, className="mb-2"),
                        
                        html.Label("Initially Infected:"),
                        dbc.Input(id="scenario2_infected", type="number", value=1, className="mb-3"),
                        
                        html.Hr(),
                        html.H5("Results:"),
                        html.H6("Chance of Exceeding 10 New Infections:"),
                        html.Div(id="scenario2_prob_result", className="mb-2"),
                        html.H6("Likely of outbreak size if exceeds 10 new infections:"),
                        html.Div(id="scenario2_unvax_result", className="mb-2"),
                        html.Div(id="scenario2_vax_result", className="mb-2")
                    ])
                ])
            ], md=6)
        ])
        
    ], style={"paddingTop": "40px"})

def create_params_from_inputs(school_size, vax_rate_percent, I0):
    params_dict = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    
    school_size = school_size if school_size is not None else SELECTOR_DEFAULTS['school_size']
    vax_rate_percent = vax_rate_percent if vax_rate_percent is not None else SELECTOR_DEFAULTS['vax_rate']
    I0 = I0 if I0 is not None else SELECTOR_DEFAULTS['I0']

    params_dict['population'] = [int(school_size)]
    params_dict['vax_prop'] = [0.01 * float(vax_rate_percent)]
    params_dict['I0'] = [int(I0)]
    params_dict['R0'] = SELECTOR_DEFAULTS.get('R0', 12.0)
    params_dict['incubation_period'] = SELECTOR_DEFAULTS.get('latent_period', 8.0)
    params_dict['infectious_period'] = SELECTOR_DEFAULTS.get('infectious_period', 7.0)
    params_dict['threshold_values'] = [10] 
    params_dict['vaccine_efficacy'] = SELECTOR_DEFAULTS.get('vaccine_efficacy_selector', 97) * 0.01
    params_dict['relative_infectiousness_vaccinated'] = 1 - SELECTOR_DEFAULTS.get('vaccinated_infectiousness_selector', 50) * 0.01


    return params_dict


def check_inputs_validity(params_dict):
    if not 0 <= params_dict["vax_prop"][0] <= 1:
        return False, "Invalid inputs: vaccination rate must be between 0-100%."
    elif params_dict["I0"][0] < 0 or params_dict["population"][0] < 0:
        return False, "Invalid inputs: school enrollment and students initially infected must be positive whole numbers. Please adjust."
    elif params_dict["I0"][0] > int((1 - params_dict["vax_prop"][0]) * params_dict["population"][0]):
        return False, "Invalid inputs: The number of initially infected students cannot exceed the number of unvaccinated students. Please adjust."
    else:
        return True, ""

def calculate_scenario_results(params_dict):    
    inputs_valid, warning_msg = check_inputs_validity(params_dict)
    if not inputs_valid:
        return warning_msg, "", ""
    
    I_unvax_init = params_dict["I0"][0]
    threshold_val = int(params_dict['threshold_values'][0])

    transition_sampler = measles_efficiency.build_transition_sampler(
        np.random.Generator(PCG64(seed=DASHBOARD_CONFIG["simulation_seed"])))

    unvax_cases_array, vax_cases_array = \
        measles_efficiency.compute_new_infections(params_dict,
                                                  transition_sampler,
                                                  DASHBOARD_CONFIG["num_simulations_results"])

    total_cases_array = unvax_cases_array + vax_cases_array
    total_cases_above_threshold_array = total_cases_array[total_cases_array >= threshold_val]

    prob_threshold_plus_new_str = \
        dashboard_exceedance_prob_str(len(total_cases_above_threshold_array) / DASHBOARD_CONFIG["num_simulations_results"])

    if len(total_cases_above_threshold_array) == 0:
        cases_expected_over_threshold_unvaccinated_str = "Fewer than {} new infections".format(int(threshold_val))
        cases_expected_over_threshold_breakthrough_str = ""
    else:
        unvax_cases_lb, unvax_cases_ub = np.percentile(unvax_cases_array[total_cases_array >= threshold_val], [2.5, 97.5])
        vax_cases_lb, vax_cases_ub = np.percentile(vax_cases_array[total_cases_array >= threshold_val], [2.5, 97.5])
        
        cases_expected_over_threshold_unvaccinated_str = \
            'Unvaccinated cases: ' + dashboard_percentiles_str(I_unvax_init, unvax_cases_lb, unvax_cases_ub)
        cases_expected_over_threshold_breakthrough_str = \
            'Vaccinated cases: ' + dashboard_percentiles_str(0, vax_cases_lb, vax_cases_ub)

    return prob_threshold_plus_new_str, cases_expected_over_threshold_unvaccinated_str, cases_expected_over_threshold_breakthrough_str
# Callback for Scenerio 1
@callback(
    [Output('scenario1_prob_result', 'children'),
     Output('scenario1_unvax_result', 'children'),
     Output('scenario1_vax_result', 'children')],
    [Input('scenario1_school_size', 'value'),
     Input('scenario1_vax', 'value'),
     Input('scenario1_infected', 'value')]
)
def update_scenario1(school_size, vax_rate_percent, I0):
    try:
        params_dict = create_params_from_inputs(school_size, vax_rate_percent, I0)
        
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        
        return prob_result, unvax_result, vax_result
        
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""

# Callback for Scenario 2
@callback(
    [Output('scenario2_prob_result', 'children'),
     Output('scenario2_unvax_result', 'children'),
     Output('scenario2_vax_result', 'children')],
    [Input('scenario2_school_size', 'value'),
     Input('scenario2_vax', 'value'),
     Input('scenario2_infected', 'value')]
)
def update_scenario2(school_size, vax_rate_percent, I0):
    try:
        params_dict = create_params_from_inputs(school_size, vax_rate_percent, I0)
        
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        
        return prob_result, unvax_result, vax_result
        
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""
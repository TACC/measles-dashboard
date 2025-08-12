# sensitivity_analysis.py
from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
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
        dcc.Store(id="show-scenario-3", data=False),
        dcc.Store(id="show-scenario-4", data=False),
        dcc.Store(id="show-scenario-5", data=False),
        
        dbc.Row([
            dbc.Col([
                html.H1("Sensitivity Analysis", className="text-center mb-2"),
                html.P("Compare different scenarios side by side", className="text-center mb-4"),
            ], md=10),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-plus me-2"), "Add Scenario"],
                    id="add-scenario-btn",
                    color="primary",
                    size="sm",
                    className="float-end mt-2"
                )
            ], md=2)
        ]),
        
        html.Div([
            dbc.Row(id="scenarios-row", children=[
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Scenario 1"),
                        dbc.CardBody([
                            html.Label("School Size:"),
                            dbc.Input(id="scenario1_school_size", type="number", value=500, className="mb-2"),
                            
                            html.Label("Vaccination Rate (%):"),
                            dbc.Input(id="scenario1_vax", type="number", value=85, className="mb-2"),
                            
                            html.Label("Initially Infected:"),
                            dbc.Input(id="scenario1_infected", type="number", value=1, className="mb-2"),

                            html.Label("Basic Reproduction Number (R0):"),
                            dcc.Slider(id="scenario1_R0_slider", min=12, max=18, step=0.1, value=15.0,
                                          marks={12: '12', 15: '15', 18: '18'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Average Latent Period (days):"),
                            dcc.Slider(id="scenario1_latent_slider", min=7, max=12, step=0.1, value=10.5,
                                          marks={7: '7', 10.5: '10.5', 12: '12'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Average Infectious Period (days):"),
                            dcc.Slider(id="scenario1_infectious_slider", min=4, max=9, step=0.1, value=5.0,
                                          marks={4: '4', 5: '5', 9: '9'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Minimum Outbreak Size (New Infections):"),
                            dcc.Slider(id="scenario1_threshold_slider", min=3, max=25, step=1, value=10,
                                        marks={3: '3', 10: '10', 25: '25'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Label("Vaccine Efficacy - Susceptibility (%):"),
                            dcc.Slider(id="scenario1_vaccine_susceptibility_slider", min=99, max=100, step=0.1, value=99.7,
                                        marks={99: '99', 99.7: '99.7', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Label("Vaccine Efficacy - Infectiousness (%):"),
                            dcc.Slider(id="scenario1_vaccine_infectiousness_slider", min=75, max=100, step=1, value=95,
                                        marks={75: '75', 95: '95', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Hr(),
                            html.H5("Results:"),
                            html.H6("Chance of Exceeding 10 New Infections:"),
                            html.Div(id="scenario1_prob_result", className="mb-2"),
                            html.H6("Likely outbreak size if exceeds 10 new infections:"),
                            html.Div(id="scenario1_unvax_result", className="mb-2"),
                            html.Div(id="scenario1_vax_result", className="mb-2")
                        ])
                    ])
                ], md=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Scenario 2"),
                        dbc.CardBody([
                            html.Label("School Size:"),
                            dbc.Input(id="scenario2_school_size", type="number", value=500, className="mb-2"),
                            
                            html.Label("Vaccination Rate (%):"),
                            dbc.Input(id="scenario2_vax", type="number", value=75, className="mb-2"),
                            
                            html.Label("Initially Infected:"),
                            dbc.Input(id="scenario2_infected", type="number", value=1, className="mb-2"),

                            html.Label("Basic Reproduction Number (R0):"),
                            dcc.Slider(id="scenario2_R0_slider", min=12, max=18, step=0.1, value=15.0,
                                          marks={12: '12', 15: '15', 18: '18'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Average Latent Period (days):"),
                            dcc.Slider(id="scenario2_latent_slider", min=7, max=12, step=0.1, value=10.5,
                                          marks={7: '7', 10.5: '10.5', 12: '12'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Average Infectious Period (days):"),
                            dcc.Slider(id="scenario2_infectious_slider", min=4, max=9, step=0.1, value=5.0,
                                          marks={4: '4', 5: '5', 9: '9'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Minimum Outbreak Size (New Infections):"),
                            dcc.Slider(id="scenario2_threshold_slider", min=3, max=25, step=1, value=10,
                                        marks={3: '3', 10: '10', 25: '25'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Label("Vaccine Efficacy - Susceptibility (%):"),
                            dcc.Slider(id="scenario2_vaccine_susceptibility_slider", min=99, max=100, step=0.1, value=99.7,
                                        marks={99: '99', 99.7: '99.7', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Label("Vaccine Efficacy - Infectiousness (%):"),
                            dcc.Slider(id="scenario2_vaccine_infectiousness_slider", min=75, max=100, step=1, value=95,
                                        marks={75: '75', 95: '95', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Hr(),
                            html.H5("Results:"),
                            html.H6("Chance of Exceeding 10 New Infections:"),
                            html.Div(id="scenario2_prob_result", className="mb-2"),
                            html.H6("Likely outbreak size if exceeds 10 new infections:"),
                            html.Div(id="scenario2_unvax_result", className="mb-2"),
                            html.Div(id="scenario2_vax_result", className="mb-2")
                        ])
                    ])
                ], md=4),
            ]),
            
            # Additional scenerios
            dbc.Row(id="additional-scenarios", children=[
                dbc.Col(id="scenario3-col", children=[
                    dbc.Card([
                        dbc.CardHeader([
                            html.Div([
                                html.H5("Scenario 3", className="mb-0"),
                                dbc.Button("×", id="remove-scenario-3", size="sm", color="link", 
                                         style={"color": "#dc3545", "textDecoration": "none", "fontSize": "18px"})
                            ], className="d-flex justify-content-between align-items-center")
                        ]),
                        dbc.CardBody([
                            html.Label("School Size:"),
                            dbc.Input(id="scenario3_school_size", type="number", value=500, className="mb-2"),
                            
                            html.Label("Vaccination Rate (%):"),
                            dbc.Input(id="scenario3_vax", type="number", value=90, className="mb-2"),
                            
                            html.Label("Initially Infected:"),
                            dbc.Input(id="scenario3_infected", type="number", value=1, className="mb-2"),

                            html.Label("Basic Reproduction Number (R0):"),
                            dcc.Slider(id="scenario3_R0_slider", min=12, max=18, step=0.1, value=15.0,
                                          marks={12: '12', 15: '15', 18: '18'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Average Latent Period (days):"),
                            dcc.Slider(id="scenario3_latent_slider", min=7, max=12, step=0.1, value=10.5,
                                          marks={7: '7', 10.5: '10.5', 12: '12'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Average Infectious Period (days):"),
                            dcc.Slider(id="scenario3_infectious_slider", min=4, max=9, step=0.1, value=5.0,
                                          marks={4: '4', 5: '5', 9: '9'},
                                          tooltip={"placement": "bottom", "always_visible": True},
                                          className="mb-3"),

                            html.Label("Minimum Outbreak Size (New Infections):"),
                            dcc.Slider(id="scenario3_threshold_slider", min=3, max=25, step=1, value=10,
                                        marks={3: '3', 10: '10', 25: '25'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Label("Vaccine Efficacy - Susceptibility (%):"),
                            dcc.Slider(id="scenario3_vaccine_susceptibility_slider", min=99, max=100, step=0.1, value=99.7,
                                        marks={99: '99', 99.7: '99.7', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Label("Vaccine Efficacy - Infectiousness (%):"),
                            dcc.Slider(id="scenario3_vaccine_infectiousness_slider", min=75, max=100, step=1, value=95,
                                        marks={75: '75', 95: '95', 100: '100'},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                        className="mb-3"),

                            html.Hr(),
                            html.H5("Results:"),
                            html.H6("Chance of Exceeding 10 New Infections:"),
                            html.Div(id="scenario3_prob_result", className="mb-2"),
                            html.H6("Likely outbreak size if exceeds 10 new infections:"),
                            html.Div(id="scenario3_unvax_result", className="mb-2"),
                            html.Div(id="scenario3_vax_result", className="mb-2")
                        ])
                    ])
                ], md=4, style={"display": "none"})
            ])
        ])
    ], style={"paddingTop": "40px"})

def create_params_from_inputs(school_size, vax_rate_percent, I0, R0=None, latent_period=None, 
                             infectious_period=None, threshold=None, vaccine_susceptibility=None, 
                             vaccine_infectiousness=None):
    params_dict = copy.deepcopy(msp.DEFAULT_MSP_PARAMS)
    
    school_size = school_size if school_size is not None else SELECTOR_DEFAULTS['school_size']
    vax_rate_percent = vax_rate_percent if vax_rate_percent is not None else SELECTOR_DEFAULTS['vax_rate']
    I0 = I0 if I0 is not None else SELECTOR_DEFAULTS['I0']
    R0 = R0 if R0 is not None else SELECTOR_DEFAULTS.get('R0', 15.0)
    latent_period = latent_period if latent_period is not None else SELECTOR_DEFAULTS.get('latent_period', 10.5)
    infectious_period = infectious_period if infectious_period is not None else SELECTOR_DEFAULTS.get('infectious_period', 5.0)
    threshold = threshold if threshold is not None else 10
    vaccine_susceptibility = vaccine_susceptibility if vaccine_susceptibility is not None else 99.7
    vaccine_infectiousness = vaccine_infectiousness if vaccine_infectiousness is not None else 95

    params_dict['population'] = [int(school_size)]
    params_dict['vax_prop'] = [0.01 * float(vax_rate_percent)]
    params_dict['I0'] = [int(I0)]
    params_dict['R0'] = float(R0)
    params_dict['incubation_period'] = float(latent_period)
    params_dict['infectious_period'] = float(infectious_period)
    params_dict['threshold_values'] = [int(threshold)]
    params_dict['vaccine_efficacy'] = float(vaccine_susceptibility * 0.01)
    params_dict['relative_infectiousness_vaccinated'] = float(1 - vaccine_infectiousness * 0.01)

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

@callback(
    [Output("scenario3-col", "style"),
     Output("add-scenario-btn", "children"),
     Output("add-scenario-btn", "disabled")],
    [Input("add-scenario-btn", "n_clicks"),
     Input("remove-scenario-3", "n_clicks")],
    [State("scenario3-col", "style")],
    prevent_initial_call=True
)
def toggle_scenario3(add_clicks, remove_clicks, current_style):
    ctx_triggered = callback_context.triggered[0]['prop_id'] if callback_context.triggered else None
    
    if ctx_triggered == "add-scenario-btn.n_clicks":
        return {"display": "block"}, [html.I(className="fas fa-check me-2"), "Scenario Added"], True
    elif ctx_triggered == "remove-scenario-3.n_clicks":
        return {"display": "none"}, [html.I(className="fas fa-plus me-2"), "Add Scenario"], False
    
    return current_style, [html.I(className="fas fa-plus me-2"), "Add Scenario"], False

@callback(
    [Output('scenario1_prob_result', 'children'),
     Output('scenario1_unvax_result', 'children'),
     Output('scenario1_vax_result', 'children')],
    [Input('scenario1_school_size', 'value'),
     Input('scenario1_vax', 'value'),
     Input('scenario1_infected', 'value'),
     Input('scenario1_R0_slider', 'value'),
     Input('scenario1_latent_slider', 'value'),
     Input('scenario1_infectious_slider', 'value'),
     Input('scenario1_threshold_slider', 'value'),
     Input('scenario1_vaccine_susceptibility_slider', 'value'),
     Input('scenario1_vaccine_infectiousness_slider', 'value')]
)
def update_scenario1(school_size, vax_rate_percent, I0, R0, latent_period, 
                    infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness):
    try:
        params_dict = create_params_from_inputs(
            school_size, vax_rate_percent, I0, R0, latent_period, 
            infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness
        )
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        return prob_result, unvax_result, vax_result
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""

@callback(
    [Output('scenario2_prob_result', 'children'),
     Output('scenario2_unvax_result', 'children'),
     Output('scenario2_vax_result', 'children')],
    [Input('scenario2_school_size', 'value'),
     Input('scenario2_vax', 'value'),
     Input('scenario2_infected', 'value'),
     Input('scenario2_R0_slider', 'value'),
     Input('scenario2_latent_slider', 'value'),
     Input('scenario2_infectious_slider', 'value'),
     Input('scenario2_threshold_slider', 'value'),
     Input('scenario2_vaccine_susceptibility_slider', 'value'),
     Input('scenario2_vaccine_infectiousness_slider', 'value')]
)
def update_scenario2(school_size, vax_rate_percent, I0, R0, latent_period, 
                    infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness):
    try:
        params_dict = create_params_from_inputs(
            school_size, vax_rate_percent, I0, R0, latent_period, 
            infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness
        )
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        return prob_result, unvax_result, vax_result
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""

@callback(
    [Output('scenario3_prob_result', 'children'),
     Output('scenario3_unvax_result', 'children'),
     Output('scenario3_vax_result', 'children')],
    [Input('scenario3_school_size', 'value'),
     Input('scenario3_vax', 'value'),
     Input('scenario3_infected', 'value'),
     Input('scenario3_R0_slider', 'value'),
     Input('scenario3_latent_slider', 'value'),
     Input('scenario3_infectious_slider', 'value'),
     Input('scenario3_threshold_slider', 'value'),
     Input('scenario3_vaccine_susceptibility_slider', 'value'),
     Input('scenario3_vaccine_infectiousness_slider', 'value')]
)
def update_scenario3(school_size, vax_rate_percent, I0, R0, latent_period, 
                    infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness):
    try:
        params_dict = create_params_from_inputs(
            school_size, vax_rate_percent, I0, R0, latent_period, 
            infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness
        )
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        return prob_result, unvax_result, vax_result
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""

@callback(
    [Output('scenario4_prob_result', 'children'),
     Output('scenario4_unvax_result', 'children'),
     Output('scenario4_vax_result', 'children')],
    [Input('scenario4_school_size', 'value'),
     Input('scenario4_vax', 'value'),
     Input('scenario4_infected', 'value'),
     Input('scenario4_R0_slider', 'value'),
     Input('scenario4_latent_slider', 'value'),
     Input('scenario4_infectious_slider', 'value'),
     Input('scenario4_threshold_slider', 'value'),
     Input('scenario4_vaccine_susceptibility_slider', 'value'),
     Input('scenario4_vaccine_infectiousness_slider', 'value')]
)
def update_scenario4(school_size, vax_rate_percent, I0, R0, latent_period, 
                    infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness):
    try:
        params_dict = create_params_from_inputs(
            school_size, vax_rate_percent, I0, R0, latent_period, 
            infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness
        )
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        return prob_result, unvax_result, vax_result
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""

@callback(
    [Output('scenario5_prob_result', 'children'),
     Output('scenario5_unvax_result', 'children'),
     Output('scenario5_vax_result', 'children')],
    [Input('scenario5_school_size', 'value'),
     Input('scenario5_vax', 'value'),
     Input('scenario5_infected', 'value'),
     Input('scenario5_R0_slider', 'value'),
     Input('scenario5_latent_slider', 'value'),
     Input('scenario5_infectious_slider', 'value'),
     Input('scenario5_threshold_slider', 'value'),
     Input('scenario5_vaccine_susceptibility_slider', 'value'),
     Input('scenario5_vaccine_infectiousness_slider', 'value')]
)
def update_scenario5(school_size, vax_rate_percent, I0, R0, latent_period, 
                    infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness):
    try:
        params_dict = create_params_from_inputs(
            school_size, vax_rate_percent, I0, R0, latent_period, 
            infectious_period, threshold, vaccine_susceptibility, vaccine_infectiousness
        )
        prob_result, unvax_result, vax_result = calculate_scenario_results(params_dict)
        return prob_result, unvax_result, vax_result
    except Exception as e:
        return f"Calculation error: {str(e)}", "", ""
    

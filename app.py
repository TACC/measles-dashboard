#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:39:51 2025
Updated on Thu Feb 20 1:55:00 2025

@author: rfp437
"""

from dash import Dash, html, dcc, callback, Output, Input#, State, Patch
import plotly.express as px
import pandas as pd
import numpy as np
import copy
import dash_bootstrap_components as dbc
import measles_single_population as msp

df = pd.read_csv('TX_MMR_vax_rate.csv')
df = df.loc[df['age_group'] == 'Kindergarten'].copy()

initial_county = 'Travis'
county_dropdown = html.Div(
    [
        dbc.Label("Select Texas county", html_for="county_dropdown"), 
        dcc.Dropdown(
            id="county-dropdown",
            options=sorted(df["County"].unique()),
            value=initial_county,
            clearable=False,
            maxHeight=600,
            optionHeight=50
        ),
    ],  className="mb-4",
    style={'fontFamily':'Sans-serif', 'font-size':'16pt'}
)

# df there should depend on the selected county
df_county = df.loc[df['County'] == initial_county]
school_options = sorted(df_county["School District or Name"].unique())
initial_school = 'AUSTIN ISD'

if initial_school not in school_options:
    initial_school = school_options[0]

# Instructions Section
instructions_section = dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                html.P(
                    "The graph below shows 20 equally plausible outbreak trajectories, assuming no intervention. Use the interactive boxes to change the number of students in the school, the number already infected at the start of the outbreak, the percent vaccinated against measles, and key epidemiological quantities. ",
                    className="text-dark",  style={"text-align": "left", "font-size": "25px", "margin-bottom": "10px"}
                )
            ),
            className="mb-3", style={'border': 'none', "text-align": "center"}
        ),
        width=12
    ),
    style={"text-align": "center"}
)

school_dropdown = html.Div(
    [
        dbc.Label("Select a school district", html_for="school_dropdown", style={'fontFamily':'Sans-serif', 'font-size':'16pt'}),
        dcc.Dropdown(
            id="school-dropdown",
            options=school_options,
            value=initial_school,
            clearable=False,
            maxHeight=600,
            optionHeight=50,
            style={"whiteSpace": "nowrap", "width": "100%"},
        ),
    ],  className="mb-4",
    style={'fontFamily':'Sans-serif', 'font-size':'16pt', 'whiteSpace': 'nowrap', 'overflow':'visible'}
)

vaccination_rate_label = html.H4(
    'Vaccination rate (%)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5, 'fontFamily':'Sans-serif', 'font-size':'16pt'})
vaccination_rate_selector = dcc.Input(
            id='vax_rate',
            type='number',
            placeholder='Vaccination rate (%)',
            value=85,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'14pt'}
        )

I0_label = html.H4(
    'Number of students initially infected',
    style={'display':'inline-block','margin-right':5, 'margin-left':5, 'fontFamily':'Sans-serif', 'font-size':'16pt'})
I0_selector = dcc.Input(
            id='I0',
            type='number',
            placeholder='Number of students initially infected',
            value=1.0,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'14pt'}
        )

school_size_label = html.H4(
    'School enrollment (number of students)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})
school_size_selector = dcc.Input(
            id='school_size',
            type='number',
            placeholder='School enrollment (number of students)',
            value=500,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'14pt'}
        )

R0_label = html.H4(
    'Reproduction number (R0)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})
R0_selector = dcc.Input(
            id='R0',
            type='number',
            placeholder='Reproductive number',
            value=15.0,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'14pt'}
        )

latent_period_label = html.H4(
    'Latent period (days)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})
latent_period_selector = dcc.Input(
            id='latent_period',
            type='number',
            placeholder='Latent period (days)',
            value=10.5,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'14pt'}
        )

infectious_period_label = html.H4(
    'Infectious period (days)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})
infectious_period_selector = dcc.Input(
            id='infectious_period',
            type='number',
            placeholder='Infectious period (days)',
            value=8.0,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'14pt'}
        )

app = Dash(
    prevent_initial_callbacks = 'initial_duplicate')

app.title = "epiENGAGE Measles Outbreak Simulator"

# Navbar component
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.Img(
                src="/assets/epiengage_logo_orange.png",  # Place the image in the "assets" folder
                height="40",
                className="header-logo",
                style={"marginRight": "10px"},
            ),
            html.Div("epiENGAGE Measles Outbreak Simulator", style={"color": "white", "fontSize": "24px", "fontWeight": "bold", "textAlign": "right"}),
        ],
        fluid=True,
    ),
    color="#102c41",
    dark=True,
    fixed="top"
)

# Footer component 
footer = dbc.Container(
    html.Div(
        "© 2025 Texas Advanced Computing Center, The University of Texas at Austin, Office of the Vice President for Research.",
        style={
            "textAlign": "center",
            "padding": "10px",
            "backgroundColor": "#282424",
            "color": "white",
            "position": "absolute",
            "bottom": "0",
            "width": "100%"
        },
        id="footer"
    ),
    fluid=True
)


# Define the accordion separately
accordion_vax = dbc.Accordion(
        [
            dbc.AccordionItem(
                dbc.Col(
                    [
                        dbc.Col(html.Div(county_dropdown),className="mb-2"),
                        dbc.Col(html.Div(school_dropdown),className="mb-2"),
                    ]
                ),
                title="Texas School Districts ▾ ", 
            ),
        ],
        flush=True,
        always_open=False,  # Ensures sections can be toggled independently
        active_item=[],  # Empty list means all sections are closed by default
    )



# Define the accordion separately
accordion = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                dbc.Col(
                    [
                        dbc.Col(html.Div(R0_label),className="mb-2"),
                        dbc.Col(html.Div(R0_selector),className="mb-2"),
                        dbc.Col(html.Div(latent_period_label),className="mb-2"),
                        dbc.Col(html.Div(latent_period_selector),className="mb-2"),
                        dbc.Col(html.Div(infectious_period_label),className="mb-2"),
                        dbc.Col(html.Div(infectious_period_selector),className="mb-2")
                    ]
                ),
                title="Additional Parameters ▾",
            ),
        ],
        flush=True,
        always_open=False,  # Ensures sections can be toggled independently
        active_item=[],  # Empty list means all sections are closed by default
    )
)


app.layout = dbc.Container(
    [
    dbc.Row([navbar], className="my-2"),
    html.Br(),
    dbc.Row([instructions_section], className="my-4"),

    # dbc.Col(accordion, width=6),


    # Main Layout with Left and Right Sections
    dbc.Row([
        # Left section
        dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    dbc.Col(html.Div(school_size_label),className="mb-2"),
                                ),

                                dbc.Row(
                                    dbc.Col(html.Div(school_size_selector),className="mb-2"),
                                ),

                                dbc.Row(
                                    dbc.Col(html.Div(I0_label),className="mb-2"),
                                ),

                                dbc.Row(
                                    dbc.Col(html.Div(I0_selector),className="mb-2"),
                                ),

                                dbc.Row([
                                    dbc.Col(html.Div(vaccination_rate_label)),
                            ]),

                            dbc.Row(
                                dbc.Col(html.I("Enter value or select school from dropdown"),className="m-0"),
                            ),
                                dbc.Row([
                                    dbc.Col(html.Div(vaccination_rate_selector), style={"font-size": "16pt", "margin-top": "1.5em"}),
                                    dbc.Col(html.Div("OR"), style={"font-size": "16pt", "margin-top": "1.5em", "textAlign": "center"}),
                                    dbc.Col(accordion_vax, className="mb-2 mt-2"),
                                 ]),
                                
                                dbc.Row(
                                    dbc.Col(accordion,className="mb-2"),
                                )
                            ]
                        ),
                        style={'border': 'none'}
                    ),
                    width=3, 
                    style={"border-right": "2px solid black", "padding": "10px"}, 
        ),

        # Right section 
        dbc.Col([
        # Outcomes section
         html.H3("Key Outbreak Statistics Without Intervention", style={"text-align": "center", "margin-top": "0.5em", "margin-bottom": "1.8em", "font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "20pt", "font-weight":"500"}),
          dbc.Row(
            [
                # First Box
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        dcc.Markdown(id='effective', 
                                                    children='Effective reproduction number at outset', 
                                                    style={'color': '#black', 'fontWeight': '500'}
                                        ),
                                        html.Br(),
                                        dcc.Markdown(id='effective_reproduction_number', 
                                                    style={'color': '#bf5700', 'fontWeight': '800', 'margin-top': '0.5em'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '16pt',
                                        'border': 'none'
                                    }
                                )
                            ]
                        ),
                        style={'border': 'none'}
                    ),
                    width=4,
                ),
                
                # Second Box
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                             [
                                html.Div(
                                    [
                                        dcc.Markdown(id='outbreak', 
                                                    children='Chance of over 20 new infections', 
                                                    style={'color': '#black', 'fontWeight': '500'}
                                        ),
                                        html.Br(),
                                        dcc.Markdown(id='p_20_pct', 
                                                    style={'color': '#bf5700', 'fontWeight': '800','margin-top': '0.5em'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '16pt',
                                        'border': 'none'
                                    }
                                )
                            ]
                        ),
                        style={'border':'none'}  
                    ),
                    width=4,
                    style={'borderLeft':'3px solid #bf5700'}  # Vertical line on the right
                ),

                # Third Box
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        dcc.Markdown(id='cases', 
                                                    children='Mean outbreak size \\[95% percentile interval\\]', 
                                                    style={'color': '#black', 'fontWeight': '500'}
                                        ),
                                        dcc.Markdown("*Assumes outbreak exceeds 20 cases*"),
                                        dcc.Markdown(id='cases_expected_over_20', 
                                                    style={'color': '#bf5700', 'fontWeight': '800'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '16pt',
                                        'border': 'none'
                                    }
                                )
                            ]
                        ),
                        style={'border':'none'}
                    ),
                    width=4,
                    style={'borderLeft': '3px solid #bf5700'}  
                ),
            ],
        ),

            
            # Bottom Component in the Right Section
            dbc.Row([
                 dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="spaghetti_plot")
                        ),
                        style={'border': 'none'}
                    ),
            width=12
        )
            ])
        ], width={"size": 8, "order": "last"})
        
    ]),  # Adds spacing

    html.Div([
        # html.A("Notes: ", style={"fontWeight": "bold", "fontSize": "16px"}),
        html.A("MODEL: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A(["This dashboard uses a simple stochastic compartmental susceptible-exposed-infectious-removed (SEIR) model. The default parameters include a basic reproduction number (", html.I([html.A(["R", html.Sub("0")])])," ) of 15 ["]),
        html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("], an average latent period of 10.5 days ["),
        html.A("CDC’s Measles Clinical Diagnosis Fact Sheet", href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("], and an average infectious period of 8 days ["), 
        html.A("CDC’s Measles Clinical Diagnosis Fact Sheet", href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}), 
        html.A("]."), 
        html.Ul("", style={"margin-bottom": "1em"}),
        html.A("OUTCOME STATISTICS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("The outcome statistics are estimated from 200 stochastic simulations as follows."),
        html.Ul([
            html.Li([html.I(["Effective reproduction number at the start of the outbreak (", html.A(["R", html.Sub("eff")]),]), " ) – The product of ", html.I([html.A(["R", html.Sub("0")])]), " and the proportion of students who are unvaccinated."]),
            html.Li([html.I("Chance of over 20 new infections"), html.A([" – The proportion of the 200 simulations that produced at least 20 infections, not counting the initial infections. This assumes no intervention."])]),
            html.Li([html.I("Outbreak size"), " – Among the simulations that produced at least 20 additional infections, the mean and 95% percentile interval in total number of infections. These values include the initial infections.", html.Br(style={"margin": "0", "padding": "0"})]),
        ], style={"margin-bottom": "1em"}),
       
        html.A("PROJECTIONS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("The 20 curves in the graph correspond to 20 independent simulations selected at random from 200 stochastic simulations. The y-axis values are seven-day moving averages of the total number of people infected (both exposed and infectious cases). The highlighted curve corresponds to the simulation that produced a total attack rate closest to the median across the 100 simulations."),
        html.Ul("", style={"margin-bottom": "1em"}),
        html.A("VACCINE COVERAGE: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("School vaccine coverage estimates were obtained from the Texas Department of Health and Human Services ["),
        html.A("Annual Report of Immunization status", href="https://www.dshs.texas.gov/immunizations/data/school/coverage", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("].")
    ],
    style={
        "backgroundColor": "#eaebec",  # Gray background
        "color": "black",  # White text color
        "padding": "10px",
        "textAlign": "left",
        "marginBottom": "10px",
        "fontSize": "18px"
    }),
    

   dbc.Row([
        dbc.Col(html.Div([
            html.P("©2025 ", style={"display": "inline", "font-size": "11px", "color": "#ffffff"}),
            html.A("Texas Advanced Computing Center", href="https://www.tacc.utexas.edu/", target="_blank",
                   style={"color": "#ffffff", "text-decoration": "none", "font-size": "11px"}),
            html.Br(),
            html.A("The University of Texas at Austin, Office of the Vice President for Research", href="https://research.utexas.edu", target="_blank",
                   style={"color": "#ffffff", "text-decoration": "none", "font-size":"11px"}),
        ], style={"textAlign": "center", "padding": "10px"}), width=12)
    ], style={"backgroundColor": "#292929", "marginTop": "auto"})  
], fluid=True, style={"min-height": "100vh", "display": "flex", "flex-direction": "column"})


@callback(
    [Output('spaghetti_plot', 'figure'),
     Output('effective_reproduction_number', 'children'),
     Output('p_20_pct', 'children'),
     Output('cases_expected_over_20', 'children')
     ],
    [Input('school_size', 'value'),
     Input('vax_rate', 'value'),
     Input('I0', 'value'),
     Input('R0', 'value'),
     Input('latent_period', 'value'),
     Input('infectious_period', 'value')]
)
def update_graph(school_size, vax_rate, I0, R0, latent_period, infectious_period):
    
    if school_size is None:
        school_size = 500
        
    if vax_rate is None:
        vax_rate = 0
        
    if I0 is None:
        I0 = 1

    if R0 is None:
        R0 = 15

    if latent_period is None:
        latent_period = 10.5

    if infectious_period is None:
        infectious_period = 8

    R0 = max(R0,0)
    
    # Update parameters, run simulations
    n_sim = 200

    params = copy.deepcopy(msp.params)
    params['population'] = [int(school_size)]
    params['vaccinated_percent'] = [0.01 * float(vax_rate)]
    params['I0'] = [int(I0)]
    params['R0'] = float(R0)
    params['incubation_period'] = float(latent_period)
    params['infectious_period'] = float(infectious_period)
    
    stochastic_sim = msp.StochasticSimulations(
        params, n_sim, print_summary_stats=False, show_plots=False)
    
    # Graph
    df_spaghetti_infected = stochastic_sim.df_spaghetti_infected
    df_spaghetti_infected_ma = stochastic_sim.df_spaghetti_infected_ma
    index_sim_closest_median = stochastic_sim.index_sim_closest_median
    
    # light_grey = px.colors.qualitative.Pastel2[-1]
    light_grey = 'rgb(220, 220, 220)'
    # light_grey = 'rgb(232, 232, 232)'
    color_map = {
        x: light_grey
        for x in df_spaghetti_infected_ma['simulation_idx'].unique()
        }
    color_map[index_sim_closest_median] = 'rgb(0, 153, 204)' #blue
    
    nb_curves_displayed = 20
    possible_idx = [
        x for x in df_spaghetti_infected_ma['simulation_idx'].unique()
        if x != index_sim_closest_median
        ]
    sample_idx = np.random.choice(possible_idx, nb_curves_displayed, replace=False)
    
    df_plot = pd.concat([
        df_spaghetti_infected_ma.loc[df_spaghetti_infected_ma['simulation_idx'].isin(sample_idx)],
        df_spaghetti_infected_ma.loc[df_spaghetti_infected_ma['simulation_idx'] == index_sim_closest_median]
        ])
    
        
    fig = px.line(
        df_plot,
        x='day',
        y='number_infected_7_day_ma',
        color='simulation_idx',
        color_discrete_map=color_map,
         labels={'simulation_idx': 'Simulation ID','number_infected': 'Number of people infected', 'day': 'Day DD', "number_infected_7_day_ma": "NN infected (7-day average)"},
        # alpha=0.1
        )
    
    fig.update_layout(showlegend=False,   
                      plot_bgcolor='white',  
                      xaxis=dict(
        title="Day",
        showgrid=True,  
        gridcolor="rgb(242,242,242)", 
        title_font=dict(size=20, color="black", family="Sans-serif"),  
        tickfont=dict(size=16, color="black", family="Sans-serif"), 
        zeroline=True,  
        zerolinecolor="black",  
        linecolor="black", 
        linewidth=2, 
        mirror=True  # Mirrors the axis line on all sides
    ),
    yaxis=dict(
        title="Number of infected people",
        showgrid=True,  
        gridcolor="rgb(242,242,242)", 
        title_font=dict(size=20, color="black", family="Sans-serif"),  
        tickfont=dict(size=16, color="black", family="Sans-serif"),  
         zeroline=True, 
        zerolinecolor="black",  
        linecolor="black",  
        linewidth=2,  
        mirror=True  # Mirrors the axis line on all sides
    ))
    
    # Summary statistics
    Rt = params['R0'] * (1 - 0.01 * float(vax_rate))

    effective_reproduction_number = '{:.2f}'.format(Rt)

    p_20_pct = '{:.0%}'.format(stochastic_sim.probability_20_plus_cases)
    outbreak_over_20 = p_20_pct

    # What uncertainty should we display for outbreak size
    outbreak_size_uncertainty_displayed = '95' # '90' '95' 'range' 'IQR'

    if stochastic_sim.expected_outbreak_size == 'NA':
        expected_outbreak_size_str = stochastic_sim.expected_outbreak_size
    else:
        expected_outbreak_size_str = str(int(stochastic_sim.expected_outbreak_size))
        
        if outbreak_size_uncertainty_displayed == '90':
            quantile_lb = 5
            quantile_ub = 95
            range_name = '90% CI'            
        elif outbreak_size_uncertainty_displayed == '95':
            quantile_lb = 2.5
            quantile_ub = 97.5
            range_name = '95% CI'
        elif outbreak_size_uncertainty_displayed == 'range':
            quantile_lb = 0
            quantile_ub = 100
            range_name = 'range'
        elif outbreak_size_uncertainty_displayed == 'IQR':
            quantile_lb = 25
            quantile_ub = 75
            range_name = 'IQR'
        
        uncertainty_outbreak_size_str = \
            ' \\[' +  \
            str(int(stochastic_sim.expected_outbreak_quantiles[quantile_lb])) + ' - ' +\
            str(int(stochastic_sim.expected_outbreak_quantiles[quantile_ub]))
        expected_outbreak_size_str += uncertainty_outbreak_size_str
    
    cases_expected_over_20 = expected_outbreak_size_str + "] cases"
              
    return fig, effective_reproduction_number, outbreak_over_20, cases_expected_over_20

@callback(
     [Output('school-dropdown', 'options'),
      Output('school-dropdown', 'value')#,
      ],
     [Input('county-dropdown', 'value')],
     prevent_initial_call=True
)
def update_school_selector(county):
    df_county = df.loc[df['County'] == county]
    new_school_options = sorted(df_county["School District or Name"].unique())
    school_selected = new_school_options[0]
    
    return new_school_options, school_selected

@callback(
     Output('vax_rate', 'value'),
     [Input('school-dropdown', 'value'),
      Input('county-dropdown', 'value')#,
      ]
)
def update_school_vax_rate(school, county):
    df_school = df.loc[
        (df['County'] == county) & 
        (df['School District or Name'] == school)# &
        ]
    school_vax_rate_pct = df_school['MMR_Vaccination_Rate'].values[0]
    school_vax_rate = float(school_vax_rate_pct.replace('%', ''))
    
        
    return school_vax_rate
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

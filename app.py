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
        dbc.Label("Select Texas County:", html_for="county_dropdown"),
        dcc.Dropdown(
            id="county-dropdown",
            options=sorted(df["County"].unique()),
            value=initial_county,
            clearable=False,
            maxHeight=600,
            optionHeight=50
        ),
    ],  className="mb-4",
    style={'width': '70%','fontFamily':'Sans-serif', 'font-size':'14pt'}
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
                    "The graph below shows 20 equally plausible outbreak trajectories, assuming no intervention. Use the interactive boxes below to change the number of students in the school, the number already infected at the start of the outbreak, the percent vaccinated against measles, and key epidemiological quantities.",
                    className="text-dark",  style={"font-style": "italic", "text-align": "left", "font-size": "20px", "margin-bottom": "10px"}
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
        dbc.Label("Select a School District", html_for="school_dropdown"),
        dcc.Dropdown(
            id="school-dropdown",
            options=school_options,
            value=initial_school,
            clearable=False,
            maxHeight=600,
            optionHeight=50
        ),
    ],  className="mb-4",
    style={'width': '70%', 'fontFamily':'Sans-serif', 'font-size':'16pt'}
)

vaccination_rate_label = html.H4(
    'Vaccination rate (%)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5, 'fontFamily':'Sans-serif', 'font-size':'16pt'})
vaccination_rate_selector = dcc.Input(
            id='vax_rate',
            type='number',
            placeholder='Vaccination rate (%)',
            value=85,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'12pt'}
        )

I0_label = html.H4(
    'Number of students initially infected',
    style={'display':'inline-block','margin-right':5, 'margin-left':5, 'fontFamily':'Sans-serif', 'font-size':'16pt'})
I0_selector = dcc.Input(
            id='I0',
            type='number',
            placeholder='Number of students initially infected',
            value=1.0,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'12pt'}
        )

school_size_label = html.H4(
    'School enrollment (number of students)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})
school_size_selector = dcc.Input(
            id='school_size',
            type='number',
            placeholder='School enrollment (number of students)',
            value=500,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'12pt'}
        )

app = Dash(
    prevent_initial_callbacks = 'initial_duplicate')

app.title = "UT Measles Outbreak Simulator"

# Navbar component
navbar = dbc.Navbar(
    dbc.Container(
        [
         #   html.Img(
         #       src="/assets/epiengage_logo_orange.png",  # Place the image in the "assets" folder
         #       height="40",
         #       className="header-logo",
         #       style={"marginRight": "10px"},
         #   ),
            html.Div("UT Measles Outbreak Simulator", style={"color": "white", "fontSize": "24px", "fontWeight": "bold", "textAlign": "right"}),
        ],
        fluid=True,
    ),
    color="#bf5700",
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


app.layout = dbc.Container(
    [
        dbc.Row([navbar], className="my-2"),
        dbc.Row([instructions_section], className="my-4"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                # First Line: school_size_label and school_size_selector
                                dbc.Row(
                                    [
                                        dbc.Col(html.Div(school_size_label), width=6),
                                        dbc.Col(html.Div(school_size_selector), width=6),
                                    ],
                                    className="my-2",  # Adds space below
                                ),

                                # Second Line: vaccination_rate_label and vaccination_rate_selector
                                dbc.Row(
                                    [
                                        dbc.Col(html.Div(I0_label), width=6),
                                        dbc.Col(html.Div(I0_selector), width=6),
                                    ],
                                    className="mb-2",  # Adds space below
                                ),

                                # Third Line: I0_label and I0_selector
                                dbc.Row(
                                    [
                                        dbc.Col(html.Div(vaccination_rate_label), width=6),
                                        dbc.Col(html.Div(vaccination_rate_selector), width=6),
                                    ],
                                    className="mb-2",  
                                ),
                            ]
                        ),
                        style={'border': 'none'}
                    ),
                    width=6,  
                    style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'}
                ),

                # Right Side (Dropdowns)
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(county_dropdown, className="mb-2"),  
                                html.Div(school_dropdown),
                            ]
                        ),
                        style={'border': 'none'}
                    ),
                    width=6,
                    style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'border': 'none'}
                ),
            ],
            style={'height': '30%', 'display': 'flex', 'flexDirection':'row'} 
        ),

        html.Hr(style={"border": "0.8px solid black", "width": "100%"}), # Horizontal line
        html.H3("Key Outbreak Statistics Without Intervention", style={"text-align": "center", "margin-top": "10px", "font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "20pt", "font-weight":"500"}),
        # Outcomes section
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
                                                    style={'color': '#bf5700', 'fontWeight': '800'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '20px',
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
                                                    style={'color': '#bf5700', 'fontWeight': '800'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '20px',
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
                                                    children='Expected outbreak size (assuming at least 20 cases)', 
                                                    style={'color': '#black', 'fontWeight': '500'}
                                        ),
                                        html.Br(),
                                        dcc.Markdown(id='cases_expected_over_20', 
                                                    style={'color': '#bf5700', 'fontWeight': '800'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '20px',
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

    # Bottom Section (Graph)
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(id="spaghetti_plot")
                ),
                style={'border': 'none'}
            ),
            width=12
        )
    ),
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
     Input('I0', 'value')]
)
def update_graph(school_size, vax_rate, I0):
    
    if school_size is None:
        school_size = 500
        
    if vax_rate is None:
        vax_rate = 0
        
    if I0 is None:
        I0 = 1
    
    # Update parameters, run simulations
    n_sim = 100

    params = copy.deepcopy(msp.params)
    params['population'] = [int(school_size)]
    params['vaccinated_percent'] = [0.01 * float(vax_rate)]
    params['I0'] = [int(I0)]
    stochastic_sim = msp.StochasticSimulations(
        params, n_sim, print_summary_stats=False, show_plots=False)
    
    # Graph
    df_spaghetti_infected_ma = stochastic_sim.df_spaghetti_infected_ma
    index_sim_closest_median = stochastic_sim.index_sim_closest_median
    
    # light_grey = px.colors.qualitative.Pastel2[-1]
    light_grey = 'rgb(220, 220, 220)'
    # light_grey = 'rgb(232, 232, 232)'
    color_map = {
        x: light_grey
        for x in df_spaghetti_infected_ma['simulation_idx'].unique()
        }
    color_map[index_sim_closest_median] = 'rgb(0, 153, 204)'
    
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
         labels={'number_infected': 'Number of people infected', 'day': 'Day'}
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

    if stochastic_sim.expected_outbreak_size == 'NA':
        expected_outbreak_size_str = stochastic_sim.expected_outbreak_size
    else:
        expected_outbreak_size_str = str(int(stochastic_sim.expected_outbreak_size)) + ' cases'
    
    cases_expected_over_20 = expected_outbreak_size_str
              
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
    app.run(debug=True)
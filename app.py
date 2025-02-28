#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:39:51 2025
Updated on Thu Feb 20 1:55:00 2025

@author: rfp437
"""
import dash
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
states = ["Texas"]

state_dropdown = html.Div(
    [
        dbc.Label("Select State"),
        dcc.Dropdown(
            id="state-dropdown",
            options=states,
            value="Texas",
            clearable=False,
            maxHeight=600,
            optionHeight=50
        ),
    ],  className="mb-4",
    style={'fontFamily':'Sans-serif', 'font-size':'16pt'}
)

county_dropdown = html.Div(
    [
        dbc.Label("Select Texas County", html_for="county_dropdown"), 
        dcc.Dropdown(
            id="county-dropdown",
            options=sorted(df["County"].unique()),
            value=initial_county,
            clearable=False,
            maxHeight=600,
            optionHeight=50,
            style={"whiteSpace": "nowrap", "width": "100%" },
            
        ),
    ],  className="mb-4 m-0",
    style={'fontFamily':'Sans-serif', 'font-size':'16pt','whiteSpace': 'nowrap', 'overflow':'visible'}
)

# df there should depend on the selected county
df_county = df.loc[df['County'] == initial_county]
school_options = sorted(df_county["School District or Name"].unique())
initial_school = 'AUSTIN ISD'

if initial_school not in school_options:
    initial_school = school_options[0]

school_dropdown = html.Div(
    [
        dbc.Label("Select a School District", html_for="school_dropdown", style={'fontFamily':'Sans-serif', 'font-size':'16pt'}),
        dcc.Dropdown(
            id="school-dropdown",
            options=school_options,
            value=initial_school,
            clearable=False,
            maxHeight=600,
            optionHeight=50,
            style={"whiteSpace": "nowrap", "width": "100%", 'font-size':'14pt'},
        ),
    ],  className="mb-4",
    style={'fontFamily':'Sans-serif', 'font-size':'16pt', 'whiteSpace': 'nowrap', 'overflow':'visible'}
)

vaccination_rate_label = html.H4(
    'Vaccination Rate (%)',
    style={'display':'inline-block','margin-bottom':0, 'margin-right':5, 'margin-left':5, 'fontFamily':'Sans-serif', 'font-size':'18pt'})
vaccination_rate_selector = dcc.Input(
            id='vax_rate',
            type='number',
            placeholder='Vaccination rate (%)',
            value=85,
            style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'16pt', 'textAlign': 'center', 'width':'7ch'}
        )

I0_label = html.H4(
    'Students Initially Infected',
    style={'display':'inline-block', 'fontFamily':'Sans-serif', 'font-size':'18pt', 'whiteSpace': 'nowrap', 'overflow':'visible'})
I0_selector = dcc.Input(
            id='I0',
            type='number',
            placeholder='Number of students initially infected',
            value=1.0,
            style={'display': 'flex', 'flexDirection': 'column', 'margin-left':'auto', 'fontFamily':'Sans-serif', 'font-size':'16pt', 'textAlign': 'center', 'width':'6ch'}
        )

school_size_label = html.H4(
    'School Enrollment',
    style={'display':'inline-block','fontFamily':'Sans-serif', 'font-size':'18pt', 'whiteSpace': 'nowrap', 'overflow':'visible'})
school_size_selector = dcc.Input(
            id='school_size',
            type='number',
            placeholder='School enrollment (number of students)',
            value=500,
            style={'display': 'flex', 'flexDirection': 'column', 'fontFamily':'Sans-serif', 'font-size':'16pt', 'textAlign': 'center', 'width':'6ch'}
        )

R0_label = html.H4([
    'Basic Reproduction Number',
     html.Span(" (R0)", 
        id="rep-tooltip",  
        style={"cursor": "pointer","color": "grey", "marginLeft": "5px"}
    )],
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})

R0_selector = dcc.Slider(
    id='R0',
    min=12,
    max=18,
    step=0.1,
    value=15,
    included=False,
    marks = {12: {'label': '12', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif'}},
             15: {'label': '15', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif', 'fontWeight': 'bold'}},
             18: {'label': '18', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif'}}
            },
    tooltip={'placement': 'top', 'always_visible': True, 'style':{"fontSize": "16pt"}},

)

latent_period_label = html.H4(
    'Average Latent Period (days)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})


latent_period_selector = dcc.Slider(
    id='latent_period',
    min=7,
    max=12,
    step=0.1,
    value=10.5,
    included=False,
    marks={7: {'label': '7', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif'}},
           10.5: {'label': '10.5', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif', 'fontWeight': 'bold'}},
           12: {'label': '12', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif'}},
    },
    tooltip={'placement': 'top', 'always_visible': True, 'style':{"fontSize": "16pt"}},
)
    
infectious_period_label = html.H4(
    'Average Infectious Period (days)',
    style={'display':'inline-block','margin-right':5, 'margin-left':5,'fontFamily':'Sans-serif', 'font-size':'16pt'})

infectious_period_selector = dcc.Slider(
    id='infectious_period',
    min=5,
    max=9,
    step=0.1,
    value=8,
    included=False,
    marks={5: {'label': '5', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif'}},
           8: {'label': '8', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif', 'fontWeight': 'bold'}},
           9: {'label': '9', 'style': {'font-size': '16pt', 'fontFamily': 'Sans-serif'}}
           },
    tooltip={'placement': 'top', 'always_visible': True, 'style':{"fontSize": "16pt"}},
)

app = Dash(
    prevent_initial_callbacks = 'initial_duplicate')
server = app.server
app.title = "epiENGAGE Measles Outbreak Simulator"

# Navbar component
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.Img(
                src="/assets/epiengage_logo_orange.png",  # Place the image in the "assets" folder
                height="50",
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
                html.Div(
                    [
                        dbc.Col(html.Div(state_dropdown),className="mb-2"),
                        dbc.Col(html.Div(county_dropdown),className="mb-2"),
                        dbc.Col(html.Div(school_dropdown),className="mb-2"),
                    ]
                ),
                title="School Lookup ▾ ", 
                style={"font-size": "18pt", "width":"100%"}, 
                className="m-0"
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
                        html.Br(),
                        dbc.Col(html.Div(R0_label),className="mb-2"),
                        dbc.Col(html.Div(R0_selector),className="mb-2"),
                        dbc.Col(html.Div(latent_period_label),className="mb-2"),
                        dbc.Col(html.Div(latent_period_selector),className="mb-2"),
                        dbc.Col(html.Div(infectious_period_label),className="mb-2"),
                        dbc.Col(html.Div(infectious_period_selector),className="mb-2"),
                    ]
                ),
                title="Change Parameters ▾",
                style={"textAlign": "center"}  # Add this line to center the text
                
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
    html.Br(),

    # Main Layout with Left and Right Sections
    dbc.Row([
        # Left section
        dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H3("Model Inputs", style={"margin-left":"0.2em", "margin-top": "0.5em","font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "24pt", "font-weight":"500", "textAlign": "center"}, className="mt-2"),
                                html.Br(),
                                dbc.Row([
                                    dbc.Col([ 
                                        html.Div(school_size_label), 
                                        html.Div(school_size_selector),
                                        ], className="d-flex flex-column align-items-center"),
                                ], className="d-flex flex-column align-items-center mb-2"),

                                dbc.Row([
                                     dbc.Col([ 
                                        html.Div(I0_label), 
                                        html.Div(I0_selector),
                                        ], className="d-flex flex-column align-items-center"),
                                ], className="d-flex flex-column align-items-center mb-2"),

                                dbc.Row([
                                    dbc.Col(html.Div(vaccination_rate_label), className="d-flex flex-column align-items-center"),
                            ]),

                            dbc.Row(
                                dbc.Col(html.I("Enter value or select school from dropdown"),className="d-flex flex-column align-items-center", style={"font-size": "12pt", "margin-bottom": "0.5em"}),
                            ),
                                dbc.Row([
                                    dbc.Col([ 
                                        html.Div(vaccination_rate_selector), html.Div(" OR ", style={"font-size": "16pt", "margin-top": "0.5em", "margin-bottom": "0.5em"}),
                                        html.Div(accordion_vax, style={"width":"100%", "textAlign": "center"}),
                                        ], className="d-flex flex-column align-items-center"),
                                    #dbc.Col(html.Div(vaccination_rate_selector), style={"font-size": "16pt"}),
                                    #dbc.Col(accordion_vax, style={"font-size": "16pt", "padding": "none"}),
                                 ], style={"border-bottom": "2px solid black", "margin-right":"0.2em"}),

                                html.Br(),
                                html.H3("Epidemic Parameters", style={"margin-left":"0.2em", "margin-top": "0.5em","font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "24pt", "font-weight":"500", "textAlign": "center"}),
                                html.Br(),
                                dbc.Row(dbc.Col(html.I("Caution – Default values reflect published estimates. Significant changes may result in inaccurate projections."),className="mb-2 align-items-center", style={"font-size": "14pt", "textAlign":"center"})),

                                dbc.Row([
                                    dbc.Col(accordion,className="mb-2"),
                            ]),
                            ]
                        ),
                        style={'border': 'none'}
                    ),
                    width=3, xs=12, sm=12, md=12, lg=12, xl=3,  
                    style={"border-right": "2px solid black", "padding": "10px"}, 
        ),

        # Right section 
        dbc.Col([
        # Outcomes section
         html.H3("School Outbreak Projections", style={"text-align": "center", "margin-top": "0.8em","font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "24pt", "font-weight":"500"}),
         html.H3("Projections assume no interventions and no breakthrough infections in vaccinated students. They do not account for infections of non-students in the surrounding community.", style={"text-align": "center", "font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "12pt", "font-weight":"400", "font-style": "italic", "line-height": "1"}),
         #html.H3("In the statistics below, an “outbreak” is defined as any simulation with at least 20 new infections beyond the initial cases",style={"text-align": "center", "font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "14pt", "font-weight":"400", "font-style": "italic", "margin":"none"}),
         html.Br(), 
          dbc.Row(
            [
             
               # Chance of an Outbreak
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                             [
                                html.Div(
                                    [
                                        dcc.Markdown(id='outbreak', 
                                                    children='Chance of an outbreak', 
                                                    style={'color': '#black', 'fontWeight': '500', 'font-size': '22pt', "margin":"none"}
                                        ),
                                        dcc.Markdown("*exceeding 20 new infections*", style={'font-size': '16pt', "margin":"none"}),
                                        dcc.Markdown(id='p_20_pct', 
                                                    style={'color': '#bf5700', 'fontWeight': '800', 'font-size': '23pt', 'margin-top':'0.5em'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '18pt',
                                        'border': 'none'
                                    }
                                )
                            ]
                        ),
                        style={'border':'none'}  
                    ),

                ),

                # Expected Outbreak Size
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    [
                                        dcc.Markdown(id='cases', 
                                                    children='Likely outbreak size', 
                                                    style={'color': '#black', 'fontWeight': '500', 'font-size': '22pt', 'margin':'none'}
                                        ),
                                        dcc.Markdown(id='cases_expected_over_20', 
                                                    style={'color': '#bf5700', 'fontWeight': '800', 'font-size':'23pt', 'margin-top':'0.5em'}
                                        ),
                                    ],
                                    style={
                                        'textAlign': 'center', 
                                        'fontFamily': '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
                                        'fontSize': '18pt',
                                        'border': 'none'
                                    }
                                )
                            ]
                        ),
                        style={'border':'none'}
                    ),
                    style={'borderLeft': '3px solid #bf5700'}  
                ),
            ],
        ),

            html.Br(),

            # Bottom Component in the Right Section
            dbc.Row([
                 dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.H3("This graph shows 20 plausible school outbreak curves.", style={"text-align": "center", "margin-top": "1em", "margin-bottom":"1em", "margin-left": "1.8em", "font-family":  '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif', "font-size": "14pt", "font-weight":"400", "font-style": "italic"}),
                            dcc.Graph(id="spaghetti_plot"),
                        ]),
                        style={'border':'none'}, 
                    ),
            #width=12,# set width to 12 for all screen sizes
        ),
            ], style={"border-top": "2px solid black", "border-left":"1em", "height": "100%", "width": "100%", "margin-top": "1em"}), 
        ], className="col-xl-9"),
    ]),  

    html.Br(),

    html.Div([
        html.A("MODEL: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A(["This dashboard uses a simple stochastic compartmental susceptible-exposed-infectious-removed (SEIR) model. The model only considers infections of students enrolled in the school and assumes that fully vaccinated students cannot be infected. The default parameters are based on estimates that are widely used by public health agencies: (1) a basic reproduction number (", html.I([html.A(["R", html.Sub("0")])])," ) of 15 ["]),
        html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("], (2) an average latent period of 10.5 days ["),
        html.A("CDC’s Measles Clinical Diagnosis Fact Sheet", href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("], and (3) an average infectious period of 8 days ["), 
        html.A("CDC’s Measles Clinical Diagnosis Fact Sheet", href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}), 
        html.A("]. Parameter ranges are based on ["),
        html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("] and ["),
        html.A("Bailey and Alfa-Steinberger 1970", href="https://doi.org/10.1093/biomet/57.1.141", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}), 
        html.A("]. "),
        html.A("The basic reproduction number is the expected number of people a single case will infect, assuming nobody has immunity from vaccination or prior infection. If a school has a high vaccination rate, the effective reproduction number at the start of an outbreak will be much lower than the basic reproduction number."),
        html.Ul("", style={"margin-bottom": "1em"}),
        html.A("KEY OUTBREAK STATISTICS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("Values are estimated from 200 stochastic simulations as follows."),
        html.Ul([
            html.Li([html.I("Chance of an outbreak"), html.A([" – The proportion of 200 simulations in which at least 20 additional students become infected, not counting the initial cases."])]),
            html.Li([html.I("Likely outbreak size"), " – For each simulation that results in at least 20 additional infections, the total number of students infected (including the initially infected) is calculated. The reported range (2.5th to 97.5th percentile) reflects the central 95% of these total infection counts.", html.Br(style={"margin": "0", "padding": "0"})]),
        ], style={"margin-bottom": "1em"}),
        html.A("PROJECTIONS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("The 20 curves in the graph correspond to 20 independent simulations selected at random from 200 stochastic simulations. The y-axis values are seven-day moving averages of the total number of people infected (both exposed and infectious cases). The highlighted curve corresponds to the simulation that produced a total outbreak size closest to the median across the 200 simulations."),
        html.Ul("", style={"margin-bottom": "1em"}),
        html.A("VACCINE RATES: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("The School Lookup menu gives the percent of kindergarten students who are completely vaccinated for MMR, as reported by the Texas Department of Health and Human Services for the 2023-2024 school year ["),
        html.A("DSHS 2023-2024 Annual Report of Immunization Status", href="https://www.dshs.texas.gov/immunizations/data/school/coverage", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
        html.A("]."),
        html.Ul("", style={"margin-bottom": "1em"}),
        html.A("ADDITIONAL DETAILS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
        html.A("epiENGAGE Measles Outbreak Simulator - Model Details", href="/assets/epiENGAGE_Measles_Outbreak_Simulator–Model Details-2025.pdf", target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
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

    # print(df_plot) 
    # df_plot.to_csv("df_plot_output.csv", index=False)
        
    fig = px.line(
        df_plot,
        x='day',
        y='number_infected_7_day_ma',
        color='simulation_idx',
        color_discrete_map=color_map,
         labels={'simulation_idx': '','number_infected': 'Number of students infected', 'day': 'Day DD', "number_infected_7_day_ma": "NN infected (7-day average)"},
        )
    
    fig.update_traces(hovertemplate="Day %{x}<br>%{y:.1f} Infected<extra></extra>")
    fig.update_traces(line=dict(width=2))  # Reduce line thickness

    fig.update_layout(showlegend=False,   
                      plot_bgcolor='white', 
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(
        title="Day",
        showgrid=True,  
        gridcolor="rgb(242,242,242)", 
        title_font=dict(size=20, color="black", family="Sans-serif"),  
        tickfont=dict(size=16, color="black", family="Sans-serif"), 
        zeroline=True,  
        zerolinecolor="white",  
        linecolor="grey", 
        linewidth=2, 
        mirror=True  # Mirrors the axis line on all sides
    ),
    yaxis=dict(
        title="Number of infected students",
        showgrid=True,  
        gridcolor="rgb(242,242,242)", 
        title_font=dict(size=20, color="black", family="Sans-serif"),  
        tickfont=dict(size=16, color="black", family="Sans-serif"),  
        zeroline=True, 
        zerolinecolor="white",  
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
        cases_expected_over_20 = "Fewer than 20 cases"

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
        
        uncertainty_outbreak_size_str = str(int(stochastic_sim.expected_outbreak_quantiles[quantile_lb])) + ' - ' +\
            str(int(stochastic_sim.expected_outbreak_quantiles[quantile_ub]))
        #expected_outbreak_size_str += uncertainty_outbreak_size_str
    
        cases_expected_over_20 = uncertainty_outbreak_size_str+ " cases"
              
    return fig, outbreak_over_20, cases_expected_over_20

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

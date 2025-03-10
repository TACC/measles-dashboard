# These functions/components do NOT change based on callbacks
#   -- these stay STATIC on the webpage

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

from app_styles import RESULTS_HEADER_STYLE, BASE_FONT_FAMILY_STR, SELECTOR_NOTE_STYLE

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
            html.Div("epiENGAGE Measles Outbreak Simulator",
                     style={"color": "white", "fontSize": "24px", "fontWeight": "bold", "textAlign": "right"}),
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

# Labels
########

vaccination_rate_label = html.H4(
    'Vaccination Rate (%)',
    style={'display': 'inline-block', 'margin-bottom': 0, 'margin-right': 5, 'margin-left': 5,
           'fontFamily': 'Sans-serif', 'font-size': '18pt'})

school_size_label = html.H4(
    'School Enrollment',
    style={'display': 'inline-block', 'fontFamily': 'Sans-serif', 'font-size': '18pt', 'whiteSpace': 'nowrap',
           'overflow': 'visible'})

I0_label = html.H4(
    'Students Initially Infected',
    style={'display': 'inline-block', 'fontFamily': 'Sans-serif', 'font-size': '18pt', 'whiteSpace': 'nowrap',
           'overflow': 'visible'})

R0_label = html.H4([
    'Basic Reproduction Number (R0)'],
    style={'display': 'inline-block', 'margin-right': 5, 'margin-left': 5, 'fontFamily': 'Sans-serif',
           'font-size': '16pt'})

latent_period_label = html.H4(
    'Average Latent Period (days)',
    style={'display': 'inline-block', 'margin-right': 5, 'margin-left': 5, 'fontFamily': 'Sans-serif',
           'font-size': '16pt'})

infectious_period_label = html.H4(
    'Average Infectious Period (days)',
    style={'display': 'inline-block', 'margin-right': 5, 'margin-left': 5, 'fontFamily': 'Sans-serif',
           'font-size': '16pt'})


def school_outbreak_projections_header():

    TITLE_STYLE = {
        "text-align": "center",
        "font-family": BASE_FONT_FAMILY_STR,
        "font-size": "24pt",
        "font-weight": "500"
    }

    # Define the list of texts
    texts = [
        "School Outbreak Projections",
        "Projections assume no interventions and no breakthrough infections "
        "among vaccinated students, and they do not account for infections "
        "among non-students in the surrounding community.",
        "Active measles control measures could lead to substantially smaller "
        "and shorter outbreaks than these projections suggest."
    ]

    # Generate the H3 components dynamically
    return html.Div([
        html.H3(text, style=TITLE_STYLE if i == 0 else SELECTOR_NOTE_STYLE)
        for i, text in enumerate(texts)])


# bottom info section
def bottom_info_section():
    return html.Div([
            html.A("MEASLES VACCINATION:", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(" For additional information about measles vaccines, visit the "),
            html.A("CDC's MMR vaccination webpage", href="https://www.cdc.gov/vaccines/vpd/mmr/public/index.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("MODEL: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "This dashboard uses a simple stochastic compartmental susceptible-exposed-infectious-removed (SEIR) model. The model includes only enrolled students, assumes vaccinated individuals cannot become infected, and does not consider intervention measures."),
            html.A([
                " The default parameters are based on estimates that are widely used by public health agencies: (1) a basic reproduction number (",
                html.I([html.A(["R", html.Sub("0")])]), " ) of 15 ["]),
            html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("], (2) an average latent period of 10.5 days ["),
            html.A("CDC’s Measles Clinical Diagnosis Fact Sheet",
                   href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("], and (3) an average infectious period of 8 days ["),
            html.A("CDC’s Measles Clinical Diagnosis Fact Sheet",
                   href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]. Parameter ranges are based on ["),
            html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("] and ["),
            html.A("Bailey and Alfa-Steinberger 1970", href="https://doi.org/10.1093/biomet/57.1.141", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]. "),
            html.A(
                "The basic reproduction number is the expected number of people a single case will infect, assuming nobody has immunity from vaccination or prior infection. If a school has a high vaccination rate, the effective reproduction number at the start of an outbreak will be much lower than the basic reproduction number."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("KEY OUTBREAK STATISTICS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A("Values are estimated from 200 stochastic simulations as follows."),
            html.Ul([
                html.Li([html.I("Chance of exceeding 20 infections"), html.A([
                    " – The proportion of 200 simulations in which at least 20 additional students become infected, not counting the initial cases."])]),
                html.Li([html.I("Likely outbreak size"),
                         " – For each simulation that results in at least 20 additional infections, the total number of students infected is calculated, including the students initially infected. The reported range reflects the middle 95% of these values (i.e., the 2.5th to 97.5th percentile).",
                         html.Br(style={"margin": "0", "padding": "0"})]),
            ], style={"margin-bottom": "1em"}),
            html.A("PROJECTIONS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "The 20 curves in the graph correspond to 20 independent simulations selected at random from 200 stochastic simulations. The y-axis values are seven-day moving averages of the total number of people currently infected (both exposed and infectious cases). The highlighted curve corresponds to the simulation that produced a total outbreak size closest to the median across the 200 simulations."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("VACCINE RATES: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "The School Lookup menu gives the percent of kindergarten and 7th grade students who are completely vaccinated for MMR, as reported by the Texas Department of Health and Human Services for the 2023-2024 school year ["),
            html.A("DSHS 2023-2024 Annual Report of Immunization Status",
                   href="https://www.dshs.texas.gov/immunizations/data/school/coverage", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("ADDITIONAL DETAILS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A("epiENGAGE Measles Outbreak Simulator - Model Details",
                   href="/assets/epiENGAGE_Measles_Outbreak_Simulator–Model Details-2025.pdf", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none"}),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("For questions, please contact ", style={"fontSize": "18px", "font-style": "italic"}),
            html.A("utpandemics@austin.utexas.edu", href="mailto:utpandemics@austin.utexas.edu", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none", "font-style": "italic"}),
            html.A("."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A(
                "This dashboard was developed with support from the CDC’s Center for Forecasting and Outbreak Analytics.",
                style={"fontSize": "18px", "font-style": "italic"}),
        ],
            style={
                "backgroundColor": "#eaebec",
                "color": "black",
                "padding": "10px",
                "textAlign": "left",
                "marginBottom": "10px",
                "fontSize": "18px"
            })

def bottom_credits():
    return dbc.Row([
            dbc.Col(html.Div([
                html.P("©2025 ", style={"display": "inline", "font-size": "11px", "color": "#ffffff"}),
                html.A("Texas Advanced Computing Center", href="https://www.tacc.utexas.edu/", target="_blank",
                       style={"color": "#ffffff", "text-decoration": "none", "font-size": "11px"}),
                html.Br(),
                html.A("The University of Texas at Austin, Office of the Vice President for Research",
                       href="https://research.utexas.edu", target="_blank",
                       style={"color": "#ffffff", "text-decoration": "none", "font-size": "11px"}),
            ], style={"textAlign": "center", "padding": "10px"}), width=12)
        ], style={"backgroundColor": "#292929", "marginTop": "auto"})

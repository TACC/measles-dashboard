# These functions/components do NOT change based on callbacks
#   -- these stay STATIC on the webpage

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

from app_styles import SELECTOR_LABEL_STYLE, BASE_FONT_FAMILY_STR, SELECTOR_NOTE_STYLE, \
    BOTTOM_CREDITS_TEXT_STYLE

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

vaccination_rate_label = html.H4('Vaccination Rate (%)', style={**SELECTOR_LABEL_STYLE})

school_size_label = html.H4('School Enrollment', style={**SELECTOR_LABEL_STYLE})

I0_label = html.H4('Students Initially Infected', style={**SELECTOR_LABEL_STYLE})

R0_label = html.H4(['Basic Reproduction Number (R0)'], style={**SELECTOR_LABEL_STYLE})

latent_period_label = html.H4('Average Latent Period (days)', style={**SELECTOR_LABEL_STYLE})

infectious_period_label = html.H4('Average Infectious Period (days)', style={**SELECTOR_LABEL_STYLE})

threshold_selector_label = html.H4('Minimum Outbreak Size (infections)', style={**SELECTOR_LABEL_STYLE})


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
            html.A("MEASLES VACCINATION: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A("Two doses of the MMR vaccine are recommended by doctors as the best way to protect against measles, mumps, and rubella. For additional information about measles vaccines, visit the ["),
            html.A("CDC's MMR vaccination webpage", href="https://www.cdc.gov/vaccines/vpd/mmr/public/index.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("MODEL: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "This dashboard uses a simple stochastic compartmental susceptible-exposed-infectious-removed (SEIR) model. The model includes only enrolled students, assumes vaccinated individuals cannot become infected, and does not consider intervention measures. Public health interventions–such as exclusion from school, active case finding, quarantine, isolation, and vaccination–would likely result in shorter and smaller outbreaks compared to these projections. "),
            html.Ul("", style={"margin-bottom": "0.5em"}),
            html.A("The default parameters are based on estimates that are widely used by public health agencies: "),
            html.A(["(1) a basic reproduction number (",
                html.I([html.A(["R"])]), html.Sub("0"), ") of 15 ["]),
            html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("], (2) an average latent period of 10.5 days ["),
            html.A("CDC’s Measles Clinical Diagnosis Fact Sheet",
                   href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("], and (3) an average infectious period of 5 days, derived from a total infectious period of approximately 8 days ["),
            html.A("CDC’s Measles Clinical Diagnosis Fact Sheet",
                   href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("],  assuming students remain home after symptom onset. Parameter ranges are based on ["),
            html.A("ECDC’s Factsheet about measles", href="https://www.ecdc.europa.eu/en/measles/facts",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("] and ["),
            html.A("Bailey and Alfa-Steinberger 1970", href="https://doi.org/10.1093/biomet/57.1.141", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]. "),
            html.Ul("", style={"margin-bottom": "0.5em"}),
            html.A(
                ["The basic reproduction number is the expected number of people a single case will infect, assuming nobody has immunity from vaccination or prior infection. If prior immunity exists due to vaccination, the effective reproduction number at the start of an outbreak will be lower than the basic reproduction number. Individual schools may experience higher or lower ",
                 html.I([html.A(["R"])]), html.Sub("0"), " values depending on classroom structures, daily activities, and other contextual factors."]),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("KEY OUTBREAK STATISTICS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(["These values are derived from 200 stochastic simulations. The default outbreak threshold of", html.I("10 additional infections"), html.A(" can be adjusted under ")]),
            html.I("Change Parameters"),
            html.A("."),
            html.Ul([
                html.Li([html.I("Chance of Exceeding 10 Infections"), html.A([
                    " – The proportion of simulations (out of 200) in which at least 10 additional students become infected (excluding the initial cases). The threshold of 10 infections was chosen to distinguish introductions that lead to sustained transmission from those that quickly fade out."])]),
                html.Li([html.I("Likely Outbreak Size"),
                         " – Among simulations that surpass the 10-infection threshold, the total number of infected students is calculated (including the initially infected students). The reported range corresponds to the middle 95% of these values (i.e., the 2.5th to 97.5th percentile).",
                         html.Br(style={"margin": "0", "padding": "0"})]),
            ], style={"margin-bottom": "1em"}),
            html.A("PROJECTIONS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "The 20 curves in the graph correspond to 20 independent simulations selected at random from 200 stochastic simulations. The y-axis values are seven-day moving averages of the total number of people currently infected (both exposed and infectious cases). The highlighted curve corresponds to the simulation that produced a total outbreak size closest to the median across the 200 simulations. Variations between simulation curves are expected, as the model accounts for inherent randomness and uncertainty present under real-world conditions."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("VACCINE RATES: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "The School/District Lookup menu gives MMR vaccination rates for schools and school districts provided by individual cities, counties, or states. They may be out of date and may not represent all grades in a school. For the state of Texas, the data are the percent of kindergarten or 7th grade students who are completely vaccinated for MMR, as reported by the Texas Department of Health and Human Services for the 2023-2024 school year ["),
            html.A("DSHS 2023-2024 Annual Report of Immunization Status",
                   href="https://www.dshs.texas.gov/immunizations/data/school/coverage", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]."),
            html.A(" For the state of North Carolina, the data are the percent of kindergarten students who received all required MMR doses, as reported by the North Carolina Department of Health and Human Services for the 2023-2024 academic year ["),
            html.A("NCDHHS Kindergarten Immunization Data Dashboard",
                   href="https://www.dph.ncdhhs.gov/programs/epidemiology/immunization/data/kindergarten-dashboard", target="_blank",
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
                       style={**BOTTOM_CREDITS_TEXT_STYLE}),
                html.Br(),
                html.A("The University of Texas at Austin, Office of the Vice President for Research",
                       href="https://research.utexas.edu", target="_blank",
                       style={**BOTTOM_CREDITS_TEXT_STYLE}),
            ], style={"textAlign": "center", "padding": "10px"}), width=12)
        ], style={"backgroundColor": "#292929", "marginTop": "auto"})

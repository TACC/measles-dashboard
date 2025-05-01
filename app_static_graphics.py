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

vaccination_rate_label = html.H4('Vaccination rate (%)', style={**SELECTOR_LABEL_STYLE})

school_size_label = html.H4('School enrollment', style={**SELECTOR_LABEL_STYLE})

I0_label = html.H4('Students initially infected', style={**SELECTOR_LABEL_STYLE})

R0_label = html.H4(['Basic reproduction number (R0)'], style={**SELECTOR_LABEL_STYLE})

latent_period_label = html.H4('Average latent period (days)', style={**SELECTOR_LABEL_STYLE})

infectious_period_label = html.H4('Average infectious period (days)', style={**SELECTOR_LABEL_STYLE})

threshold_selector_label = html.H4('Minimum outbreak size (new infections)', style={**SELECTOR_LABEL_STYLE})

vaccine_efficacy_selector_label = html.H4('Vaccine efficacy - susceptibility (%)', style={**SELECTOR_LABEL_STYLE})

vaccinated_infectiousness_selector_label = html.H4('Vaccine efficacy - infectiousness (%)', style={**SELECTOR_LABEL_STYLE})


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
        "Projections assume no interventions and do not account for infections "
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
                "This dashboard uses a simple stochastic compartmental susceptible-exposed-infectious-removed-vaccinated (SEIRV) model. It assumes that vaccination reduces both susceptibility to infection and infectiousness when infected. The model includes only enrolled students and does not consider intervention measures. Public health interventions–such as exclusion from school, active case finding, quarantine, isolation, and vaccination campaigns–would likely result in shorter and smaller outbreaks compared to these projections. "),
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
            html.A("], (3) an average infectious period of 5 days, derived from a total infectious period of approximately 8 days ["),
            html.A("CDC’s Measles Clinical Diagnosis Fact Sheet",
                   href="https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("],  assuming students remain home after symptom onset, (4) a vaccine efficacy of 99.7% in preventing infection ["),
            html.A("van Boven et al. 2010", href="https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2010.0086",
                   target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("], and (5) a 95% reduction in infectiousness among vaccinated individuals ["),
            html.A("Tranter et al. 2024", href="https://wwwnc.cdc.gov/eid/article/30/9/24-0150_article",
                    target="_blank", style={"color": "#1b96bf", "textDecoration": "none"}),
            html.A("]. Parameter ranges are based on ["),
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
            html.A(["These values are derived from 1000 stochastic simulations. The default outbreak threshold of", html.I("10 additional infections"), html.A(" can be adjusted under ")]),
            html.I("Change Parameters"),
            html.A("."),
            html.Ul([
                html.Li([html.I("Chance of Exceeding 10 Infections"), html.A([
                    " – The proportion of simulations (out of 1000) in which at least 10 additional students become infected (excluding the initial cases). The threshold of 10 infections was chosen to distinguish introductions that lead to sustained transmission from those that quickly fade out."])]),
                html.Li([html.I("Likely Outbreak Size"),
                         " – Among simulations that surpass the 10-infection threshold, the total number of infected students is calculated for both vaccinated and unvaccinated (including the initially infected unvaccinated students). The reported ranges correspond to the middle 95% of these values (i.e., the 2.5th to 97.5th percentile).",
                         html.Br(style={"margin": "0", "padding": "0"})]),
            ], style={"margin-bottom": "1em"}),
            html.A("PROJECTIONS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A(
                "The 20 curves in the graph correspond to 20 independent stochastic simulations. The y-axis values are seven-day moving averages of the total number of people currently infected (both exposed and infectious cases). The highlighted curve corresponds to the simulation that produced a total outbreak size closest to the median across the 1000 simulations. Variations between simulation curves are expected, as the model accounts for inherent randomness and uncertainty present under real-world conditions."),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("VACCINE RATES: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A("The School/District lookup menu provides MMR vaccination rates for schools and school districts, based on data reported by individual cities, counties, or states. Unless otherwise noted, these estimates reflect the specific grade level indicated (e.g., kindergarten or 7th grade) for the 2023–2024 academic year. Please note that these values may be outdated and may not represent all grade levels within a school. For additional information, please refer to the original sources below."),
            html.Ul([
                html.Li([html.A("Alabama ("),
                         html.A("Provided directly by ADPH)")]),
                html.Li([html.A("Arizona ("),
                         html.A("AZ DHS Immunization Coverage",
                                href="https://www.azdhs.gov/preparedness/epidemiology-disease-control/immunization/index.php#reports-immunization-coverage", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("California ("),
                         html.A("CPDPH Immunization",
                                href="https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Fwww.cdph.ca.gov%2FPrograms%2FCID%2FDCDC%2FCDPH%2520Document%2520Library%2FImmunization%2F2023_24CAKindergartenGradeData_Letter.xlsx&data=05%7C02%7C%7Cd2dcc0072b3e43928b6d08dd8391abf2%7C31d7e2a5bdd8414e9e97bea998ebdfe1%7C0%7C0%7C638811383882331939%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=nF%2B05XKDV2XtOch2WZ%2BWWQkPEDD6pNZiTiTcVQyFa5I%3D&reserved=0", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A("). Precautions for student de-identification are applied and based on school enrollment. For schools with:"),
                         html.Ul([
                             html.Li("20–49 enrollees: Values ≥95% reported as 95%; values ≤5% reported as 5%"),
                             html.Li("50–99 enrollees: Values ≥98% reported as 98%; values ≤2% reported as 2%"),
                             html.Li("100 or more enrollees: Values ≥99% reported as 99%; values ≤1% reported as 1%")
                            ], style={"margin-bottom": "0em"}),
                            ]),
                html.Li([html.A("Colorado ("),
                         html.A("CDPHE School and Child Care Immunization",
                                href="https://data-cdphe.opendata.arcgis.com/search?q=school%20and%20child%20care%20immunization%20data%20reporting", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Connecticut ("),
                         html.A("CTDPH Immunization Rates by School",
                                href="https://data.ct.gov/Health-and-Human-Services/2023-2024-Kindergarten-Immunization-Rates-by-Schoo/iux5-vrzq/about_data", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Iowa ("),
                         html.A("Provided directly by Iowa HHS)")]),
                html.Li([html.A("Maryland ("),
                         html.A("MDH Immunization Rates by School",
                                href="https://health.maryland.gov/phpa/OIDEOR/IMMUN/Pages/Kindergarten_Immunization_Rates_by_School.aspx", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Massachusetts ("),
                         html.A("MA DPH School Immunization",
                                href="https://www.mass.gov/info-details/school-immunizations", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Michigan ("),
                         html.A("Provided directly by MDHHS)")]),
                html.Li([html.A("Minnesota ("),
                         html.A("MDH School Immunization",
                                href="https://www.health.state.mn.us/people/immunize/stats/school/index.html", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("New Mexico ("),
                         html.A("Provided directly by NM Health)")]),
                html.Li([html.A("New York ("),
                         html.A("NY DOH School Immunization Survey",
                                href="https://health.data.ny.gov/Health/School-Immunization-Survey-Beginning-2019-20-Schoo/btkd-y8bp/about_data", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("North Carolina ("),
                         html.A("NCDHHS Kindergarten Immunization Data Dashboard",
                                href="https://www.dph.ncdhhs.gov/programs/epidemiology/immunization/data/kindergarten-dashboard", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Oregon ("),
                         html.A("OHA School Immunization Coverage",
                                href="https://www.oregon.gov/oha/PH/PREVENTIONWELLNESS/VACCINESIMMUNIZATION/GETTINGIMMUNIZED/Pages/SchRateMap.aspx", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Pennsylvania ("),
                         html.A("Pennsylvania DOH School Immunization Rates",
                                href="https://www.pa.gov/agencies/health/programs/immunizations/rates.html", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Texas ("),
                         html.A("DSHS 2023-2024 Annual Report of Immunization Status",
                                href="https://www.dshs.texas.gov/immunizations/data/school/coverage", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                html.Li([html.A("Washington ("),
                         html.A("WA DOH School Immunization",
                                href="https://doh.wa.gov/data-and-statistical-reports/washington-tracking-network-wtn/school-immunization/dashboard", target="_blank",
                                style={"color": "#1b96bf", "textDecoration": "none"}),
                         html.A(")")]),
                # html.Li([html.A("New_state ("),
                #          html.A("Source_name",
                #                 href="https_link", target="_blank",
                #                 style={"color": "#1b96bf", "textDecoration": "none"}),
                #          html.A(")")]),
            ], style={"margin-bottom": "1em"}),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("ADDITIONAL DETAILS: ", style={"fontWeight": "bold", "fontSize": "18px"}),
            html.A("epiENGAGE Measles Outbreak Simulator - Model Details",
                   href="/assets/epiENGAGE_Measles_Outbreak_Simulator–Model Details-2025.pdf", target="_blank",
                   style={"color": "#1b96bf", "textDecoration": "none"}),
            html.Ul("", style={"margin-bottom": "1em"}),
            html.A("For questions, please contact ", style={"fontSize": "18px", "font-style": "italic"}),
            html.A("epiengage@austin.utexas.edu", href="mailto:epiengage@austin.utexas.edu", target="_blank",
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

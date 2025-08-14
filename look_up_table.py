# look_up_table
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import pandas as pd

def lookup_table_layout():
    return dbc.Container([
        html.H1("Outbreak Risk Lookup Table", className="text-center mb-4"),
        html.Div([
            dbc.Label("Select Metric:", html_for="metric-dropdown"),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[
                    {"label": "Probability of exceeding 3 total infections", "value": "percent_probability_exceeding_3_total_cases"},
                    {"label": "Probability of exceeding 10 total infections", "value": "percent_probability_exceeding_10_total_cases"},
                    {"label": "Probability of exceeding 20 total infections", "value": "percent_probability_exceeding_20_total_cases"},
                    {"label": "Number of infections - AVERAGE", "value": "total_infected_mean"},
                    {"label": "Number of infections - MEDIAN", "value": "total_infected_median"},
                    {"label": "Number of infections - 2.5th Percentile", "value": "total_infected_q2.5"},
                    {"label": "Number of infections - 97.5th Percentile", "value": "total_infected_q97.5"},
                ],
                value="percent_probability_exceeding_10_total_cases",
                clearable=False,                 
                maxHeight=600,                     
                optionHeight=50,  
                style={"width": "50%", "margin-bottom": "20px"}
            ),
        ], className="mb-4"),
        
        html.Div(id="lookup-table-container")
    ], style={"paddingTop": "40px"})

@callback(
    Output("lookup-table-container", "children"),
    Input("metric-dropdown", "value")
)


def update_table(selected_metric):

    df = pd.read_csv("Measles Outbreak Risk Lookup Table.csv")

    #df ['vax_prop'] = pd.to_numeric(df['vax_prop'], errors='coerce')

    data = df[df['metric'] == selected_metric]
    
    # School sizes
    school_sizes = [10, 25, 100, 250, 500, 750, 1000, 1500, 2500, 5000]
    vaccination_rates = sorted(data['vax_prop'].unique())
    
    # Create data rows
    rowData = []
    for rate in vaccination_rates:
        rate_percent = int(rate)
        rate_data = data[data['vax_prop'] == rate].iloc[0]
        
        row = {"vaccination_rate": rate_percent}
        
        for size in school_sizes:
            if str(size) in rate_data:
                value = rate_data[str(size)]
                row[f"school_{size}"] = round(value, 1)
            else:
                row[f"school_{size}"] = 0
        
        rowData.append(row)
    
    columnDefs = [
        {"field": "vaccination_rate", "headerName": "Percent Vaccinated", "pinned": "center", "width": 150},
        {
            "headerName": "                        School Enrollment",
            "sortable": False, 
            "children": [
                {
                    "field": f"school_{size}", 
                    "headerName": f"{size}",
                    "sortable": True,
                    "width": 80
                } for size in school_sizes
            ]
        }
    ]
    
    return html.Div([
    dag.AgGrid(
        id="lookup-table",
        rowData=rowData,
        columnDefs=columnDefs,
        columnSize="sizeToFit",
    )

])


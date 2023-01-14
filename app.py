from dash import Dash, dcc, html, Input, Output
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
import warnings

warnings.filterwarnings("ignore")

# Importing income data

df_income = pd.read_csv("./sources/income.csv")

# Importing partners data

df_partners = pd.read_csv("./sources/partners.csv")

# Preprocessing

income_category = df_income[['Formatted Name',
                             'Rent 20% of AMHI',
                             'Rent 50% of AMHI',
                             'Rent 80% of AMHI',
                             'Rent 120% of AMHI'
                            ]]

partners_income_category = df_partners[['Geography',
                                        'Percent HH with income 20% or under of AMHI in core housing need',
                                        'Percent HH with income 21% to 50% of AMHI in core housing need',
                                        'Percent HH with income 51% to 80% of AMHI in core housing need',
                                        'Percent HH with income 81% to 120% of AMHI in core housing need',
                                        'Percent HH with income 121% or more of AMHI in core housing need'
                                        ]]

income_category = income_category.rename(columns = {'Formatted Name': 'Geography'})

joined_df = income_category.merge(partners_income_category, how = 'left', on = 'Geography')

joined_df_filtered = joined_df.query('Geography == "Fraser Valley (CD, BC)"')

x_base =['Very Low Income',
            'Low Income',
            'Moderate Income',
            'Median Income',
            'High Income',
            ]

x_columns = ['Rent 20% of AMHI',
             'Rent 50% of AMHI',
             'Rent 80% of AMHI',
             'Rent 120% of AMHI',
             'Rent 120% of AMHI'
            ]

x_list = []

i = 0
for b, c in zip(x_base, x_columns):
    if i < 4:
        x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
        x_list.append(x)
    else:
        x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
        x_list.append(x)
    i += 1

columns = ['Percent HH with income 20% or under of AMHI in core housing need',
            'Percent HH with income 21% to 50% of AMHI in core housing need',
            'Percent HH with income 51% to 80% of AMHI in core housing need',
            'Percent HH with income 81% to 120% of AMHI in core housing need',
            'Percent HH with income 121% or more of AMHI in core housing need'
            ]

plot_df = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered[columns].T.iloc[:,0].tolist()})

colors = ['#0B4952', '#4A8F97', '#4a5b97', '#FAB88A', '#FFDD5D', ]

fig = go.Figure()
for i, c in zip(plot_df['Income_Category'], colors):
    plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
    fig.add_trace(go.Bar(
        x = plot_df_frag['Income_Category'],
        y = plot_df_frag['Percent HH'],
        name = i,
        marker_color = c,
#         hovertemplate= '%{x} - ' + f'{i}: ' + '%{y:.3s}<extra></extra>'
    ))
fig.update_layout(plot_bgcolor='#f0faff', title = 'Percent HH By Geography', legend_title = "Income")



# Setting layout for dashboard

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(children = [
        html.Div(
        children = [
            html.Div([
                html.H2(children = html.Strong("HART Dashboard"), id = 'home')
            ]),

            # Overview by Scenario

            html.H3(children = html.Strong('Income Level / Percent HH'), id = 'overview-scenario'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph',
                        figure=fig
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),

            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov-download-csv"),
                dcc.Download(id="ov-download-text")
                ], 
                style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
                ),
            ]
            ),

        ], className = 'dashboard'
    ), 
], className = 'background'#style = {'backgroud-color': '#fffced'}
)

# Setting format for hover text

# hovertemplate_bar_x = '%{x},'
# hovertemplate_bar_y = ': %{y:.3s}<extra></extra>'
# hovertemplate_line = '%{x}: %{y:.3s}<extra></extra>'


# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph', 'figure'),
    Input('ov-geo-dropdown', 'value'),
)
def update_geo_figure(geo):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    x_list = []

    i = 0
    for b, c in zip(x_base, x_columns):
        if i < 4:
            x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
            x_list.append(x)
        else:
            x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
            x_list.append(x)
        i += 1

    plot_df = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered[columns].T.iloc[:,0].tolist()})

    colors = ['#0B4952', '#4A8F97', '#4a5b97', '#FAB88A', '#FFDD5D', ]

    fig = go.Figure()
    for i, c in zip(plot_df['Income_Category'], colors):
        plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
        fig.add_trace(go.Bar(
            x = plot_df_frag['Income_Category'],
            y = plot_df_frag['Percent HH'],
            name = i,
            marker_color = c,
    #         hovertemplate= '%{x} - ' + f'{i}: ' + '%{y:.3s}<extra></extra>'
        ))
    fig.update_layout(plot_bgcolor='#f0faff', title = 'Percent HH By Geography', legend_title = "Income")

    
    return fig



# Creating raw csv data file for download option

@app.callback(
    Output("ov-download-text", "data"),
    Input("ov-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# To run Dash surver

if __name__ == "__main__":
    app.run_server(debug=True)





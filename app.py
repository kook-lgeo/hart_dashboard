from dash import Dash, dcc, dash_table, html, Input, Output
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings("ignore")

# Importing income data

engine = create_engine('sqlite:///sources//hart.db')

df_income = pd.read_sql_table('income', engine.connect())
# df_income = pd.read_csv("./sources/income.csv")

# Importing partners data

df_partners = pd.read_sql_table('partners', engine.connect())
#df_partners = pd.read_csv("./sources/partners_small.csv")

# Preprocessing

income_category = df_income.drop(['Geography'], axis=1)

income_category = income_category.rename(columns = {'Formatted Name': 'Geography'})

joined_df = income_category.merge(df_partners, how = 'left', on = 'Geography')

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

table = joined_df_filtered[columns]

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

        # Percent of Households (HHs) in Core Housing Need, by Household Income Category

            html.H3(children = html.Strong('Percent of Households (HHs) in Core Housing Need, by Household Income Category'), id = 'overview-scenario'),

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


        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('Percent of Household Size Categories in Core Housing Need, by AMHI'), id = 'overview-scenario2'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov2-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph2',
                        figure=fig
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),

            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov2-download-csv"),
                dcc.Download(id="ov2-download-text")
                ], 
                style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
                ),
            ]
            ),



        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('Area Median Household Income (AMHI) Categories and Shelter Costs'), id = 'overview-scenario3'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov3-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Graphs

            html.Div(children = [ 
            #     html.Div(
            #         dcc.Graph(
            #             id='graph3',
            #             figure=fig
            #         ),
            #         style={'width': '100%', 'display': 'inline-block'}
            #     ),

                html.Div([
                    dash_table.DataTable(
                        id='datatable-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": True, "selectable": True} for i in table.columns
                        ],
                        data=table.to_dict('records'),
                        editable=True,
                        # filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable="multi",
                        row_selectable="multi",
                        row_deletable=True,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                    ),
                    html.Div(id='datatable-interactivity-container')
                ], style={'padding-top': '30px', 'padding-bottom': '30px'}
                ),


            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov3-download-csv"),
                dcc.Download(id="ov3-download-text")
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
    fig.update_layout(plot_bgcolor='#f0faff', title = 'Percent HH By Income Category', legend_title = "Income")

    
    return fig



# Creating raw csv data file for download option

@app.callback(
    Output("ov-download-text", "data"),
    Input("ov-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph2', 'figure'),
    Input('ov2-geo-dropdown', 'value'),
)
def update_geo_figure2(geo):

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

    hh_p_num_list = [1,2,3,4,'5 or more']
    income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

    h_hold_value = []
    hh_p_num_list_full = []

    for h in hh_p_num_list:
        for i in income_lv_list:
            column = f'Per HH with income {i} of AMHI in core housing need that are {h} person HH'
            h_hold_value.append(joined_df_filtered[column].tolist()[0])
            hh_p_num_list_full.append(h)

    plot_df = pd.DataFrame({'HH_Size': hh_p_num_list_full, 'Income_Category': x_list * 5, 'Percent': h_hold_value})

    colors = ['#FFDD5D', '#FAB88A', '#4a5b97', '#4A8F97', '#0B4952']

    fig2 = go.Figure()

    for h, c in zip(plot_df['HH_Size'].unique(), colors):
        plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
        fig2.add_trace(go.Bar(
            x = plot_df_frag['Income_Category'],
            y = plot_df_frag['Percent'],
            name = h,
            marker_color = c,
            hovertemplate= '%{x}, ' + f'HH Size: {h} - ' + '%{y}<extra></extra>',
        ))
        
    fig2.update_layout(barmode='stack', plot_bgcolor='#f0faff', title = 'Percent HH By Area Mean Household Income', legend_title = "Household Size")

    return fig2



# Creating raw csv data file for download option

@app.callback(
    Output("ov2-download-text", "data"),
    Input("ov2-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# Refreshing Overview by Sectors plots by selected sector

# @app.callback(
#     Output('graph3', 'figure'),
#     Input('ov3-geo-dropdown', 'value'),
# )
# def update_geo_figure3(geo):

#     joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

#     portion_of_total_hh = []
#     for x in x_base:
#         portion_of_total_hh.append(round(joined_df_filtered[f'Percent of Total HH that are in {x}'].tolist()[0] * 100, 2))

#     amhi_range = ['20% or under of AMHI', '21% to 50% of AMHI', '51% to 80% of AMHI', '81% to 120% of AMHI', '121% and more of AMHI']
#     amhi_list = []
#     for a in amhi_range:
#         amhi_list.append(joined_df_filtered[a].tolist()[0])

#     shelter_range = ['20% or under of AMHI.1', '21% to 50% of AMHI.1', '51% to 80% of AMHI.1', '81% to 120% of AMHI.1', '121% and more of AMHI.1']
#     shelter_list = []
#     for s in shelter_range:
#         shelter_list.append(joined_df_filtered[s].tolist()[0])

#     income_ct = [x + f" ({a})" for x, a in zip(x_base, amhi_range)]

#     table = pd.DataFrame({'Area Median Household Income': income_ct, 'Portion of total HHs(%)': portion_of_total_hh , 'Annual Household Income': amhi_list, 'Affordable shelter cost (2015 CAD$)': shelter_list})

#     fig3 = go.Figure(data=[go.Table(
#         header=dict(values=list(table.columns),
#                     fill_color='paleturquoise',
#                     align='center'),
#         cells=dict(values=[table['Area Median Household Income'], 
#                         table['Portion of total HHs(%)'],
#                         table['Annual Household Income'],                      
#                         table['Affordable shelter cost (2015 CAD$)'],                                            
#                         ],
#                 fill_color='lavender',
#                 align='left'
#                 ))
#     ])


#     return fig3


@app.callback(
    Output('datatable-interactivity', 'columns'),
    Output('datatable-interactivity', 'data'),
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('ov3-geo-dropdown', 'value'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_styles(geo, selected_columns):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    portion_of_total_hh = []
    for x in x_base:
        portion_of_total_hh.append(round(joined_df_filtered[f'Percent of Total HH that are in {x}'].tolist()[0] * 100, 2))

    amhi_range = ['20% or under of AMHI', '21% to 50% of AMHI', '51% to 80% of AMHI', '81% to 120% of AMHI', '121% and more of AMHI']
    amhi_list = []
    for a in amhi_range:
        amhi_list.append(joined_df_filtered[a].tolist()[0])

    shelter_range = ['20% or under of AMHI.1', '21% to 50% of AMHI.1', '51% to 80% of AMHI.1', '81% to 120% of AMHI.1', '121% and more of AMHI.1']
    shelter_list = []
    for s in shelter_range:
        shelter_list.append(joined_df_filtered[s].tolist()[0])

    income_ct = [x + f" ({a})" for x, a in zip(x_base, amhi_range)]

    table = pd.DataFrame({'Area Median Household Income': income_ct, 'Portion of total HHs(%)': portion_of_total_hh , 'Annual Household Income': amhi_list, 'Affordable shelter cost (2015 CAD$)': shelter_list})


    return [{"name": i, "id": i, "deletable": True, "selectable": True} for i in table.columns], table.to_dict('record'), [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


# Creating raw csv data file for download option

@app.callback(
    Output("ov3-download-text", "data"),
    Input("ov3-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")







# To run Dash surver

if __name__ == "__main__":
    app.run_server(debug=True)





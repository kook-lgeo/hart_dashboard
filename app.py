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

hh_p_num_list = [1,2,3,4,'5 or more']

amhi_range = ['20% or under of AMHI', '21% to 50% of AMHI', '51% to 80% of AMHI', '81% to 120% of AMHI', '121% and more of AMHI']

income_ct = [x + f" ({a})" for x, a in zip(x_base, amhi_range)]

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
        hovertemplate= '%{x} - ' + '%{y:.3s}<extra></extra>'
    ))
fig.update_layout(plot_bgcolor='#f0faff', title = 'Percent HH By Geography', legend_title = "Income")

table = joined_df_filtered[columns]
table2 = joined_df_filtered[columns]
fig5 = fig
fig6 = fig
fig7 = fig


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

        # Percent of Household Size Categories in Core Housing Need, by Area Median Household Income (AMHI)

            html.H3(children = html.Strong('Area Median Household Income (AMHI) Categories and Shelter Costs'), id = 'overview-scenario3'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov3-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Table

            html.Div(children = [ 

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

            html.H3(children = html.Strong('2016 Affordable Housing Deficit'), id = 'overview-scenario4'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov4-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Table

            html.Div(children = [ 

                html.Div([
                    dash_table.DataTable(
                        id='datatable2-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": True, "selectable": True} for i in table2.columns
                        ],
                        data=table2.to_dict('records'),
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
                    html.Div(id='datatable2-interactivity-container')
                ], style={'padding-top': '30px', 'padding-bottom': '30px'}
                ),


            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov4-download-csv"),
                dcc.Download(id="ov4-download-text")
                ], 
                style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
                ),
            ]
            ),


        # Percentage of Households (HHs) in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population'), id = 'overview-scenario5'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov5-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph5',
                        figure=fig5
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),

            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov5-download-csv"),
                dcc.Download(id="ov5-download-text")
                ], 
                style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
                ),
            ]
            ),


        # Percentage of Households (HHs) in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population and Income'), id = 'overview-scenario6'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov6-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph6',
                        figure=fig6
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),

            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov6-download-csv"),
                dcc.Download(id="ov6-download-text")
                ], 
                style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
                ),
            ]
            ),


        # Percentage of Households (HHs) in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population and HH Size'), id = 'overview-scenario7'),

            # Dropdown for sector selection

            html.Div(children = [
                html.Strong('Select Sector'),
                dcc.Dropdown(joined_df['Geography'].unique(), 'Greater Vancouver (CD, BC)', id='ov7-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph7',
                        figure=fig7
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),

            # Raw data download

                html.Div([
                html.Button("Download Full Raw Data", id="ov7-download-csv"),
                dcc.Download(id="ov7-download-text")
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
    #colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

    fig = go.Figure()
    for i, c in zip(plot_df['Income_Category'], colors):
        plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
        fig.add_trace(go.Bar(
            y = plot_df_frag['Income_Category'],
            x = plot_df_frag['Percent HH'],
            name = i,
            marker_color = c,
            orientation = 'h', 
            hovertemplate= '%{x} - ' + '%{y}<extra></extra>'
        ))
    fig.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor='#f0faff', title = 'Percent HH By Income Category', legend_title = "Income")

    
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

    income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

    h_hold_value = []
    hh_p_num_list_full = []

    for h in hh_p_num_list:
        for i in income_lv_list:
            column = f'Per HH with income {i} of AMHI in core housing need that are {h} person HH'
            h_hold_value.append(joined_df_filtered[column].tolist()[0])
            hh_p_num_list_full.append(h)

    plot_df = pd.DataFrame({'HH_Size': hh_p_num_list_full, 'Income_Category': x_list * 5, 'Percent': h_hold_value})

    # colors = ['#FFDD5D', '#FAB88A', '#4a5b97', '#4A8F97', '#0B4952']
    colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

    fig2 = go.Figure()

    for h, c in zip(plot_df['HH_Size'].unique(), colors):
        plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
        fig2.add_trace(go.Bar(
            y = plot_df_frag['Income_Category'],
            x = plot_df_frag['Percent'],
            name = h,
            marker_color = c,
            orientation = 'h', 
            hovertemplate= '%{x}, ' + f'HH Size: {h} - ' + '%{y}<extra></extra>',
        ))
        
    fig2.update_layout(legend_traceorder = 'normal', yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = 'Percent HH By Area Mean Household Income', legend_title = "Household Size")

    return fig2



# Creating raw csv data file for download option

@app.callback(
    Output("ov2-download-text", "data"),
    Input("ov2-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov2(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('datatable-interactivity', 'columns'),
    Output('datatable-interactivity', 'data'),
    Output('datatable-interactivity', 'style_data_conditional'),
    Input('ov3-geo-dropdown', 'value'),
    Input('datatable-interactivity', 'selected_columns')
)
def update_table1(geo, selected_columns):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    portion_of_total_hh = []
    for x in x_base:
        portion_of_total_hh.append(round(joined_df_filtered[f'Percent of Total HH that are in {x}'].tolist()[0] * 100, 2))

    amhi_list = []
    for a in amhi_range:
        amhi_list.append(joined_df_filtered[a].tolist()[0])

    shelter_range = ['20% or under of AMHI.1', '21% to 50% of AMHI.1', '51% to 80% of AMHI.1', '81% to 120% of AMHI.1', '121% and more of AMHI.1']
    shelter_list = []
    for s in shelter_range:
        shelter_list.append(joined_df_filtered[s].tolist()[0])

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
def func_ov3(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")





# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('datatable2-interactivity', 'columns'),
    Output('datatable2-interactivity', 'data'),
    Output('datatable2-interactivity', 'style_data_conditional'),
    Input('ov4-geo-dropdown', 'value'),
    Input('datatable2-interactivity', 'selected_columns')
)
def update_table2(geo, selected_columns):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    table2 = pd.DataFrame({'Income Group': income_ct})

    h_hold_value = []
    hh_p_num_list = [1,2,3,4,'5 or more']
    income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

    for h in hh_p_num_list:
        h_hold_value = []
        if h == 1:
            h2 = '1 person'
        elif h == '5 or more':
            h2 = '5 or more persons household'
        else:
            h2 = f'{str(h)} persons'
        for i in income_lv_list:
            if i == '20% or under':
                column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of area median household income (AMHI)-Households in core housing need'
                h_hold_value.append(joined_df_filtered[column].tolist()[0])

            else:
                column = f'Total - Private households by presence of at least one or of the combined activity limitations (Q11a, Q11b, Q11c or Q11f or combined)-{h2}-Households with income {i} of AMHI-Households in core housing need'
                h_hold_value.append(joined_df_filtered[column].tolist()[0])
                
        table2[f'{h} p HH'] = h_hold_value

    table2['All HH Sizes'] = table2.sum(axis = 1)

    return [{"name": i, "id": i, "deletable": True, "selectable": True} for i in table2.columns], table2.to_dict('record'), [{
        'if': { 'column_id': i },
        'background_color': '#D2F3FF'
    } for i in selected_columns]


# Creating raw csv data file for download option

@app.callback(
    Output("ov4-download-text", "data"),
    Input("ov4-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov4(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph5', 'figure'),
    Input('ov5-geo-dropdown', 'value'),
)
def update_geo_figure5(geo):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    hh_category_dict = {
                    'Percent Women-led HH in core housing need' : 'Women-led HH', 
                    'Percent Single Mother led HH in core housing need' : 'Single mother-led HH', 
                    'Percent Indigenous HH in core housing need' : 'Indigenous HH', 
                    'Percent Visible minority HH in core housing need' : 'Visible minority HH', 
                    'Percent Black-led HH in core housing need' : 'Black-led HH', 
                    'Percent New migrant-led HH in core housing need' : 'New migrant-led HH', 
                    'Percent Refugee claimant-led HH in core housing need' : 'Refugee claimant-led HH', 
                    'Percent HH head under 25 in core housing need' : 'HH head under 25', 
                    'Percent HH head over 65 in core housing need' : 'HH head over 65', 
                    'Percent HH head over 85 in core housing need' : 'HH head over 85', 
                    'Percent HH with physical act. limit. in core housing need' : 'HH with physical activity limitation', 
                    'Percent HH with cognitive, mental, or addictions activity limitation in core housing need' : 'HH with cognitive, mental, or addictions activity limitation', 
                    'Percent HH in core housing need' : 'Community (all HH)'
                   }

    columns = hh_category_dict.keys()
    hh_categories = list(hh_category_dict.values())

    percent_hh = [joined_df_filtered[c].tolist()[0] for c in columns]

    plot_df = pd.DataFrame({'HH_Category': hh_categories, 'Percent_HH': percent_hh})

    colors = ['#fff194','#4A8F97', '#210b52', '#0B4952', '#FFDD5D', '#158232', '#4a5b97', '#6ed0db', '#bfd5ff', '#ff8d3d', '#166370', '#FAB88A',  '#ffe28f']

    fig5 = go.Figure()
    for i, c in zip(hh_categories, colors):
        plot_df_frag = plot_df.loc[plot_df['HH_Category'] == i, :]
        fig5.add_trace(go.Bar(
            y = plot_df_frag['HH_Category'],
            x = plot_df_frag['Percent_HH'],
            name = i,
            marker_color = c,
            orientation = 'h', 
            hovertemplate= '%{x} - ' + '%{y}<extra></extra>',
            
        ))
    fig5.update_layout(yaxis=dict(autorange="reversed"), showlegend = True, plot_bgcolor='#f0faff', title = 'Percentage of Households (HHs) in Core Housing Need by Priority Population', legend_title = "HH Category")

    return fig5



# Creating raw csv data file for download option

@app.callback(
    Output("ov5-download-text", "data"),
    Input("ov5-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov5(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph6', 'figure'),
    Input('ov6-geo-dropdown', 'value'),
)
def update_geo_figure6(geo):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    hh_category_dict2 = {
                        'Percent of Women-led HH in core housing need' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing need' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing need' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing need' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing need' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing need' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing need' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing need' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing need' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing need' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing need' : 'HH with physical activity limitation', 
                        'Percent of HH with with cognitive, mental, or addictions activity limitation in core housing need' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    hh_category_dict3 = {
                        'Percent of Women-led HH core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation', 
                        }

    hh_category_dict4 = {
                        'Percent of Women-led HH in core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    columns2 = hh_category_dict2.keys()
    columns3 = hh_category_dict3.keys()
    columns4 = hh_category_dict4.keys()

    income_col = []
    percent_col = []
    hh_cat_col = []

    income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

    for c, c2, c3 in zip(columns2, columns3, columns4):
        for i in income_lv_list:
            if i == '20% or under':
                p_hh = joined_df_filtered[f'{c} with income {i} of the AMHI'].tolist()[0]
                hh_cat_col.append(hh_category_dict2[c])
            elif i == '21% to 50%':
                p_hh = joined_df_filtered[f'{c2} with income {i} of AMHI'].tolist()[0]
                hh_cat_col.append(hh_category_dict3[c2])
            else:
                p_hh = joined_df_filtered[f'{c3} with income {i} of AMHI'].tolist()[0]
                hh_cat_col.append(hh_category_dict4[c3])
                
            income_col.append(i)
            percent_col.append(p_hh)

    plot_df = pd.DataFrame({'Income_Category': income_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

    fig6 = go.Figure()

    colors = ['#bfd5ff', '#4A8F97', '#4a5b97', '#0B4952', '#210b52']

    for i, c in zip(plot_df['Income_Category'].unique(), colors):
        plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
        fig6.add_trace(go.Bar(
            y = plot_df_frag['HH_Category'],
            x = plot_df_frag['Percent'],
            name = i,
            marker_color = c,
            orientation = 'h', 
            hovertemplate= '%{y}, ' + f'Income Level: {i} - ' + '%{x}<extra></extra>',
        ))
        
    fig6.update_layout(legend_traceorder="normal", yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = 'Percentage of Households (HHs) in Core Housing Need by Priority Population and Income', legend_title = "Income Category")

    return fig6



# Creating raw csv data file for download option

@app.callback(
    Output("ov6-download-text", "data"),
    Input("ov6-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov6(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# Refreshing Overview by Sectors plots by selected sector

@app.callback(
    Output('graph7', 'figure'),
    Input('ov7-geo-dropdown', 'value'),
)
def update_geo_figure7(geo):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    hh_category_dict2 = {
                        'Percent of Women-led HH in core housing need' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing need' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing need' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing need' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing need' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing need' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing need' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing need' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing need' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing need' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing need' : 'HH with physical activity limitation', 
                        'Percent of HH with with cognitive, mental, or addictions activity limitation in core housing need' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    hh_category_dict3 = {
                        'Percent of Women-led HH core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation', 
                        }

    hh_category_dict4 = {
                        'Percent of Women-led HH in core housing' : 'Women-led HH', 
                        'Percent of Single Mother led HH in core housing' : 'Single mother-led HH', 
                        'Percent of Indigenous HH in core housing' : 'Indigenous HH', 
                        'Percent of Visible minority HH in core housing' : 'Visible minority HH', 
                        'Percent of Black-led HH in core housing' : 'Black-led HH', 
                        'Percent of New migrant-led HH in core housing' : 'New migrant-led HH', 
                        'Percent of Refugee claimant-led HH in core housing' : 'Refugee claimant-led HH', 
                        'Percent of HH head under 25 in core housing' : 'HH head under 25', 
                        'Percent of HH head over 65 in core housing' : 'HH head over 65', 
                        'Percent of HH head over 85 in core housing' : 'HH head over 85', 
                        'Percent of HH with physical act. limit. in core housing' : 'HH with physical activity limitation', 
                        'Percent of HH with cognitive, mental, or addictions activity limitation in core housing' : 'HH with cognitive, mental, or addictions activity limitation',
                        }

    columns2 = hh_category_dict2.keys()
    columns3 = hh_category_dict3.keys()
    columns4 = hh_category_dict4.keys()

    hh_size_col = []
    percent_col = []
    hh_cat_col = []

    hh_size = ['1 person', '2 people', '3 people', '4 people', '5 or more people']

    for c, c2, c3 in zip(columns2, columns3, columns4):
        for h in hh_size:
            if h == '1 person':
                p_hh = joined_df_filtered[f'{c} with a HH size of {h}'].tolist()[0]
                hh_cat_col.append(hh_category_dict2[c])
            elif h == '2 people':
                p_hh = joined_df_filtered[f'{c2} with a HH size of {h}'].tolist()[0]
                hh_cat_col.append(hh_category_dict3[c2])
            elif h == '5 or more people':
                p_hh = joined_df_filtered[f'{c3}with a HH size of {h}'].tolist()[0]
                hh_cat_col.append(hh_category_dict4[c3])
            else:
                p_hh = joined_df_filtered[f'{c3} with a HH size of {h}'].tolist()[0]
                hh_cat_col.append(hh_category_dict4[c3])
                
            hh_size_col.append(h)
            percent_col.append(p_hh)

    plot_df = pd.DataFrame({'HH_Size': hh_size_col, 'HH_Category': hh_cat_col, 'Percent': percent_col})

    fig7 = go.Figure()

    colors = ['#fff194', '#FFDD5D', '#FAB88A', '#ff8d3d', '#fc7b5d']

    for h, c in zip(plot_df['HH_Size'].unique(), colors):
        plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
        fig7.add_trace(go.Bar(
            y = plot_df_frag['HH_Category'],
            x = plot_df_frag['Percent'],
            name = h,
            marker_color = c,
            orientation = 'h', 
            hovertemplate= '%{y}, ' + f'Income Level: {h} - ' + '%{x}<extra></extra>',
        ))
            
    fig7.update_layout(legend_traceorder="normal", yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#f0faff', title = 'Percentage of Households (HHs) in Core Housing Need by Priority Population and HH Size', legend_title = "HH Size")

    return fig7



# Creating raw csv data file for download option

@app.callback(
    Output("ov7-download-text", "data"),
    Input("ov7-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov7(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")



# To run Dash surver

if __name__ == "__main__":
    app.run_server(debug=True)





from dash import Dash, dcc, dash_table, html, Input, Output, ctx, callback
import dash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.scatter.marker import Line
from dash.dash_table.Format import Format, Scheme, Sign, Symbol, Group
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import warnings
import json
import geopandas as gpd
import fiona
# from app import app
# from pages import projection
fiona.supported_drivers  


warnings.filterwarnings("ignore")

# Importing income data

engine = create_engine('sqlite:///sources//hart.db')

df_income = pd.read_sql_table('income', engine.connect())
# df_income = pd.read_csv("./sources/income.csv")

# Importing partners data

df_partners = pd.read_sql_table('partners', engine.connect())
#df_partners = pd.read_csv("./sources/partners_small.csv")

# Importing Geo Code Information

mapped_geo_code = pd.read_sql_table('geocodes_integrated', engine.connect())
df_geo_list = pd.read_sql_table('geocodes', engine.connect())
df_region_list = pd.read_sql_table('regioncodes', engine.connect())
df_region_list.columns = df_geo_list.columns
df_province_list = pd.read_sql_table('provincecodes', engine.connect())
df_province_list.columns = df_geo_list.columns

# Importing Projection Data

df_csd_proj = pd.read_sql_table('csd_hh_projections', engine.connect())
df_cd_proj = pd.read_sql_table('cd_hh_projections', engine.connect())
df_cd_grow = pd.read_sql_table('cd_growthrates', engine.connect())

# Merging Projection data with Geography codes

df_csd_proj_merged = df_geo_list.merge(df_csd_proj, how = 'left', on = 'Geo_Code')
df_cd_proj_merged = df_region_list.merge(df_cd_proj, how = 'left', on = 'Geo_Code')
df_cd_grow_merged = df_region_list.merge(df_cd_grow, how = 'left', on = 'Geo_Code')

# Importing Province Boundaries shape data


gdf_p_code_added = gpd.read_file('./sources/mapdata_simplified/province.shp')
gdf_p_code_added = gdf_p_code_added.set_index('Geo_Code')

# Importing subregions which don't have data

not_avail = pd.read_csv('not_in_list.csv')
not_avail['CSDUID'] = not_avail['CSDUID'].astype(str)

# Configuration for plot icons

config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['zoom', 'lasso2d', 'pan', 'select', 'zoomIn', 'zoomOut', 'autoScale',]}

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

# colors = ['#D7F3FD', '#B0E6FC', '#88D9FA', '#61CDF9', '#39C0F7']
# hh_colors = ['#D8EBD4', '#B7DCAE', '#93CD8A', '#6EC067', '#3DB54A']
colors = ['#D7F3FD', '#88D9FA', '#39C0F7', '#099DD7', '#044762']
hh_colors = ['#D8EBD4', '#93CD8A', '#3DB54A', '#297A32', '#143D19']
hh_type_color = ['#3949CE', '#3EB549', '#39C0F7']
columns_color_fill = ['#F3F4F5', '#EBF9FE', '#F0FAF1']
map_colors_wo_black = ['#39C0F7', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#F4F739']
map_colors_w_black = ['#000000', '#39C0F7', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#F4F739']
modebar_color = '#099DD7'
modebar_activecolor = '#044762'

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
fig.update_layout(plot_bgcolor='#F8F9F9', title = 'Percent HH By Geography', legend_title = "Income")

table = joined_df_filtered[['Rent 20% of AMHI', 'Rent 50% of AMHI']]
table2 = joined_df_filtered[['Rent 20% of AMHI', 'Rent 50% of AMHI']]
fig5 = fig
fig6 = fig
fig7 = fig


gdf_p_code_added["rand"] = np.random.randint(1, 100, len(gdf_p_code_added))

fig_m = go.Figure()

fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                locations = gdf_p_code_added.index, 
                                z = gdf_p_code_added.rand, 
                                showscale = False, 
                                hovertext= gdf_p_code_added.NAME,
                                marker = dict(opacity = 0.4),
                                marker_line_width=.5))


fig_m.update_layout(mapbox_style="carto-positron",
                mapbox_center = {"lat": gdf_p_code_added.geometry.centroid.y.mean()+10, "lon": gdf_p_code_added.geometry.centroid.x.mean()},
                height = 500,
                width = 1000,
                mapbox_zoom = 1.4,
                autosize=True)



# Setting layout for dashboard

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# dash.register_page(__name__)

layout = html.Div(children = [
        dcc.Store(id='main-area', storage_type='local'),
        dcc.Store(id='comparison-area', storage_type='local'),
        html.Div(
        children = [
            html.Div([
                html.H2(children = html.Strong("Main Page"), id = 'home')
            ]),

            # Area Scale Selection

            html.H3(children = html.Strong('Census Geography Area Selection'), id = 'area-scale'),

            html.Div(children = [ 

                html.Div(children = [                     
                    html.Button('View Census Subdivision (CSD)', id='to-geography-1', n_clicks=0),     
                                    ], className = 'region_button'
                    ),           
                html.Div(children = [ 
                    html.Button('View Census Division (CD)', id='to-region-1', n_clicks=0),
                                    ], className = 'region_button'
                    ),         
                html.Div(children = [ 
                    html.Button('View Province', id='to-province-1', n_clicks=0),
                                    ], className = 'region_button'
                    ),         
                ], 
                style={'width': '100%', 'display': 'inline-block', 'padding-bottom': '20px', 'padding-top': '10px'}
            ),


        # Area Median Household Income (AMHI) Categories and Shelter Costs

            html.H3(children = html.Strong('Area Median Household Income (AMHI) Categories and Shelter Costs'), id = 'overview-scenario3'),

            # Table

            html.Div([
                dash_table.DataTable(
                    id='datatable-interactivity',
                    columns=[
                        {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                    ],
                    data=table.to_dict('records'),
                    editable=True,
                    # filter_action="native",
                    sort_action="native",
                    sort_mode="multi",
                    column_selectable=False,#"multi",
                    row_selectable=False,#"multi",
                    row_deletable=False,
                    selected_columns=[],
                    selected_rows=[],
                    page_action="native",
                    page_current= 0,
                    page_size= 10,
                    merge_duplicate_headers=True,
                    export_format = "csv",
                    style_cell = {'font-family': 'Bahnschrift'},
                    # style_data = {'font_size': '1.0rem', 'width': '100px'},
                    style_header = {'text-align': 'middle', 'fontWeight': 'bold'}#{'whiteSpace': 'normal', 'font_size': '1.0rem'}
                ),
                html.Div(id='datatable-interactivity-container')
            ], style={'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px', 'display': 'block'}
            ),



        # Percent of Households (HHs) in Core Housing Need, by Household Income Category

            html.H3(children = html.Strong('Percent of Households (HHs) in Core Housing Need, by Household Income Category'), id = 'overview-scenario'),

            # Graph

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph',
                        figure=fig,
                        config = config,
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),


        # Percent of Household Size Categories in Core Housing Need, by AMHI

            html.H3(children = html.Strong('Percent of Household Size Categories in Core Housing Need, by AMHI'), id = 'overview-scenario2'),

            # Graph

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph2',
                        figure=fig,
                        config = config,
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),

        # 2016 Affordable Housing Deficit

            html.H3(children = html.Strong('2016 Affordable Housing Deficit'), id = 'overview-scenario4'),

            # Table
            
            html.Div(children = [ 

                html.Div([
                    dash_table.DataTable(
                        id='datatable2-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": False} for i in table2.columns
                        ],
                        data=table2.to_dict('records'),
                        editable=True,
                        # filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable=False,#"multi",
                        row_selectable=False,#"multi",
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        merge_duplicate_headers=True,
                        # style_table={'minWidth': '100%'},
                        style_cell = {'font-family': 'Bahnschrift'},
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'},
                        export_format = "csv"
                    ),
                    html.Div(id='datatable2-interactivity-container')
                ], style={'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px'}
                ),

            ],style={'width': '80%'}
            ),


        # Percentage of HHs in Core Housing Need by Priority Population

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population'), id = 'overview-scenario5'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph5',
                        figure=fig5,
                        config = config,
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),


        # Percentage of HHs in Core Housing Need by Priority Population and Income

            html.H3(children = html.Strong('Percentage of HHs in Core Housing Need by Priority Population and Income'), id = 'overview-scenario6'),

            # Graphs

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='graph6',
                        figure=fig6,
                        config = config,
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),

            # Raw data download

            html.Div([
            html.Button("Download This Community", id="ov7-download-csv"),
            dcc.Download(id="ov7-download-text")
            ], 
            style={'width': '12%', 'display': 'inline-block', 'padding-bottom': '50px'}
            ),


        ], className = 'dashboard'
    ), 
], className = 'background'#style = {'backgroud-color': '#fffced'}
)


# Area Median Household Income (AMHI) Categories and Shelter Costs

def table_amhi_shelter_cost(geo, IsComparison):

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

    if IsComparison != True:
        table = pd.DataFrame({'Area Median HH Income': income_ct, '% of Total HHs': portion_of_total_hh , 'Annual Household Income': amhi_list, 'Affordable shelter cost (2015 CAD$)': shelter_list})
        table['% of Total HHs'] = table['% of Total HHs'].astype(str) + '%'
    else:
        table = pd.DataFrame({'Area Median HH Income': income_ct, '% of Total HHs ': portion_of_total_hh , 'Annual HH Income ': amhi_list, 'Affordable Shelter Cost ': shelter_list})
        table['% of Total HHs '] = table['% of Total HHs '].astype(str) + '%'

    return table

@callback(
    Output('datatable-interactivity', 'columns'),
    Output('datatable-interactivity', 'data'),
    Output('datatable-interactivity', 'style_data_conditional'),
    Output('datatable-interactivity', 'style_cell_conditional'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('datatable-interactivity', 'selected_columns'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_table1(geo, geo_c, selected_columns, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        table = table_amhi_shelter_cost(geo, IsComparison = False)
    
        col_list = []

        for i in table.columns:
            col_list.append({"name": [geo, i], "id": i})
        
        # print(col_list)

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table.columns[1:]
        ] + [
            {
                'if': {'column_id': table.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional
        
    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        table = table_amhi_shelter_cost(geo, IsComparison = False)

        col_list = []

        for i in table.columns:
            if i == 'Area Median HH Income':
                col_list.append({"name": ["Income Category", i], "id": i})
            else:
                col_list.append({"name": [geo, i], "id": i})

        # Comparison

        if geo_c == None:
            geo_c = geo

        table_c = table_amhi_shelter_cost(geo_c, IsComparison = True)

        table_j = table.merge(table_c, how = 'left', on = 'Area Median HH Income')

        for i in table_c.columns[1:]:
            col_list.append({"name": [geo_c, i], "id": i})

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional




# Percent of Households (HHs) in Core Housing Need, by Household Income Category

def plot_df_core_housing_need_by_income(geo, IsComparison):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    x_list = []

    i = 0
    for b, c in zip(x_base, x_columns):
        if i < 4:
            if IsComparison != True:
                x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
            else:
                x = " ($" + str(joined_df_filtered[c].tolist()[0]) + ") "
            x_list.append(x)
        else:
            if IsComparison != True:
                x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
            else:
                x = " (>$" + str(joined_df_filtered[c].tolist()[0]) + ") "
            x_list.append(x)
        i += 1

    plot_df = pd.DataFrame({'Income_Category': x_list, 'Percent HH': joined_df_filtered[columns].T.iloc[:,0].tolist()})

    return plot_df

@callback(
    Output('graph', 'figure'),
    # Input('all-geo-dropdown', 'value'),
    # Input('comparison-geo-dropdown', 'value'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure(geo, geo_c, btn1, btn2, btn3):
    # print(geo, geo_c)
    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_income(geo, IsComparison = False)

        fig = go.Figure()
        for i, c in zip(plot_df['Income_Category'], colors):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent HH'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>'
            ))

        fig.update_layout(yaxis=dict(autorange="reversed"), modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, plot_bgcolor='#F8F9F9', title = f'Percent HH By Income Category - {geo}', legend_title = "Income")
        fig.update_xaxes(range = [0, 1])
        fig.update_yaxes(title = 'Income Categories<br>and Max. affordable shelter costs')

        return fig


    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]


        plot_df = plot_df_core_housing_need_by_income(geo, IsComparison = False)

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_xaxes=True)

        n = 0
        for i, c, b in zip(plot_df['Income_Category'], colors, x_base):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent HH'],
                name = b.replace(" Income", ""),
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}'
            ), row = 1, col = 1)
            n += 1

        fig.update_yaxes(title = 'Income Categories<br>and Max. affordable shelter costs', row = 1, col = 1)

        # Comparison plot

        plot_df_c = plot_df_core_housing_need_by_income(geo_c, IsComparison = True)

        n = 0
        for i, c, b in zip(plot_df_c['Income_Category'], colors, x_base):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['Income_Category'] == i, :]
            fig.add_trace(go.Bar(
                y = plot_df_frag_c['Income_Category'],
                x = plot_df_frag_c['Percent HH'],
                name = b.replace(" Income", ""),
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False,
            ), row = 1, col = 2)
            n += 1


        fig.update_layout(title = f'Percent HH By Income Category', modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, plot_bgcolor='#F8F9F9', legend_title = "Income", legend = dict(font = dict(size = 9)))
        fig.update_yaxes(tickfont = dict(size = 9.5), autorange = "reversed")
        fig.update_xaxes(range = [0, 1])

        return fig



# Percent of Household Size Categories in Core Housing Need, by AMHI

def plot_df_core_housing_need_by_amhi(geo, IsComparison):
    
    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    x_list = []

    i = 0
    for b, c in zip(x_base, x_columns):
        if i < 4:
            if IsComparison != True:
                x = b + " ($" + str(joined_df_filtered[c].tolist()[0]) + ")"
            else:
                x = " ($" + str(joined_df_filtered[c].tolist()[0]) + ") "
            x_list.append(x)
        else:
            if IsComparison != True:
                x = b + " (>$" + str(joined_df_filtered[c].tolist()[0]) + ")"
            else:
                x = " (>$" + str(joined_df_filtered[c].tolist()[0]) + ") "
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
    
    return plot_df


@callback(
    Output('graph2', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure2(geo, geo_c, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_amhi(geo, False)

        fig2 = go.Figure()

        for h, c in zip(plot_df['HH_Size'].unique(), hh_colors):
            plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
            fig2.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'HH Size: {h} - ' + '%{x: .2%}<extra></extra>',
            ))
            
        fig2.update_layout(legend_traceorder = 'normal', modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#F8F9F9', title = f'Percent HH By Income Category and AMHI - {geo}', legend_title = "Household Size")
        fig2.update_yaxes(title = 'Income Categories<br>and Max. affordable shelter costs')

        return fig2

    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_amhi(geo, False)

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_xaxes=True)

        n = 0
        for h, c in zip(plot_df['HH_Size'].unique(), hh_colors):
            plot_df_frag = plot_df.loc[plot_df['HH_Size'] == h, :]
            fig2.add_trace(go.Bar(
                y = plot_df_frag['Income_Category'],
                x = plot_df_frag['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'HH Size: {h} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}'
            ), row = 1, col = 1)
            n += 1
        
        fig2.update_yaxes(title = 'Income Categories<br>and Max. affordable shelter costs', row = 1, col = 1)


        # Comparison plot

        plot_df_c = plot_df_core_housing_need_by_amhi(geo_c, True)

        n = 0
        for h, c in zip(plot_df_c['HH_Size'].unique(), hh_colors):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['HH_Size'] == h, :]
            fig2.add_trace(go.Bar(
                y = plot_df_frag_c['Income_Category'],
                x = plot_df_frag_c['Percent'],
                name = h,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'HH Size: {h} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False,
            ), row = 1, col = 2)
            n += 1

        fig2.update_layout(legend_traceorder = 'normal', modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Percent HH By Income Category and AMHI', legend_title = "Household Size", legend = dict(font = dict(size = 9)))
        fig2.update_yaxes(tickfont = dict(size = 9.5), autorange = "reversed")

        return fig2



# 2016 Affordable Housing Deficit

def table_core_affordable_housing_deficit(geo, IsComparison):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    table2 = pd.DataFrame({'Area Median HH Income': income_ct})

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
                
        if IsComparison != True:
            if h == 1:        
                table2[f'{h} Person HH'] = h_hold_value
            elif h == '5 or more':
                table2[f'5 >= People HH'] = h_hold_value
            else:
                table2[f'{h} People HH'] = h_hold_value
                
            table2['Total'] = table2.sum(axis = 1)
            
        else:
            if h == 1:        
                table2[f'{h} Person HH '] = h_hold_value
            elif h == '5 or more':
                table2[f'5 >= People HH '] = h_hold_value
            else:
                table2[f'{h} People HH '] = h_hold_value
                
            table2['Total '] = table2.sum(axis = 1)    
    
    return table2


@callback(
    Output('datatable2-interactivity', 'columns'),
    Output('datatable2-interactivity', 'data'),
    Output('datatable2-interactivity', 'style_data_conditional'),
    Output('datatable2-interactivity', 'style_cell_conditional'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('datatable2-interactivity', 'selected_columns'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_table2(geo, geo_c, selected_columns, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        table2 = table_core_affordable_housing_deficit(geo, False)
        table2 = table2[['Area Median HH Income', '1 Person HH', '2 People HH',
                        '3 People HH', '4 People HH', '5 >= People HH', 'Total']]

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table2.columns[1:]
        ] + [
            {
                'if': {'column_id': table2.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table2.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table2.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional


    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        table2 = table_core_affordable_housing_deficit(geo, False)
        table2 = table2[['Area Median HH Income', '1 Person HH', '2 People HH',
                        '3 People HH', '4 People HH', '5 >= People HH', 'Total']]

        col_list = []

        for i in table2.columns:
            if i == 'Area Median HH Income':
                col_list.append({"name": ["Income Category", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Table

        if geo == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'

        table2_c = table_core_affordable_housing_deficit(geo_c, True)
        table2_c = table2_c[['Area Median HH Income', '1 Person HH ', '2 People HH ',
                        '3 People HH ', '4 People HH ', '5 >= People HH ', 'Total ']]

        for i in table2_c.columns[1:]:
            if i == 'Area Median HH Income':
                col_list.append({"name": ["Income Category", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        table2_j = table2.merge(table2_c, how = 'left', on = 'Area Median HH Income')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table2.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table2_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table2.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table2_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional



# Percentage of HHs in Core Housing Need by Priority Population

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

hh_columns = hh_category_dict.keys()
hh_categories = list(hh_category_dict.values())

def plot_df_core_housing_need_by_priority_population(geo):

    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    percent_hh = [joined_df_filtered[c].tolist()[0] for c in hh_columns]

    plot_df = pd.DataFrame({'HH_Category': hh_categories, 'Percent_HH': percent_hh})
    plot_df['Percent_HH'] = plot_df['Percent_HH'].fillna(0)
    
    return plot_df

def color_dict_core_housing_need_by_priority_population(plot_df):

    color_dict = {}

    for h in plot_df['HH_Category'].unique():

        if plot_df['Percent_HH'].max() == 0:
            color_dict[h] = hh_type_color[2]
        else:
            if h == plot_df.loc[plot_df['Percent_HH'] == plot_df['Percent_HH'].max(), 'HH_Category'].tolist()[0]:
                    color_dict[h] = hh_type_color[0]
            elif h == 'Community (all HH)':
                color_dict[h] = hh_type_color[1]
            else:
                color_dict[h] = hh_type_color[2]
    
    return color_dict

@callback(
    Output('graph5', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure5(geo, geo_c, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_priority_population(geo)
        color_dict = color_dict_core_housing_need_by_priority_population(plot_df)

        fig5 = go.Figure()
        for i in hh_categories:
            plot_df_frag = plot_df.loc[plot_df['HH_Category'] == i, :]
            fig5.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent_HH'],
                name = i,
                marker_color = color_dict[i],
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                
            ))
        fig5.update_layout(yaxis=dict(autorange="reversed"), modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, showlegend = False, plot_bgcolor='#F8F9F9', title = f'Percentage of HHs in Core Housing Need by Priority Population - {geo}', legend_title = "HH Category")

        return fig5

    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_priority_population(geo)
        color_dict = color_dict_core_housing_need_by_priority_population(plot_df)

        fig5 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i in hh_categories:
            plot_df_frag = plot_df.loc[plot_df['HH_Category'] == i, :]
            fig5.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent_HH'],
                name = i,
                marker_color = color_dict[i],
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                
            ),row = 1, col = 1)

        # Comparison Plot

        plot_df_c = plot_df_core_housing_need_by_priority_population(geo_c)
        color_dict = color_dict_core_housing_need_by_priority_population(plot_df_c)

        for i in hh_categories:
            plot_df_frag_c = plot_df_c.loc[plot_df_c['HH_Category'] == i, :]
            fig5.add_trace(go.Bar(
                y = plot_df_frag_c['HH_Category'],
                x = plot_df_frag_c['Percent_HH'],
                name = i,
                marker_color = color_dict[i],
                orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x: .2%}<extra></extra>',
                
            ),row = 1, col = 2)
        fig5.update_layout(yaxis=dict(autorange="reversed"), modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, showlegend = False, plot_bgcolor='#F8F9F9', title = f'Percentage of HHs in Core Housing Need by Priority Population', legend_title = "HH Category")
        fig5.update_xaxes(range=[0, max(plot_df['Percent_HH'].max(), plot_df_c['Percent_HH'].max())])

        return fig5


# Percentage of HHs in Core Housing Need by Priority Population and Income

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

income_lv_list = ['20% or under', '21% to 50%', '51% to 80%', '81% to 120%', '121% or more']

def plot_df_core_housing_need_by_priority_population_income(geo):
    
    joined_df_filtered = joined_df.query('Geography == '+ f'"{geo}"')

    income_col = []
    percent_col = []
    hh_cat_col = []

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
    
    return plot_df


@callback(
    Output('graph6', 'figure'),
    Input('main-area', 'data'),
    Input('comparison-area', 'data'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure6(geo, geo_c, btn1, btn2, btn3):

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None):

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'


        if "to-geography-1" == ctx.triggered_id:
            geo = geo
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_priority_population_income(geo)
        
        fig6 = go.Figure()

        for i, c in zip(plot_df['Income_Category'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig6.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {i} - ' + '%{x: .2%}<extra></extra>',
            ))
            
        fig6.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#F8F9F9', title = f'Percentage of HHs in Core Housing Need by Priority Population and Income - {geo}', legend_title = "Income Category")

        return fig6

    else:

        if "to-geography-1" == ctx.triggered_id:
            geo = geo
            geo_c = geo_c
        elif "to-region-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Region'].tolist()[0]
        elif "to-province-1" == ctx.triggered_id:
            geo = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Province'].tolist()[0]
            geo_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c,:]['Province'].tolist()[0]

        plot_df = plot_df_core_housing_need_by_priority_population_income(geo)

        fig6 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        n = 0
        for i, c in zip(plot_df['Income_Category'].unique(), colors):
            plot_df_frag = plot_df.loc[plot_df['Income_Category'] == i, :]
            fig6.add_trace(go.Bar(
                y = plot_df_frag['HH_Category'],
                x = plot_df_frag['Percent'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate= '%{y}, ' + f'Income Level: {i} - ' + '%{x: .2%}<extra></extra>',
                legendgroup= f'{n}'
            ), row = 1, col = 1)
            n += 1
            
        # Comparison plot

        plot_df_c = plot_df_core_housing_need_by_priority_population_income(geo_c)

        n = 0
        for i, c in zip(plot_df_c['Income_Category'].unique(), colors):
            plot_df_frag_c = plot_df_c.loc[plot_df_c['Income_Category'] == i, :]
            fig6.add_trace(go.Bar(
                y = plot_df_frag_c['HH_Category'],
                x = plot_df_frag_c['Percent'],
                name = i,
                marker_color = c,
                orientation = 'h', 
                hovertemplate = '%{y}, ' + f'Income Level: {i} - ' + '%{x: .2%}<extra></extra>',
                legendgroup = f'{n}',
                showlegend = False
            ), row = 1, col = 2)
            n += 1
            
        fig6.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, yaxis=dict(autorange="reversed"), barmode='stack', plot_bgcolor='#F8F9F9', title = f'Percentage of HHs in Core Housing Need by Priority Population and Income', legend_title = "Income Category")

        return fig6



# Creating raw csv data file for download option

@callback(
    Output("ov7-download-text", "data"),
    Input("ov7-download-csv", "n_clicks"),
    prevent_initial_call=True,
)
def func_ov7(n_clicks):
    return dcc.send_data_frame(joined_df.to_csv, "result.csv")

from dash import Dash, dcc, dash_table, html, Input, Output, ctx
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

# New projection data
updated_csd = pd.read_csv('./sources/updated_csd.csv')

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
app = dash.Dash(__name__)#, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(children = [
        html.Div(
        children = [
            html.Div([
                html.H2(children = html.Strong("HART Dashboard"), id = 'home')
            ]),

            # Dropdown for sector selection
            html.H3(children = html.Strong('Area Selection'), id = 'area-selection'),

            # Reset Button for Map

            html.Div(children = [ 

                html.Div(children = [                     
                    html.Button('Reset Map', id='reset-map', n_clicks=0),     
                                    ], className = 'region_button'
                    ),                
                ], 
                style={'width': '55%', 'display': 'inline-block', 'padding-bottom': '20px', 'padding-top': '10px'}
            ),

            # Map picker

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='canada_map',
                        figure=fig_m,
                        config = config,
                    ),
                    style={'width': '100%', 'display': 'inline-block'}
                ),
            ]
            ),

            html.Div(
                id = 'all-geo-dropdown-parent',
                children = [
                html.Strong('Select Area'),
                dcc.Dropdown(joined_df['Geography'].unique()[1:], 'Greater Vancouver A RDA (CSD, BC)', id='all-geo-dropdown'),
                # dcc.Dropdown(df_geo_list['Geography'].unique(), 'Greater Vancouver A RDA (CSD, BC)', id='all-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '20px', 'padding-top': '20px'}
            ),

            html.Div(
                id = 'comparison-geo-dropdown-parent',
                children = [
                html.Strong('Comparison Area'),
                dcc.Dropdown(joined_df['Geography'].unique()[1:], id='comparison-geo-dropdown'),
                # dcc.Dropdown(df_geo_list['Geography'].unique(), id='comparison-geo-dropdown'),
                ], 
                style={'width': '20%', 'display': 'inline-block', 'padding-right': '30px', 'padding-bottom': '10px', 'padding-top': '20px'}
            ),

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


        # 2026 Projections by HH Size and Income Level

            html.H3(children = html.Strong('2026 Projections by HH Size and Income Level'), id = 'overview-scenario8'),

            # Table

            html.Div([
                html.Div([

                    html.H5(children = html.Strong('Community 2026 HH'), className = 'table-title'),

                    dash_table.DataTable(
                        id='datatable3-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                        ],
                        data=table.to_dict('records'),
                        editable=True,
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable=False,
                        row_selectable=False,
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        merge_duplicate_headers=True,
                        style_cell = {'font-family': 'Bahnschrift'},
                        export_format="csv",
                        # style_table={'minWidth': '100%'},
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable3-interactivity-container'),

                    # Graphs

                    html.Div(children = [ 
                        html.Div(
                            dcc.Graph(
                                id='graph7',
                                figure=fig5,
                                config = config,
                            ),
                            style={'width': '100%', 'display': 'inline-block'}
                        ),
                    ]
                    ),

                ], className = 'tables'),

                html.Div([
                   
                    html.H5(children = html.Strong('Community 2026 HH (Regional Rates)'), className = 'table-title'),

                    dash_table.DataTable(
                        id='datatable4-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                        ],
                        data=table.to_dict('records'),
                        editable=True,
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable=False,
                        row_selectable=False,
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        style_cell = {'font-family': 'Bahnschrift'},
                        merge_duplicate_headers=True,
                        export_format="csv",
                        # style_table={'minWidth': '100%'},
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable4-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 
                        html.Div(
                            dcc.Graph(
                                id='graph8',
                                figure=fig5,
                                config = config,
                            ),
                            style={'width': '100%', 'display': 'inline-block'}
                        ),
                    ]
                    ),
                ], className = 'tables'),


                html.Div([
                   
                    html.H5(children = html.Strong('2026 Population Projections by Income Category'), className = 'table-title'),

                    dash_table.DataTable(
                        id='datatable5-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                        ],
                        data=table.to_dict('records'),
                        editable=True,
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable=False,
                        row_selectable=False,
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        style_cell = {'font-family': 'Bahnschrift'},
                        merge_duplicate_headers=True,
                        export_format="csv",
                        # style_table={'minWidth': '100%'},
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable5-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 
                        html.Div(
                            dcc.Graph(
                                id='graph9',
                                figure=fig5,
                                config = config,
                            ),
                            style={'width': '100%', 'display': 'inline-block'}
                        ),
                    ]
                    ),
                ], className = 'tables'),

                html.Div([
                   
                    html.H5(children = html.Strong('2026 Population Projections by Household Size'), className = 'table-title'),

                    dash_table.DataTable(
                        id='datatable6-interactivity',
                        columns=[
                            {"name": i, "id": i, "deletable": False, "selectable": False} for i in table.columns
                        ],
                        data=table.to_dict('records'),
                        editable=True,
                        sort_action="native",
                        sort_mode="multi",
                        column_selectable=False,
                        row_selectable=False,
                        row_deletable=False,
                        selected_columns=[],
                        selected_rows=[],
                        page_action="native",
                        page_current= 0,
                        page_size= 10,
                        style_cell = {'font-family': 'Bahnschrift'},
                        merge_duplicate_headers=True,
                        export_format="csv",
                        # style_table={'minWidth': '100%'},
                        style_header = {'text-align': 'middle', 'fontWeight': 'bold'}
                    ),
                    html.Div(id='datatable6-interactivity-container'),


                    # Graphs

                    html.Div(children = [ 
                        html.Div(
                            dcc.Graph(
                                id='graph10',
                                figure=fig5,
                                config = config,
                            ),
                            style={'width': '100%', 'display': 'inline-block'}
                        ),
                    ]
                    ),
                ], className = 'tables'),


            ], #style={'width': '80%', 'padding-top': '30px', 'padding-bottom': '30px', 'display': 'block'}
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



# Area Selection Map

def province_map(value, random_color):

    clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Province_Code'].tolist()[0]

    gdf_p_code_added['Geo_Code'] = gdf_p_code_added.index
    
    if random_color == True:
        gdf_p_code_added["rand"] = [i for i in range(0,len(gdf_p_code_added))]
    else:
        gdf_p_code_added["rand"] = gdf_p_code_added['Geo_Code'].apply(lambda x: 0 if x == int(clicked_code) else 100)  

    fig_m = go.Figure()

    fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                    locations = gdf_p_code_added.index, 
                                    z = gdf_p_code_added.rand, 
                                    showscale = False, 
                                    colorscale = map_colors_wo_black,
                                    hovertext= gdf_p_code_added.NAME,
                                    marker = dict(opacity = 0.4),
                                    marker_line_width=.5))
    fig_m.update_layout(mapbox_style="carto-positron",
                mapbox_center = {"lat": gdf_p_code_added['lat'].mean()+10, "lon": gdf_p_code_added['lon'].mean()},
                height = 500,
                width = 1000,
                mapbox_zoom = 2.0,
                margin=dict(b=0,t=10,l=0,r=10),
                modebar_color = modebar_color, modebar_activecolor = modebar_activecolor,
                autosize=True)
    
    return fig_m


def region_map(value, random_color, clicked_code):
    
    if clicked_code == 'N':

        clicked_province_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Province_Code'].tolist()[0]
        clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Region_Code'].tolist()[0]

        gdf_r_filtered = gpd.read_file(f'./sources/mapdata_simplified/region_data/{clicked_province_code}.shp')
        
    else:

        gdf_r_filtered = gpd.read_file(f'./sources/mapdata_simplified/region_data/{clicked_code}.shp')
        
    if random_color == True:
        gdf_r_filtered["rand"] = [i for i in range(0,len(gdf_r_filtered))]
        
    else:
        gdf_r_filtered["rand"] = gdf_r_filtered['CDUID'].apply(lambda x: 0 if x == clicked_code else 100) 
        
        
    gdf_r_filtered = gdf_r_filtered.set_index('CDUID')

    fig_mr = go.Figure()

    fig_mr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_r_filtered.geometry.to_json()), 
                                    locations = gdf_r_filtered.index, 
                                    z = gdf_r_filtered.rand, 
                                    showscale = False,
                                    colorscale = map_colors_wo_black,
                                    hovertext= gdf_r_filtered.CDNAME,
                                    marker = dict(opacity = 0.4),
                                    marker_line_width=.5))


    fig_mr.update_layout(mapbox_style="carto-positron",
                    mapbox_center = {"lat": gdf_r_filtered['lat'].mean(), "lon": gdf_r_filtered['lon'].mean()},
                    height = 500,
                    width = 1000,
                    mapbox_zoom = 3.0,
                    margin=dict(b=0,t=10,l=0,r=10),
                    modebar_color = modebar_color, modebar_activecolor = modebar_activecolor,
                    autosize=True)
    
    return fig_mr

def subregion_map(value, random_color, clicked_code):
  
    # Importing Subregion Maps for selected Region
    if clicked_code == 'N':
        clicked_region_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Region_Code'].tolist()[0]
        clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Geo_Code'].tolist()[0]
        clicked_code = str(clicked_code)

        gdf_sr_filtered = gpd.read_file(f'./sources/mapdata_simplified/subregion_data/{clicked_region_code}.shp')
    
    else:
        clicked_code_region = clicked_code[:4]
        gdf_sr_filtered = gpd.read_file(f'./sources/mapdata_simplified/subregion_data/{clicked_code_region}.shp')
    
    if random_color == True:
        gdf_sr_filtered["rand"] = gdf_sr_filtered['CSDUID'].apply(lambda x: 0 if x in not_avail['CSDUID'].tolist() else np.random.randint(30, 100))
    
    else:
        gdf_sr_filtered["rand"] = gdf_sr_filtered['CSDUID'].apply(lambda x: 0 if x in not_avail['CSDUID'].tolist() else (50 if x == clicked_code else 100))
    
    gdf_sr_filtered = gdf_sr_filtered.set_index('CSDUID')


    if 0 in gdf_sr_filtered["rand"].tolist():
        colorlist = map_colors_w_black
    else:
        colorlist = map_colors_wo_black

    fig_msr = go.Figure()

    fig_msr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_sr_filtered.geometry.to_json()), 
                                    locations = gdf_sr_filtered.index, 
                                    z = gdf_sr_filtered.rand, 
                                    showscale = False, 
                                    hovertext= gdf_sr_filtered.CSDNAME,
                                    colorscale = colorlist,
                                    marker = dict(opacity = 0.4),
                                    marker_line_width=.5))

    max_bound = max(abs((gdf_sr_filtered['lat'].max() - gdf_sr_filtered['lat'].min())), 
                    abs((gdf_sr_filtered['lon'].max() - gdf_sr_filtered['lon'].min()))) * 111

    zoom = 11.5 - np.log(max_bound)

    if len(gdf_sr_filtered) == 1:
        zoom = 9

    fig_msr.update_layout(mapbox_style="carto-positron",
                    mapbox_center = {"lat": gdf_sr_filtered['lat'].mean(), "lon": gdf_sr_filtered['lon'].mean()},
                    height = 500,
                    width = 1000,
                    mapbox_zoom = zoom,
                    margin=dict(b=0,t=10,l=0,r=10),
                    modebar_color = modebar_color, modebar_activecolor = modebar_activecolor,
                    autosize=True)
    return fig_msr

@app.callback(
    Output('canada_map', 'figure'),
    Output('all-geo-dropdown', 'value'),
    [Input('canada_map', 'clickData')],
    Input('reset-map', 'n_clicks'),
    Input('all-geo-dropdown', 'value'),
    Input('all-geo-dropdown-parent', 'n_clicks'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
    )
def update_map(clickData, btn1, value, btn2, btn3, btn4, btn5):
    
    default_value = 'Greater Vancouver A RDA (CSD, BC)'

    if value == None:
        value = default_value

    clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Geo_Code'].tolist()[0]
    clicked_code = str(clicked_code)

    # When users click 'View Province' button or select a province on dropbox menu

    if (len(clicked_code) == 2 and 'all-geo-dropdown-parent' == ctx.triggered_id) or "to-province-1" == ctx.triggered_id:

        fig_m = province_map(value, False)
        
        return fig_m, value

    # When users click 'View Census Division' button or select a Census Division on dropbox menu

    if (len(clicked_code) == 4 and 'all-geo-dropdown-parent' == ctx.triggered_id) or "to-region-1" == ctx.triggered_id:

        # When users click 'View Census Division' button after selecting Province on dropbox menu
        # -> Show map for Province

        if len(clicked_code) == 2:

            fig_m = province_map(value, False)

            return fig_m, value

        # When users select Census Division on dropbox menu
        # or When users click 'View Census Division' button after selecting Census Division on dropbox menu
        # -> Show map for Census Division

        fig_mr = region_map(value, False, 'N')

        return fig_mr, value

    # When users click 'View Census SubDivision' button or select a Census SubDivision on dropbox menu

    if (len(clicked_code) > 4 and 'all-geo-dropdown-parent' == ctx.triggered_id) or "to-geography-1" == ctx.triggered_id:

        # When users click 'View Census SubDivision' button after selecting Census Division on dropbox menu
        # -> Show map for Census Division

        if len(clicked_code) == 4:

            fig_mr = region_map(value, False, 'N')
                        
            return fig_mr, value

        # When users click 'View Census SubDivision' button after selecting Province on dropbox menu
        # -> Show map for Province

        elif len(clicked_code) == 2:

            fig_m = province_map(value, False)

            return fig_m, value

        # When users select Census SubDivision on dropbox menu 
        # or when users click 'View Census SubDivision' button after selecting Census SubDivision on dropbox menu
        # -> Show map for Census SubDivision

        elif len(clicked_code) == 7:
            
            fig_msr = subregion_map(value, False, 'N')
            
            return fig_msr, value

    # When Reset-Map button is clicked

    if "reset-map" == ctx.triggered_id:

        fig_m = province_map(value, True)

        return fig_m, default_value


    # When users clicked province on the map

    if type(clickData) == dict:

        clicked_code = str(clickData['points'][0]['location'])

        if len(clicked_code) == 2:

            fig_mr = region_map(value, True, clicked_code)

            region_name = df_province_list.query("Geo_Code == " + f"{clicked_code}")['Geography'].tolist()[0]

            return fig_mr, region_name

        # When users clicked region on the regional map after clicking province 
        # -> show subregion map

        elif len(clicked_code) == 4:

            fig_msr = subregion_map(value, True, clicked_code)

            region_name = df_region_list.query("Geo_Code == " + f"{clicked_code}")['Geography'].tolist()[0]

            return fig_msr, region_name

        # When users clicked subregion on the regional map after clicking province 
        # -> remains subregion map and send subregion code to area selection dropdown

        elif len(clicked_code) > 4:
            
            fig_msr = subregion_map(value, False, clicked_code)

            subregion_name = mapped_geo_code.query("Geo_Code == " + f"{clicked_code}")['Geography'].tolist()[0]

            return fig_msr, subregion_name


    # default map (show provinces) before clicking anything on the map

    else:

        fig_m = province_map(value, True)

        return fig_m, default_value



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

@app.callback(
    Output('datatable-interactivity', 'columns'),
    Output('datatable-interactivity', 'data'),
    Output('datatable-interactivity', 'style_data_conditional'),
    Output('datatable-interactivity', 'style_cell_conditional'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
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

@app.callback(
    Output('graph', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
)
def update_geo_figure(geo, geo_c, btn1, btn2, btn3):

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


@app.callback(
    Output('graph2', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
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


@app.callback(
    Output('datatable2-interactivity', 'columns'),
    Output('datatable2-interactivity', 'data'),
    Output('datatable2-interactivity', 'style_data_conditional'),
    Output('datatable2-interactivity', 'style_cell_conditional'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
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

@app.callback(
    Output('graph5', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
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


@app.callback(
    Output('graph6', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
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



# 2026 Projections by HH Size and Income Level

income_col_list = ['20% or under of area median household income (AMHI)', 
                    '21% to 50% of AMHI', 
                    '51% to 80% of AMHI', 
                    '81% to 120% of AMHI', 
                    '121% or over of AMHI']

pp_list = ['1pp', '2pp', '3pp', '4pp', '5pp']

income_col_list_r = ['20%orunderofareamedianhouseholdincome(AMHI)', 
                    '21%to50%ofAMHI', 
                    '51%to80%ofAMHI', 
                    '81%to120%ofAMHI', 
                    '121%oroverofAMHI']

pp_list_r = ['1person', '2persons', '3persons', '4persons', '5ormorepersonshousehold']

def projections_2026(geo, IsComparison):
    
    df_csd_proj_merged_filtered = df_csd_proj_merged.loc[df_csd_proj_merged['Geography'] == geo,:]
    geo_region = int(mapped_geo_code.loc[mapped_geo_code['Geography'] == geo]['Region_Code'].values[0])
    df_cd_grow_merged_filtered = df_cd_grow_merged.loc[df_cd_grow_merged['Geo_Code'] == geo_region,:]

    geo_region_name = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Region'].tolist()[0]

    income_l = []
    pp_l = []
    result_csd_l = []
    result_g_l = []
    result_t_l = []

    for i, i_r in zip(income_col_list, income_col_list_r):
        for p, p_r in zip(pp_list, pp_list_r):
            col_format_r = f'TotalPrivatehouseholdsbyhouseholdtypeincludingcensusfamilystructure - Householdswithincome{i_r} - {p_r} - 2016'
            col_format_p = f'2026 Projected {p} HH with income {i}'
            col_format_g = f'2026 Projected Growth {p} HH with income {i}'
            income_l.append(i)
            pp_l.append(p)
            result_csd_l.append(df_csd_proj_merged_filtered[col_format_p].tolist()[0])
            result_t_l.append(df_csd_proj_merged_filtered[col_format_r].tolist()[0])
            result_g_l.append(df_cd_grow_merged_filtered[col_format_g].tolist()[0])

    income_l = ['Very Low Income'] * 5 + ['Low Income'] * 5 + ['Moderate Income'] * 5 + ['Median Income'] * 5 + ['High Income'] * 5

    table3 = pd.DataFrame({'Income_Category': income_l, 'HH_Category': pp_l, 'CSD_Projection': result_csd_l, 'CSD_Total': result_t_l, 'Growth': result_g_l})
    table3 = table3.fillna(0)
    table3['CSD_Total'] = table3['CSD_Total'].astype(float)
    table3['Growth'] = table3['Growth'].astype(float)
    table3['CSD_Projection'] = np.round(table3['CSD_Projection'].astype(float), -1)
    table3['Projection'] = np.round((table3['CSD_Total'] * table3['Growth']) + table3['CSD_Total'], -1)

    table3_csd = table3.pivot_table(values='CSD_Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
    table3_csd = table3_csd.reset_index()

    table3_cd_r = table3.pivot_table(values='Projection', index=['Income_Category'], columns=['HH_Category'], sort = False)
    table3_cd_r = table3_cd_r.reset_index()

    table3_csd_plot = table3_csd.replace([np.inf, -np.inf], 0)
    table3_csd_plot = pd.melt(table3_csd_plot, id_vars = 'Income_Category', value_vars = ['1pp', '2pp', '3pp', '4pp', '5pp'])

    table3_cd_r_plot = table3_cd_r.replace([np.inf, -np.inf], 0)
    table3_cd_r_plot = pd.melt(table3_cd_r_plot, id_vars = 'Income_Category', value_vars = ['1pp', '2pp', '3pp', '4pp', '5pp'])

    table3_csd = table3_csd.replace([np.inf, -np.inf], 0)
    row_total_csd = table3_csd.sum(axis=0)
    row_total_csd[0] = 'Total'
    table3_csd.loc[5, :] = row_total_csd

    table3_cd_r = table3_cd_r.replace([np.inf, -np.inf], 0)
    row_total_cd_r = table3_cd_r.sum(axis=0)
    row_total_cd_r[0] = 'Total'
    table3_cd_r.loc[5, :] = row_total_cd_r

    if IsComparison != True:
        table3_csd['Total'] = table3_csd.sum(axis=1)
        table3_cd_r['Total'] = table3_cd_r.sum(axis=1)
    else:
        table3_csd['Total '] = table3_csd.sum(axis=1)
        table3_cd_r['Total '] = table3_cd_r.sum(axis=1) 
        table3_csd = table3_csd.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })
        table3_cd_r = table3_cd_r.rename(columns = {'1pp': '1pp ', '2pp': '2pp ', '3pp': '3pp ', '4pp': '4pp ', '5pp': '5pp ', })
               
        
    return table3_csd, table3_cd_r, table3_csd_plot, table3_cd_r_plot, geo_region_name


@app.callback(
    Output('datatable3-interactivity', 'columns'),
    Output('datatable3-interactivity', 'data'),
    Output('datatable3-interactivity', 'style_data_conditional'),
    Output('datatable3-interactivity', 'style_cell_conditional'),
    Output('datatable4-interactivity', 'columns'),
    Output('datatable4-interactivity', 'data'),
    Output('datatable4-interactivity', 'style_data_conditional'),
    Output('datatable4-interactivity', 'style_cell_conditional'),
    Output('graph7', 'figure'),
    Output('graph8', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('datatable3-interactivity', 'selected_columns'),
    Input('datatable4-interactivity', 'selected_columns'),
)
def update_table3(geo, geo_c, selected_columns, selected_columns2):

    clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo, :]['Geo_Code'].tolist()[0]

    if geo_c != None:
        clicked_code_c = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo_c, :]['Geo_Code'].tolist()[0]
    else:
        clicked_code_c = 0

    # If Census Division or Province is selected

    if len(str(clicked_code)) < 6:

        table3_csd = pd.DataFrame({'Not Available in CD/Province level. Please select CSD level region':[0]})
        table3_cd_r = pd.DataFrame({'Not Available in CD/Province level. Please select CSD level region':[0]})
        
        col_list_csd = []

        for i in table3_csd.columns:
            col_list_csd.append({"name": [i],
                                    "id": i, })

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            col_list_cd_r.append({"name": [i],
                                    "id": i, })

        style_cell_conditional_csd=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_csd.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_csd.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        style_cell_conditional_cd_r=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_cd_r.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_cd_r.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        fig_csd = px.line(x = ['Not Available in CD/Province level. Please select CSD level region'], y = ['Not Available in CD/Province level. Please select CSD level region'])
        fig_cd_r = px.line(x = ['Not Available in CD/Province level. Please select CSD level region'], y = ['Not Available in CD/Province level. Please select CSD level region'])

        return col_list_csd, \
                table3_csd.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                style_cell_conditional_csd, \
                col_list_cd_r, \
                table3_cd_r.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                style_cell_conditional_cd_r, fig_csd, fig_cd_r
    

    # If Census SubDivision is selected

    if geo == geo_c or geo_c == None or (geo == None and geo_c != None) or len(str(clicked_code_c)) < 6:

        if geo == None and geo_c != None:
            geo = geo_c
        elif geo == None and geo_c == None:
            geo = 'Greater Vancouver A RDA (CSD, BC)'

        table3_csd, table3_cd_r, table3_csd_plot, table3_cd_r_plot, geo_region_name = projections_2026(geo, IsComparison = False)

        fig_csd = go.Figure()
        for i, c in zip(table3_csd_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_csd_plot.loc[table3_csd_plot['HH_Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ))

        fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH - {geo}', legend_title = "Income Category")


        fig_cd_r = go.Figure()
        for i, c in zip(table3_cd_r_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_cd_r_plot.loc[table3_cd_r_plot['HH_Category'] == i, :]
            fig_cd_r.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ))

        fig_cd_r.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH (Regional Rates) - {geo_region_name}', legend_title = "Income Category")

        col_list_csd = []

        for i in table3_csd.columns:
            col_list_csd.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            col_list_cd_r.append({"name": [geo_region_name, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        style_cell_conditional_csd=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_csd.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_csd.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        style_cell_conditional_cd_r=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_cd_r.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_cd_r.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        return col_list_csd, \
                table3_csd.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                style_cell_conditional_csd, \
                col_list_cd_r, \
                table3_cd_r.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                style_cell_conditional_cd_r, fig_csd, fig_cd_r

    else:


        table3_csd, table3_cd_r, table3_csd_plot, table3_cd_r_plot, geo_region_name = projections_2026(geo, IsComparison = False)
        
        # Comparison Tables/Plots
        
        table3_csd_c, table3_cd_r_c, table3_csd_c_plot, table3_cd_r_c_plot, geo_region_name_c = projections_2026(geo_c, IsComparison = True)

        fig_csd = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        n = 0
        for i, c in zip(table3_csd_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_csd_plot.loc[table3_csd_plot['HH_Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
            ), row = 1, col = 1)
            n += 1

        n = 0
        for i, c in zip(table3_csd_c_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_csd_c_plot.loc[table3_csd_c_plot['HH_Category'] == i, :]
            fig_csd.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
                showlegend = False
            ), row = 1, col = 2)
            n += 1
            
        fig_csd.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026', legend_title = "HH Category")
        fig_csd.update_yaxes(range=[0, max(table3_csd_plot.groupby('Income_Category')['value'].sum().max(), table3_csd_c_plot.groupby('Income_Category')['value'].sum().max())+5000])

        fig_cd_r = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo_region_name}", f"{geo_region_name_c}"), shared_yaxes=True, shared_xaxes=True)

        n = 0
        for i, c in zip(table3_cd_r_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_cd_r_plot.loc[table3_cd_r_plot['HH_Category'] == i, :]
            fig_cd_r.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
            ), row = 1, col = 1)
            n += 1

        n = 0
        for i, c in zip(table3_cd_r_c_plot['HH_Category'].unique(), colors):
            plot_df_frag = table3_cd_r_c_plot.loc[table3_cd_r_c_plot['HH_Category'] == i, :]
            fig_cd_r.add_trace(go.Bar(
                x = plot_df_frag['Income_Category'],
                y = plot_df_frag['value'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>',
                legendgroup = n,
                showlegend = False
            ), row = 1, col = 2)
            n += 1
            
        fig_cd_r.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'Community 2026 HH (Regional Rates)', legend_title = "HH Category")
        fig_cd_r.update_yaxes(range=[0, max(table3_cd_r_plot.groupby('Income_Category')['value'].sum().max(), table3_cd_r_c_plot.groupby('Income_Category')['value'].sum().max())+5000])


        col_list_csd = []

        for i in table3_csd.columns:
            if i == 'Income_Category':
                col_list_csd.append({"name": ["Subregions", i], "id": i})
            else:
                col_list_csd.append({"name": [geo, i],
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        col_list_cd_r = []

        for i in table3_cd_r.columns:
            if i == 'Income_Category':
                col_list_cd_r.append({"name": ["Regions", i], "id": i})
            else:
                col_list_cd_r.append({"name": [geo_region_name, i],
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        for i in table3_csd_c.columns[1:]:
            if i == 'Income_Category':
                col_list_csd.append({"name": ["Subregions", i], "id": i})
            else:
                col_list_csd.append({"name": [geo_c, i], 
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        for i in table3_cd_r_c.columns[1:]:
            if i == 'Income_Category':
                col_list_cd_r.append({"name": ["Regions", i], "id": i})
            else:
                col_list_cd_r.append({"name": [geo_region_name_c, i], 
                                      "id": i, 
                                      "type": 'numeric', 
                                      "format": Format(
                                                        group=Group.yes,
                                                        scheme=Scheme.fixed,
                                                        precision=0
                                                        )})

        table3_csd_j = table3_csd.merge(table3_csd_c, how = 'left', on = 'Income_Category')
        table3_cd_r_j = table3_cd_r.merge(table3_cd_r_c, how = 'left', on = 'Income_Category')

        style_cell_conditional_csd=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_csd.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table3_csd_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_csd.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]

        style_cell_conditional_cd_r=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table3_cd_r.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table3_cd_r_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table3_cd_r.columns[0]},
                'backgroundColor': columns_color_fill[0],
                'width': '130px'
            }
        ]


        return col_list_csd, \
                table3_csd_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns], \
                style_cell_conditional_csd, \
                col_list_cd_r, \
                table3_cd_r_j.to_dict('record'), \
                [{
                    'if': { 'column_id': i },
                    'background_color': '#D2F3FF'
                } for i in selected_columns2], \
                style_cell_conditional_cd_r, fig_csd, fig_cd_r


# new plot 1 for projection

def plot1_new_projection(geo, IsComparison):

    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Geo_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")

    updated_csd_filtered_2016_plot1 = updated_csd_filtered[[
        'Total - Private households by household type including census family structure -   Households with income 20% or under of area median household income (AMHI) - Total - Household size', 
        'Total - Private households by household type including census family structure -   Households with income 21% to 50% of AMHI - Total - Household size',
        'Total - Private households by household type including census family structure -   Households with income 51%  to 80% of AMHI - Total - Household size',
        'Total - Private households by household type including census family structure -   Households with income 81% to 120% of AMHI - Total - Household size',
        'Total - Private households by household type including census family structure -   Households with income 121% or over of AMHI - Total - Household size'
        ]].T.reset_index().drop(columns = ['index'])

    updated_csd_filtered_2026_plot1 = updated_csd_filtered[[
        '2026 Population Delta with income 20% or under of area median household income (AMHI)',
        '2026 Population Delta with income 21% to 50% of AMHI',
        '2026 Population Delta with income 51% to 80% of AMHI',
        '2026 Population Delta with income 81% to 120% of AMHI',
        '2026 Population Delta with income 121% or over of AMHI'
        ]].T.reset_index().drop(columns = ['index'])

    income_category = ['Very Low Income', 'Low Income', 'Moderate Income', 'Median Income', 'High Income']


    table1 = pd.DataFrame({'Income Category': income_category, 
                            'Category': (['2016 Pop'] * len(income_category)),
                            'Pop': updated_csd_filtered_2016_plot1.iloc[:,0]})
    
    table1_2016 = table1.copy()

    table1['2026 Delta'] = np.round(updated_csd_filtered_2026_plot1.iloc[:,0],0)
    table1 = table1.drop(columns = ['Category'])

    if IsComparison != True:
        table1.columns = ['Income Category', '2016 Population', '2026 Delta']
    else:
        table1.columns = ['Income Category', '2016 Population ', '2026 Delta ']

    plot_df = pd.concat([table1_2016,
                        pd.DataFrame({'Income Category': income_category,
                            'Category': (['2026 Delta'] * len(income_category)),
                            'Pop': np.round(updated_csd_filtered_2026_plot1.iloc[:,0],0)})])
    
    return plot_df, table1
    


@app.callback(
    Output('datatable5-interactivity', 'columns'),
    Output('datatable5-interactivity', 'data'),
    Output('datatable5-interactivity', 'style_data_conditional'),
    Output('datatable5-interactivity', 'style_cell_conditional'),
    Output('graph9', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks'),
    Input('datatable5-interactivity', 'selected_columns'),
)
def update_geo_figure6(geo, geo_c, btn1, btn2, btn3, selected_columns):

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

        plot_df, table1 = plot1_new_projection(geo, False)

        fig_new_proj_1 = go.Figure()

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['Income Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ))

        fig_new_proj_1.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'2026 Population Projections - {geo}', legend_title = "Category")

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1


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

        plot_df, table1 = plot1_new_projection(geo, False)

        fig_new_proj_1 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['Income Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ),row = 1, col = 1)

        col_list = []

        for i in table1.columns:
            if i == 'Income Category':
                col_list.append({"name": ["Income", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        plot_df_c, table1_c = plot1_new_projection(geo_c, True)

        for i, c in zip(plot_df_c['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df_c.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['Income Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'Income Category':
                col_list.append({"name": ["Income", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_new_proj_1.update_layout(xaxis=dict(autorange="reversed"), modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', showlegend = False, plot_bgcolor='#F8F9F9', title = f'2026 Population Projections', legend_title = "Category")
        fig_new_proj_1.update_yaxes(range=[0, max(plot_df.groupby('Income Category')['Pop'].sum().max(), plot_df_c.groupby('Income Category')['Pop'].sum().max())+100])

        table1_j = table1.merge(table1_c, how = 'left', on = 'Income Category')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1

# new plot 2 for projection

def plot2_new_projection(geo, IsComparison):

    geo_code_clicked = mapped_geo_code.loc[mapped_geo_code['Geography'] == geo,:]['Geo_Code'].tolist()[0]
    updated_csd_filtered = updated_csd.query('Geo_Code ==' +  f"{geo_code_clicked}")

    updated_csd_filtered_2016_plot2 = updated_csd_filtered[[
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   1 person',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   2 persons',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   3 persons',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   4 persons',
'Total - Private households by household type including census family structure - Total – Private households by household income proportion to AMHI_1 -   5 or more persons household'
        ]].T.reset_index().drop(columns = ['index'])

    updated_csd_filtered_2026_plot2 = updated_csd_filtered[[
    '2026 Population Delta 1pp HH',
    '2026 Population Delta 2pp HH',
    '2026 Population Delta 3pp HH',
    '2026 Population Delta 4pp HH',
    '2026 Population Delta 5pp HH'
        ]].T.reset_index().drop(columns = ['index'])

    hh_category = ['1 Person HH', '2 People HH', '3 People HH', '4 People HH', '5+ People HH']


    table2 = pd.DataFrame({'HH Category': hh_category, 
                            'Category': (['2016 Pop'] * len(hh_category)),
                            'Pop': updated_csd_filtered_2016_plot2.iloc[:,0]})
    
    table2_2016 = table2.copy()

    table2['2026 Delta'] = np.round(updated_csd_filtered_2026_plot2.iloc[:,0],0)
    table2 = table2.drop(columns = ['Category'])

    if IsComparison != True:
        table2.columns = ['HH Category', '2016 Population', '2026 Delta']
    else:
        table2.columns = ['HH Category', '2016 Population ', '2026 Delta ']

    plot_df = pd.concat([table2_2016,
                        pd.DataFrame({'HH Category': hh_category,
                            'Category': (['2026 Delta'] * len(hh_category)),
                            'Pop': np.round(updated_csd_filtered_2026_plot2.iloc[:,0],0)})])
    
    return plot_df, table2

@app.callback(
    Output('datatable6-interactivity', 'columns'),
    Output('datatable6-interactivity', 'data'),
    Output('datatable6-interactivity', 'style_data_conditional'),
    Output('datatable6-interactivity', 'style_cell_conditional'),
    Output('graph10', 'figure'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks'),
    Input('datatable6-interactivity', 'selected_columns'),
)
def update_geo_figure7(geo, geo_c, btn1, btn2, btn3, selected_columns):

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

        plot_df, table1 = plot2_new_projection(geo, False)

        fig_new_proj_1 = go.Figure()

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ))

        fig_new_proj_1.update_layout(legend_traceorder="normal", modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', plot_bgcolor='#F8F9F9', title = f'2026 Population Projections - {geo}', legend_title = "Category")

        col_list = []

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        for i in table1.columns:
            col_list.append({"name": [geo, i],
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        return col_list, table1.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1


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

        plot_df, table1 = plot2_new_projection(geo, False)

        fig_new_proj_1 = make_subplots(rows=1, cols=2, subplot_titles=(f"{geo}", f"{geo_c}"), shared_yaxes=True, shared_xaxes=True)

        for i, c in zip(plot_df['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ),row = 1, col = 1)

        col_list = []

        for i in table1.columns:
            if i == 'HH Category':
                col_list.append({"name": ["Household", i], "id": i})
            else:
                col_list.append({"name": [geo, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        # Comparison Plot

        plot_df_c, table1_c = plot2_new_projection(geo_c, True)

        for i, c in zip(plot_df_c['Category'].unique(), colors[3:]):
            plot_df_frag = plot_df_c.loc[plot_df['Category'] == i, :]
            fig_new_proj_1.add_trace(go.Bar(
                x = plot_df_frag['HH Category'],
                y = plot_df_frag['Pop'],
                name = i,
                marker_color = c,
                # orientation = 'h', 
                hovertemplate= '%{y} - ' + '%{x}<extra></extra>'
            ),row = 1, col = 2)

        for i in table1_c.columns[1:]:
            if i == 'HH Category':
                col_list.append({"name": ["Household", i], "id": i})
            else:
                col_list.append({"name": [geo_c, i], 
                                    "id": i, 
                                    "type": 'numeric', 
                                    "format": Format(
                                                    group=Group.yes,
                                                    scheme=Scheme.fixed,
                                                    precision=0
                                                    )})

        fig_new_proj_1.update_layout(xaxis=dict(autorange="reversed"), modebar_color = modebar_color, modebar_activecolor = modebar_activecolor, barmode='stack', showlegend = False, plot_bgcolor='#F8F9F9', title = f'2026 Population Projections', legend_title = "Category")
        fig_new_proj_1.update_yaxes(range=[0, max(plot_df.groupby('HH Category')['Pop'].sum().max(), plot_df_c.groupby('HH Category')['Pop'].sum().max())+100])

        table1_j = table1.merge(table1_c, how = 'left', on = 'HH Category')

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[1]
            } for c in table1.columns[1:]
        ] + [
            {
                'if': {'column_id': c},
                'backgroundColor': columns_color_fill[2]
            } for c in table1_c.columns[1:]
        ] + [
            {
                'if': {'column_id': table1.columns[0]},
                'backgroundColor': columns_color_fill[0]
            }
        ]

        return col_list, table1_j.to_dict('record'), [{
            'if': { 'column_id': i },
            'background_color': '#D2F3FF'
        } for i in selected_columns], style_cell_conditional, fig_new_proj_1




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
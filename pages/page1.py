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




        ], className = 'dashboard'
    ), 
], className = 'background'#style = {'backgroud-color': '#fffced'}
)


@callback(
    Output('main-area', 'data'),
    Output('comparison-area', 'data'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    )
def store_geo(geo, geo_c):
    return geo, geo_c


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

@callback(
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

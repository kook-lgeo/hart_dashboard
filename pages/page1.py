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

income_category = df_income.drop(['Geography'], axis=1)

income_category = income_category.rename(columns = {'Formatted Name': 'Geography'})

joined_df = income_category.merge(df_partners, how = 'left', on = 'Geography')

# Importing Geo Code Information

mapped_geo_code = pd.read_sql_table('geocodes_integrated', engine.connect())
df_geo_list = pd.read_sql_table('geocodes', engine.connect())
df_region_list = pd.read_sql_table('regioncodes', engine.connect())
df_region_list.columns = df_geo_list.columns
df_province_list = pd.read_sql_table('provincecodes', engine.connect())
df_province_list.columns = df_geo_list.columns


gdf_p_code_added = gpd.read_file('./sources/mapdata_simplified/province.shp')
gdf_p_code_added = gdf_p_code_added.set_index('Geo_Code')

# Importing subregions which don't have data

not_avail = pd.read_csv('not_in_list.csv')
not_avail['CSDUID'] = not_avail['CSDUID'].astype(str)

# Configuration for plot icons

config = {'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['zoom', 'lasso2d', 'pan', 'select', 'autoScale', 'resetScale', 'resetViewMapbox']}


# colors = ['#D7F3FD', '#B0E6FC', '#88D9FA', '#61CDF9', '#39C0F7']
# hh_colors = ['#D8EBD4', '#B7DCAE', '#93CD8A', '#6EC067', '#3DB54A']
colors = ['#D7F3FD', '#88D9FA', '#39C0F7', '#099DD7', '#044762']
hh_colors = ['#D8EBD4', '#93CD8A', '#3DB54A', '#297A32', '#143D19']
hh_type_color = ['#3949CE', '#3EB549', '#39C0F7']
columns_color_fill = ['#F3F4F5', '#EBF9FE', '#F0FAF1']

map_colors_province = ['#1a3758', '#78cb80', '#74d3f9', '#a480bb', '#80c2c0', '#7480dd', '#ffe6d6', '#e98098', '#b6657c', '#490076', '#008481', '#622637']
map_colors_wo_black = ['#7480dd', '#1a3758', '#7480dd', '#b6657c', '#622637', '#80c2c0', '#78cb80', '#ffe6d6', '#e98098', '#a480bb', '#490076', '#008481', '#74d3f9']
# map_colors_wo_black = ['#3EB549', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#39C0F7']
map_colors_w_black = ['#000000', '#1a3758', '#7480dd', '#b6657c', '#622637', '#80c2c0', '#78cb80', '#ffe6d6', '#e98098', '#a480bb', '#490076', '#008481', '#74d3f9']
# map_colors_w_black = ['#000000', '#F4F739', '#fa6464', '#3EB549', '#EE39F7', '#752100', '#39C0F7']


modebar_color = '#099DD7'
modebar_activecolor = '#044762'

opacity_value = 0.2

mapped_geo_code2 = mapped_geo_code.copy()
mapped_geo_code2['Geo_Code'] = mapped_geo_code2['Geo_Code'].astype(str)
order = pd.DataFrame(joined_df['Geography']).merge(mapped_geo_code2[['Geography','Geo_Code','Region_Code', 'Province_Code']], how = 'left', on = 'Geography')
order = order.sort_values(by = ['Province_Code','Region_Code','Geo_Code'])

gdf_p_code_added["rand"] = np.random.randint(1, 100, len(gdf_p_code_added))

fig_m = go.Figure()

fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                locations = gdf_p_code_added.index, 
                                z = gdf_p_code_added.rand, 
                                showscale = False, 
                                hovertext= gdf_p_code_added.NAME,
                                marker = dict(opacity = opacity_value),
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

        dcc.Store(id='area-scale-store', storage_type='local'),
        dcc.Store(id='main-area', storage_type='local'),
        dcc.Store(id='comparison-area', storage_type='local'),

        html.Div(
        children = [
            # html.Div([
            #     html.H2(children = html.Strong("Census Geography Area Selection"), id = 'home')
            # ], className = 'title-lgeo'),

            # Dropdown for sector selection
            # html.H3(children = html.Strong('Area Selection'), id = 'area-selection'),


            html.Div(children = [ 
                html.Div(
                    id = 'all-geo-dropdown-parent',
                    children = [
                    html.Strong('Select Census Geography'),
                    dcc.Dropdown(order['Geography'].unique()[1:], 'Greater Vancouver A RDA (CSD, BC)', id='all-geo-dropdown'),
                    ], 
                    className = 'dropdown-lgeo'
                ),

                html.Div(
                    id = 'comparison-geo-dropdown-parent',
                    children = [
                    html.Strong('Select Comparison Census Geography (Optional)'),
                    dcc.Dropdown(order['Geography'].unique()[1:], id='comparison-geo-dropdown'),
                    ], 
                    className = 'dropdown-lgeo'
                ),
            ], 
                className = 'dropdown-box-lgeo'
            ),

            html.Div(children = [ 
                # html.Div(children = [ 
                    html.Div(children = [                     
                        html.Button('View Census Subdivision (CSD)', id='to-geography-1', n_clicks=0, className = 'region-button-lgeo'),     
                                        ], className = 'region-button-box-lgeo'
                        ),           
                    html.Div(children = [ 
                        html.Button('View Census Division (CD)', id='to-region-1', n_clicks=0, className = 'region-button-lgeo'),
                                        ], className = 'region-button-box-lgeo'
                        ),         
                    html.Div(children = [ 
                        html.Button('View Province / Territory', id='to-province-1', n_clicks=0, className = 'region-button-lgeo'),
                                        ], className = 'region-button-box-lgeo'
                        ),    
                #     ], 
                #     className = 'scale-buttons-lgeo'
                # ),
                ], 
                className = 'scale-button-box-lgeo'
            ),

            # Map picker

            html.Div(children = [ 
                html.Div(
                    dcc.Graph(
                        id='canada_map',
                        figure=fig_m,
                        config = config,
                    ),
                    className = 'map-lgeo'

                ),

                html.Div(children = [                     
                    html.Button('Reset Map', id='reset-map', n_clicks=0),                   
                    ], className = 'reset-button-lgeo'
                ),
 
             ], className = 'map-area-box-lgeo'
            ),
            # Reset Button for Map

            # html.Div(children = [ 


   
            #     ], 
            #     className = 'reset-button-lgeo'
            # ),

        ], className = 'dashboard-pg1-lgeo'
    ), 
], className = 'background-lgeo'
)


@callback(
    Output('main-area', 'data'),
    Output('comparison-area', 'data'),
    Output('area-scale-store', 'data'),
    Input('all-geo-dropdown', 'value'),
    Input('comparison-geo-dropdown', 'value'),
    Input('all-geo-dropdown-parent', 'n_clicks'),
    Input('comparison-geo-dropdown-parent', 'n_clicks'),
    Input('to-geography-1', 'n_clicks'),
    Input('to-region-1', 'n_clicks'),
    Input('to-province-1', 'n_clicks')
    )
def store_geo(geo, geo_c, btn1, btn2, btn3, btn4, btn5):
    id_name = str(ctx.triggered_id)

    return geo, geo_c, id_name


# @callback(
#     Output('area-scale-store', 'data'),
#     Input('all-geo-dropdown-parent', 'n_clicks'),
#     Input('comparison-geo-dropdown-parent', 'n_clicks'),
#     Input('to-geography-1', 'n_clicks'),
#     Input('to-region-1', 'n_clicks'),
#     Input('to-province-1', 'n_clicks')
#     )
# def store_area_scale(btn1, btn2, btn3, btn4, btn5):
#     # print(ctx.triggered_id)
#     # if 'all-geo-dropdown-parent' == ctx.triggered_id or 'comparison-geo-dropdown-parent' == ctx.triggered_id:
#     #     return 'reset'

#     # elif 'to-geography-1' == ctx.triggered_id:
#     #     return 'to-geography-1'

#     # elif 'to-region-1' == ctx.triggered_id:
#     #     return 'to-region-1'

#     # elif 'to-province-1' == ctx.triggered_id:
#     #     return 'to-province-1'

#     return ctx.triggered_id



# Area Selection Map



def province_map(value, random_color):

    clicked_code = mapped_geo_code.loc[mapped_geo_code['Geography'] == value, :]['Province_Code'].tolist()[0]

    gdf_p_code_added['Geo_Code'] = gdf_p_code_added.index
    
    if random_color == True:
        gdf_p_code_added["rand"] = [round(i/(len(gdf_p_code_added) -1)*100) for i in range(0,len(gdf_p_code_added))]
        map_colors = map_colors_wo_black
    else:
        gdf_p_code_added["rand"] = gdf_p_code_added['Geo_Code'].apply(lambda x: 0 if x == int(clicked_code) else 100)
        map_colors = ['#37BB31', '#74D3F9']


    fig_m = go.Figure()

    fig_m.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_p_code_added.geometry.to_json()), 
                                    locations = gdf_p_code_added.index, 
                                    z = gdf_p_code_added.rand, 
                                    showscale = False, 
                                    colorscale = map_colors,
                                    text = gdf_p_code_added.NAME,
                                    hovertemplate= '%{text} - %{location}<extra></extra>',
                                    marker = dict(opacity = opacity_value),
                                    marker_line_width=.5))
    fig_m.update_layout(mapbox_style="carto-positron",
                mapbox_center = {"lat": gdf_p_code_added['lat'].mean()+10, "lon": gdf_p_code_added['lon'].mean()},
                # height = 500,
                # width = 1000,
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
        map_colors = map_colors_wo_black
        
    else:
        gdf_r_filtered["rand"] = gdf_r_filtered['CDUID'].apply(lambda x: 0 if x == clicked_code else 100) 
        map_colors = ['#37BB31', '#74D3F9']

        
        
    gdf_r_filtered = gdf_r_filtered.set_index('CDUID')

    fig_mr = go.Figure()

    fig_mr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_r_filtered.geometry.to_json()), 
                                    locations = gdf_r_filtered.index, 
                                    z = gdf_r_filtered.rand, 
                                    showscale = False,
                                    colorscale = map_colors,
                                    text = gdf_r_filtered.CDNAME,
                                    hovertemplate= '%{text} - %{location}<extra></extra>',
                                    marker = dict(opacity = opacity_value),
                                    marker_line_width=.5))


    fig_mr.update_layout(mapbox_style="carto-positron",
                    mapbox_center = {"lat": gdf_r_filtered['lat'].mean(), "lon": gdf_r_filtered['lon'].mean()},
                    # height = 500,
                    # width = 1000,
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
        if 0 in gdf_sr_filtered["rand"].tolist():
            colorlist = map_colors_w_black
        else:
            colorlist = map_colors_wo_black

    else:
        gdf_sr_filtered["rand"] = gdf_sr_filtered['CSDUID'].apply(lambda x: 0 if x in not_avail['CSDUID'].tolist() else (50 if x == clicked_code else 100))

        if 0 in gdf_sr_filtered["rand"].tolist():
            colorlist = ['#000000', '#37BB31', '#74D3F9']
        else:
            colorlist = ['#37BB31', '#74D3F9']
        
    
    gdf_sr_filtered = gdf_sr_filtered.set_index('CSDUID')




    fig_msr = go.Figure()

    fig_msr.add_trace(go.Choroplethmapbox(geojson = json.loads(gdf_sr_filtered.geometry.to_json()), 
                                    locations = gdf_sr_filtered.index, 
                                    z = gdf_sr_filtered.rand, 
                                    showscale = False, 
                                    text = gdf_sr_filtered.CSDNAME,
                                    hovertemplate= '%{text} - %{location}<extra></extra>',
                                    colorscale = colorlist,
                                    marker = dict(opacity = opacity_value),
                                    marker_line_width=.5))

    max_bound = max(abs((gdf_sr_filtered['lat'].max() - gdf_sr_filtered['lat'].min())), 
                    abs((gdf_sr_filtered['lon'].max() - gdf_sr_filtered['lon'].min()))) * 111

    zoom = 11.5 - np.log(max_bound)

    if len(gdf_sr_filtered) == 1:
        zoom = 9

    fig_msr.update_layout(mapbox_style="carto-positron",
                    mapbox_center = {"lat": gdf_sr_filtered['lat'].mean(), "lon": gdf_sr_filtered['lon'].mean()},
                    # height = 500,
                    # width = 1000,
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


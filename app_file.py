
# Define the Dash App and it's attributes here 

import dash
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP], 
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=0.5"}],
                suppress_callback_exceptions=True)
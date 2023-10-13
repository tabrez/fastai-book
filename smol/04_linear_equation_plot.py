import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='live-graph', animate=True),
    dcc.Slider(
        id='slope-slider',
        min=-10,
        max=10,
        value=1,
        step=0.1,
        marks={i: '{}'.format(i) for i in range(-10, 11)},
    ),
    html.Div(id='slope-output-container'),
    dcc.Slider(
        id='intercept-slider',
        min=-10,
        max=10,
        value=0,
        step=0.1,
        marks={i: '{}'.format(i) for i in range(-10, 11)},
    ),
    html.Div(id='intercept-output-container')
])

# Define callback to update graph
@app.callback(
    Output('live-graph', 'figure'),
    [Input('slope-slider', 'value'),
     Input('intercept-slider', 'value')]
)
def update_graph(slope, intercept):
    x = np.linspace(-10, 10, 400)
    y = slope * x + intercept

    fig = go.Figure(
        data=[go.Scatter(x=x, y=y, mode='lines')],
        layout=go.Layout(
            xaxis=dict(range=[-10, 10]),
            yaxis=dict(range=[-10, 10]),
            title="y = {}x + {}".format(slope, intercept)
        )
    )

    return fig

# Define callback to update slider output
@app.callback(
    Output('slope-output-container', 'children'),
    [Input('slope-slider', 'value')]
)
def update_slope_output(value):
    return 'Slope: {}'.format(value)

@app.callback(
    Output('intercept-output-container', 'children'),
    [Input('intercept-slider', 'value')]
)
def update_intercept_output(value):
    return 'Intercept: {}'.format(value)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

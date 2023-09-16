# https://altair-viz.github.io/getting_started/starting.html
#%% The Data
import numpy as np
import pandas as pd
data = pd.DataFrame({'a': list('CCCDDDEEE'),
                     'b': [2, 7, 4, 1, 2, 6, 8, 4, 7]})
data

# The Chart Object
import altair as alt
chart = alt.Chart(data)

## Encodings and Marks

chart = alt.Chart(data).mark_point().encode(x='a', y='b')
chart

#%% Data Transformation: Aggregation
chart1 = alt.Chart(data).mark_point().encode(x='a', y='average(b)')
chart2 = alt.Chart(data).mark_bar().encode(x='a', y='average(b)')
chart3 = alt.Chart(data).mark_bar().encode(x='average(b)', y='a')
chart4 = alt.Chart(data).mark_bar().encode(
  alt.X('b', type='quantitative', aggregate='average'),
  alt.Y('a', type='nominal')
)
chart5 = alt.Chart(data).mark_bar(color='firebrick').encode(
  alt.X('b', type='quantitative', aggregate='average'),
  alt.Y('a', type='nominal')
)
chart5.save('chart.html')
chart5

#%% Control plot using a slider html input
# https://altair-viz.github.io/user_guide/interactions.html
import numpy as np
import pandas as pd

rand = np.random.RandomState(42)
df = pd.DataFrame({
  'xval': range(100),
  'yval': rand.randn(100).cumsum()
})
slider = alt.binding_range(min=0, max=100, step=1, name='Cutoff ')
selector = alt.param(name='SelectorName', value=50, bind=slider)

alt.Chart(df).mark_point().encode(
  x='xval',
  y='yval',
  color=alt.condition(
    alt.datum.xval < selector,
    alt.value('red'),
    alt.value('blue')
  )
).add_params(selector)

#%% approximate linear expression

x = np.arange(100)
y = 5 * x + 9 + np.random.normal(loc=47, scale=47, size=x.shape)
df = pd.DataFrame({
  'xval': x,
  'yval': y
})

bind_range = alt.binding_range(min=0, max=15, name='W: ')
# W = alt.param(bind=bind_range)
sel = alt.selection_point(name='W', fields=['val'], bind=bind_range)
# W = alt.param('val', value=5, bind=bind_range)
# y_expr = alt.expr(f'{W.val} * x')
# y_expr = alt.param(expr=alt.expr(W * x))
alt.Chart(df).mark_point().encode(
  x='xval',
  y='yval'
)
alt.Chart(df).mark_line().encode(
  x='xval',
  y='ypred:Q'
).add_params(
  sel
).transform_calculate(
  ypred=f'datum.W_val * datum.xval + 9'
)

#%%
import plotly.graph_objects as go
import numpy as np

# Create figure
fig = go.Figure()

# Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name="𝜈 = " + str(step),
            x=np.arange(0, 10, 0.01),
            y=np.sin(step * np.arange(0, 10, 0.01))))

# Make 10th trace visible
fig.data[10].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()

#%%
import plotly.express as px

df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

fig["layout"].pop("updatemenus") # optional, drop animation buttons
fig.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def update_plot(m, b):
    x = np.linspace(-100, 100, 100)
    y = m * x + b
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Line: y = {}x + {}'.format(m, b))
    plt.grid(True)
    plt.ylim(-100, 100)
    plt.xlim(-100, 100)
    plt.show()

interact(update_plot, m=FloatSlider(min=-5.0, max=15.0, step=0.1, value=1.0),
         b=FloatSlider(min=-150, max=150, step=1, value=0.0))

#%%
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def update_plot(m, b):
    x = np.linspace(-100, 100, 100)
    y = m * x + b
    plt.plot(x, y)

    # Scatter plot with x2 and y2
    x2 = np.linspace(-100, 100, 100)
    y2 = 11 * x2 + 1123 + np.random.normal(0, 200, len(x2))
    plt.scatter(x2, y2, color='blue', label='Scatter Plot')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Line: y = {}x + {}'.format(m, b))
    plt.grid(True)
    plt.ylim(-1000, 1500)
    plt.xlim(-100, 100)
    plt.legend()
    plt.show()

interact(update_plot, m=FloatSlider(min=-5.0, max=15.0, step=1, value=1.0),
         b=FloatSlider(min=-150, max=150, step=1, value=0.0))

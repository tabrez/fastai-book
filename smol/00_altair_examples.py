# https://altair-viz.github.io/getting_started/starting.html
#%% The Data

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

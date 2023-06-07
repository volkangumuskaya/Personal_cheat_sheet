# PLOTS

Some plots

## Import libs. Note that ```pio.renderers='svg'``` is needed for pycharm for ```.show```

```python
import plotly.express as px
import plotly.io as pio
pio.renderers='svg'
```

## Histogram simple
```python
df.sepal_length_cm.hist(bins=30, facecolor='green', alpha=0.75)
```

## Histogram detailed with plotly
```python
fig=px.histogram(df.sepal_length_cm,title="Histogram")
fig.update_xaxes(title_text="sepal_length_cm")
fig.update_yaxes(title_text="n")
fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        tick0 = 2,
        dtick = 0.5
    ),bargap=0.01
)
fig.update_xaxes(showgrid=True)

fig.update_yaxes(showgrid=True)
fig.update_xaxes(range=[2,10])
fig.update_yaxes(range=[0,50])
fig.update_traces(marker_color='green')
fig.show()
```


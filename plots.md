# PLOTS

Some plots

### Import libs. Note that ```pio.renderers='svg'``` is needed for pycharm for ```fig.show```. Also quite often when ```fig``` is opened in a browser, you may need to refresh, otherwise it is stuck in loading.

```python
import plotly.express as px
import plotly.io as pio
pio.renderers='svg'
```

## Histogram simple
```python
df.sepal_length_cm.hist(bins=30, facecolor='green', alpha=0.75)
```
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/41029f63-2332-4bb8-a2b9-ed15fb9d2bd5)


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
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/56586666-5fe4-41b9-8789-506b03eb616b)

# Box plot with plotly
```python
px.box(df,\
       x="origin_country", y="sepal_length_cm", color="target",\
       facet_col='dest_country',\
       # category_orders={'DT_name':['dt_before_postcodes','dt_before_corridor','dt']},\
       points='all').show()
```
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/398d8445-81b4-416d-9911-24d8e2565d03)

# Bar and scatter chart together with a fixed line

```python
df['flower_id']=np.arange(0,len(df))


from plotly.subplots import make_subplots
import plotly.graph_objects as go


fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(
    go.Bar(x=df.flower_id, y=df.sepal_width_cm,
           # text=round(metrics.R2, 2),
           marker_color='light blue', opacity=0.7,
           name="Sepal_width"),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=df.flower_id, y=df.sepal_length_cm,
               mode='markers', name="Sepal length"),
    secondary_y=True
)
#Optional line
fig.add_trace(
    go.Scatter(x=[0,150], y=[0,5])
)

fig.update_traces(marker=dict(size=10,
                              line=dict(width=1.5, color='black')),
                  selector=dict(mode='markers'))
fig.update_layout(
    title="Bar and scatter",
    xaxis_title="Flower ID",
    yaxis_title="Length",
    legend_title="Trace type",
)
fig.update_yaxes(title_text="Length_2", secondary_y=True)
fig.update_yaxes(range=[0, 5], secondary_y=False)
fig.update_yaxes(range=[0, 10], secondary_y=True)
fig.show()
```
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/83f14eb1-8288-4872-abaf-b92a11bfc755)

# Scatter plot with size and custom markers
```python
fig = go.Figure(
    data=[go.Scatter(x=df.flower_id, y=df.sepal_length_cm, text=df.int_class,
                     # textposition="top center",
                     mode='markers+text',
                    textposition='middle center',
                     showlegend=False,
                     marker=dict(
                         size=df.petal_length_cm,
                         color=df.target,
                         sizemode='area',
                         sizeref=2. * max(df.petal_length_cm,) / (40. ** 2),
                         sizemin=0,
                         showscale=True,
                         colorscale=[(0, "green"), (0.5, "yellow"), (1, "red")],
                         colorbar=dict(orientation='v', title='target')
                     )
                     ),
          ]
)

fig.update_layout(title="Title",
                  xaxis_title='Flower_id',
                  yaxis_title='Sepal length')
# fig.layout.update(xaxis_range=[0, 150])
# fig.layout.update(yaxis_range=[0, 8])
fig.show()
```
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/80eb29d9-68f1-4c3d-b36a-dfec3dd459a1)

# Subplots with different types
```python
fig = make_subplots(1, 2, specs=[[{"secondary_y": True}, {"secondary_y": True}]],
                    shared_yaxes='all')

fig.add_trace(
    go.Scatter(x=df.flower_id, y=df.sepal_length_cm,
               mode='lines+markers',
               marker=dict(color='red',
                           size=df.int_class,
                           sizeref=max(df.int_class) / (4.5 ** 2),
                           sizemin=0),
               name='Sepal length'),
    row=1,
    col=1,
    secondary_y=False
)
fig.add_trace(
    go.Bar(x=df.flower_id, y=df.int_class,
           opacity=0.4,marker={'color':'navy'},
           name='Class'),
    row=1,
    col=1,
    secondary_y=True
)
fig.add_trace(
    go.Scatter(x=df.flower_id, y=df.petal_length_cm,
               mode='markers',
               marker=dict(color='red',
                           size=df.int_class,
                           sizeref=max(df.int_class) / (4.5 ** 2),
                           sizemin=0),
               name='Petal length'),
    row=1,
    col=2,
    secondary_y=False
)
fig.add_trace(
    go.Bar(x=df.flower_id, y=df.int_class,
           opacity=0.4,marker={'color':'navy'},
           name='Class'),
    row=1,
    col=2,
    secondary_y=True
)
fig.update_yaxes(showgrid=False, secondary_y=True,range=[0,20])
fig.show()
```
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/04023f7a-021b-4a1a-bc0d-dd635d4591fa)

# Scatter plot with ordered x-axis

```python
x_categories= df.sort_values('int_class').flower_id.to_list()

fig=px.scatter(df,x='flower_id',y='petal_length_cm',
               size='int_class',color='int_class',color_continuous_scale='ylorrd')
fig.update_xaxes(type="category",categoryorder='array', categoryarray= x_categories)
fig.update_xaxes(tickangle=90)
fig.update_traces(marker=dict(line=dict(width=1.5, color='black')))
fig.show()
```
![image](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/69f3a02d-5c93-43d0-a939-4b5b1e400295)


# Save plotly figure offline 
```python
#save plotly figure
import plotly
#import plotly.offline
plotly.offline.plot(fig,filename='images\\'+'test.html',auto_open=False)
#fig.savefig("/dbfs/FileStore/filename.png")
```

# Correlations
```python
from scipy.stats import pearsonr
pearsonr(df.sepal_length_cm, df.petal_length_cm)
df.sepal_length_cm.corr(df.petal_length_cm)
```

# weighted correlation
```python
def linear_fit_func(x, a, b):
    return a + b * x

def m_weighted_corr(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov_weighted_corr(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m_weighted_corr(x, w)) * (y - m_weighted_corr(y, w))) / np.sum(w)


def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return cov_weighted_corr(x, y, w) / np.sqrt(cov_weighted_corr(x, x, w) * cov_weighted_corr(y, y, w))

weighted_corr(df.dropna().sepal_length_cm,
              df.dropna().petal_length_cm,
              len(df.dropna())*[1])
```

# normality test
```python
from scipy import stats
norm_test_arr = np.random.randn(100000)
stats.shapiro(norm_test_arr)
```

# Plot a heatmap from consuion matrix
```python
import matplotlib.pyplot as plt
import seaborn as sns
#create y_test and y_pred from a list
data_tmp = [1, 2, 3,1,1,2,3,1,1,2,3,2,2,2,1,1,1,1]
y_test_example = pd.Series(data_tmp, copy=False)
data_tmp = [1, 2, 3,1,2,2,3,1,1,1,3,2,3,2,3,1,1,1]
y_pred_example = pd.Series(data_tmp, copy=False)
#plot heatmap
import seaborn as sns
kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}

cf_matrix = confusion_matrix(y_test_example, y_pred_example)
tmp = pd.DataFrame(cf_matrix).transpose()

loc_labels=np.unique(y_test_example.to_list())
fig=sns.heatmap(cf_matrix, cmap='Blues', xticklabels=loc_labels, yticklabels=loc_labels, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('PREDICTION \n #preds')
```
![Figure_1](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/0bc8cc67-d1fb-448e-a740-e9e85a8108a0)

# plot multiple heatmap
```python
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
kwargs = {
'cbar': False,
'linewidths': 0.2,
'linecolor': 'white',
'annot': True}

cf_matrix = confusion_matrix(y_test_example, y_pred_example)
sns.heatmap(cf_matrix, cmap='Blues', xticklabels=loc_labels, yticklabels=loc_labels, ax=ax1, **kwargs,fmt='g')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')
ax1.title.set_text('PREDICTION \n #preds')

# Normalise
cf_matrix_normalized  = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(cf_matrix_normalized, cmap='Reds', xticklabels=loc_labels, yticklabels=loc_labels, ax=ax2, **kwargs,fmt='.0%')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')
ax2.title.set_text('PREDICTION \n Normalized for actuals -recall')
# Normalise
cf_matrix_normalized  = cf_matrix.astype('float') / cf_matrix.sum(axis=0)[np.newaxis,:]
sns.heatmap(cf_matrix_normalized, cmap='Greens', xticklabels=loc_labels, yticklabels=loc_labels, ax=ax3, **kwargs,fmt='.0%')
ax3.set_ylabel('Actual')
ax3.set_xlabel('Predicted')
ax3.title.set_text('PREDICTION \n Normalized for Predcitions -precision')
fig.tight_layout()
```
![Figure_1](https://github.com/volkangumuskaya/Personal_cheat_sheet/assets/54629964/2ee5ea83-ebe8-48fe-b39f-2fa26f980103)




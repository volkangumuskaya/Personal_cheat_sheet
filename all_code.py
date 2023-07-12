#Column selection: alt+shift+insert

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

for name in dir():
    if not name.startswith('_'):
        del locals()[name]

import numpy as np
import pandas as pd
import os

pd.options.display.width = 150
pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 100)
pd.set_option('max_colwidth',150)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
diabetes = pd.DataFrame(data= np.c_[diabetes['data'], diabetes['target']],
                     columns= diabetes['feature_names'] + ['target'])

#The name of the raw data file
file_name='air_quality.csv'

# Path of directory, used to save/load files in different sub folders based on type
dir_path = os.path.dirname(os.path.realpath(__file__))

#Load dataset from subfolder 'Raw_data'
air_q=pd.read_csv(dir_path + '\\raw_files\\' + file_name)


#Sort for convenience (not really necessary)
df=iris.copy()
data_steps=[]
data_steps.append(['iris',len(df.columns),len(df),'Raw_data'])


#replace spaces with '_'
df.columns = df.columns.str.replace(' ', '_')
data_steps.append(['iris',len(df.columns),len(df),'replace spaces'])


#replace '(' with nothing
df.columns = df.columns.str.replace('(', '')
data_steps.append(['iris',len(df.columns),len(df),'replace left paranthesis'])

df.columns = df.columns.str.replace(')', '')
data_steps.append(['iris',len(df.columns),len(df),'replace right paranthesis'])

data_steps_summary=pd.DataFrame(data_steps,columns=['File_name','nCols','nRows','Note'])

#Create some columns randomly
import random
country_list=['NL','DE','GB','FR']
df['Orig_cntry']=random.choices(country_list,k=len(df))
df['Dest_cntry']=random.choices(country_list,k=len(df))
df['int_class']=random.choices(range(0,10),k=len(df))

print(np.arange(1,11))
print(range(1,11))

for i in range(1,11):
    print(i)

for i in np.arange(1,11):
    print(i)

#CROSSTAB
pd.crosstab(df.Orig_cntry,df.Dest_cntry,dropna=False)


#VALUECOUNTS
df.Orig_cntry.value_counts(dropna=False)
df[['Orig_cntry','Dest_cntry']].value_counts(dropna=False).sort_index()

#change certain_Columns
cols_str=['Orig_cntry','Dest_cntry']

#modify certain column names
df.rename(columns=dict(zip(df[cols_str].columns, df[cols_str].columns.str.replace('_', '-'))),inplace=True)

#modify certain columns
cols_str=['Orig-cntry','Dest-cntry']
df[cols_str] =df[cols_str].apply(lambda x: x.str.strip())

#select columns based on dtype
obj_cols=list(df.select_dtypes(include=['object']).columns)
print(df[obj_cols].head())

obj_float_cols=list(df.select_dtypes(include=['object','float']).columns)
print(df[obj_float_cols].head())

#summarize
df.info(show_counts=True,memory_usage=True)

df.describe(percentiles=[0,0.1,0.25,0.5,0.75,0.9,1],include='all')
df.describe(percentiles=[0,0.1,0.25,0.5,0.75,0.9,1],include=['float','int'])
df.describe(include=['object','string'])
df.describe(include=[np.number],percentiles=[0,0.1,0.25,0.5,0.75,0.9,1])
df.describe(exclude=[np.number])

#drop_duplicates
df.drop_duplicates(subset=['int_class'])
df[df.duplicated(subset=['int_class'],keep=False)]
df[~df.duplicated(subset=['int_class'],keep='first')]
df[~df.duplicated(subset=['int_class'],keep='last')]

# count per group with a condition
df['nonNL']=(df['Dest-cntry'] != 'NL').groupby(df['Orig-cntry']).transform('sum')

#find max per group row-wise
df['max_total'] = pd.concat([df['sepal_length_cm'] + df['sepal_width_cm'],
                             df['petal_length_cm'] + df['petal_width_cm']],
                                    axis=1).max(axis=1)

#groupby example

cond_positive_target=df.target>0
df_positive_target_totals = df[cond_positive_target].groupby('int_class',observed=True).\
    agg({'sepal_length_cm': 'sum',
         'petal_length_cm': 'sum',}).\
    reset_index().\
    rename(columns={'sepal_length_cm':'sum_sepal',
                    'petal_length_cm':'sum_petal'})

#groupby sums
df['sum_sepal_lengths_per_int_class'] = df.groupby('int_class')['sepal_length_cm'].transform('sum')
df['sum_sepal_lengths_per_int_class_non_zero_target'] = df['sepal_length_cm'].where(df['target'] >0).groupby(df['int_class']).transform(
    'sum')

#compare floats
df['Compare_lengths_tol_0.1']=np.where(np.isclose(df.sepal_length_cm,df.petal_length_cm,atol=0.1),1,0)
df['Compare_lengths_tol_3']=np.where(np.isclose(df.sepal_length_cm,df.petal_length_cm,atol=3),1,0)

#rename columns
df=df.rename(columns={'Orig-cntry': "origin_country", 'Dest-cntry': "dest_country"})
df.columns

#cuts
df['cut']=pd.cut(df.sepal_length_cm,bins=[0,1,2,3,4,5,6,7,np.inf])
df['cut_lower']=pd.cut(df.sepal_length_cm,bins=[0,1,2,3,4,5,6,7,np.inf]).map(lambda x: x.left)
df['cut_upper']=pd.cut(df.sepal_length_cm,bins=[0,1,2,3,4,5,6,7,np.inf]).map(lambda x: x.right)
df['cut'].value_counts()



####################################
###############PLOTS################
####################################

#Histogram
import plotly.express as px
import plotly.io as pio
pio.renderers='svg'
df.sepal_length_cm.hist(bins=30, facecolor='green', alpha=0.75)

#Histogram detailed with px
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

# Box plot with plotly
px.box(df,\
       x="origin_country", y="sepal_length_cm", color="target",\
       facet_col='dest_country',\
       # category_orders={'DT_name':['dt_before_postcodes','dt_before_corridor','dt']},\
       points='all').show()

# Bar abd scatter chart together
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

#save plotly figure
import plotly.offline
plotly.offline.plot(fig,filename='images\\'+'test.html',auto_open=False)

# Scatter plot with size and custom markers
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
# fig.update_layout(coloraxis_colorbar=dict(orientation="h"))
# fig.update_layout(legend_orientation="h")
# fig.layout.update(xaxis_range=[0, 150])
# fig.layout.update(yaxis_range=[0, 8])
fig.show()

#Subplots with different types
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

#Scatter plot with ordered x-axis

x_categories= df.sort_values('int_class').flower_id.to_list()

fig=px.scatter(df,x='flower_id',y='petal_length_cm',
               size='int_class',color='int_class',color_continuous_scale='ylorrd')
fig.update_xaxes(type="category",categoryorder='array', categoryarray= x_categories)
fig.update_xaxes(tickangle=90)
fig.update_traces(marker=dict(line=dict(width=1.5, color='black')))
fig.show()

##Correlations
from scipy.stats import pearsonr
pearsonr(df.sepal_length_cm, df.petal_length_cm)
df.sepal_length_cm.corr(df.petal_length_cm)

#weighted correlation
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

#normality
from scipy import stats
norm_test_arr = np.random.randn(100000)
stats.shapiro(norm_test_arr)


#Train model
from sklearn import datasets
import numpy as np
import pandas as pd

# Load the iris dataset
iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
df=iris.copy()
df['target']=df['target'].astype('category')

# Train LGBM model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


lgbm_params = {
    'n_estimators': 10,  # 100, opt 1000
    'max_depth': 6,  # 6, opt 14
    'learning_rate': 0.5,  # 0.3
    'reg_alpha': 0.5,  # none
    'reg_lambda': 0,  # none,
    # 'monotone_constraints': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #This is used when we want monotonic constraints for example for regression wrt a feature
}

features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target = "target"

#Change string and object typ[e columns to catgeory for LGBM
for col in df.columns:
    col_type = df[col].dtype
    if col_type == 'object' or col_type.name == 'string':
        df[col] = df[col].astype('category')
df.dtypes

X = df[features].copy()  # Features table
y = df[target]  # Target table (Natural logarithm of the target is used!)

# Split randomly
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.20,random_state=42)

# Train model
# lgbr = lgb.LGBMRegressor(**lgbm_params)  # lgbr.get_params()
lgbr = lgb.LGBMClassifier(**lgbm_params)  # lgbr.get_params()
lgbr.fit(X_train, Y_train, eval_set=(X_test, Y_test), feature_name='auto', categorical_feature='auto',verbose=1)

# predictions on log scale
pred_test = lgbr.predict(X_test)
pred_train = lgbr.predict(X_train)

pred_test_df = pd.DataFrame(pred_test, index=X_test.index)

#Accuracy
acc_test=(pred_test==Y_test).sum()/len(Y_test)
acc_train=(pred_train==Y_train).sum()/len(Y_train)

# Collect Performance metrics on the trained model (not meaning for classification but anyway))
# R2_test = lgbr.score(X_test, Y_test)  # R2 on test set
# MAPE = np.mean((Y_test - pred_test) / Y_test)  # MAPE on original values
# MAE = (np.mean((Y_test - pred_test)))  # MAE on test values in percentage

#Hyperparameter tuning
search_params = {
    'n_estimators' : [5,10,50],  # 100
    'max_depth' : [6,10,15],  # 6
    'learning_rate' : [0.3,0.5],  # 0.3
    }

#create confusion matrix
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix,confusion_matrix
# LOG Combined_confusion_table regression

classificationReport = classification_report(Y_test,
                                             pred_test, output_dict=True,
                                             zero_division=1)
tmp = pd.DataFrame(classificationReport).transpose()


#plot heatmap
import seaborn as sns
kwargs = {
    'cbar': False,
    'linewidths': 0.2,
    'linecolor': 'white',
    'annot': True}

cf_matrix = confusion_matrix(Y_test, pred_test)
loc_labels=np.unique(Y_test.to_list())
fig=sns.heatmap(cf_matrix, cmap='Blues', xticklabels=loc_labels, yticklabels=loc_labels, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('PREDICTION \n #preds')

#plot multiple heatmap
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
kwargs = {
'cbar': False,
'linewidths': 0.2,
'linecolor': 'white',
'annot': True}

cf_matrix = confusion_matrix(Y_test, pred_test)
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
loc_labels=np.unique(y_test_example.to_list())
fig=sns.heatmap(cf_matrix, cmap='Blues', xticklabels=loc_labels, yticklabels=loc_labels, **kwargs, fmt='g')
fig.set_ylabel('Actual')
fig.set_xlabel('Predicted')
fig.title.set_text('PREDICTION \n #preds')

#plot multiple heatmap
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

# Randomized search
lgb_CV = RandomizedSearchCV(estimator=lgb.LGBMClassifier(), param_distributions=search_params,
                            scoring='r2', cv=4, verbose=10, n_iter=12)
lgb_CV.fit(X_train, Y_train)
RandomizedSearchCV_df=pd.DataFrame(lgb_CV.cv_results_).sort_values('rank_test_score')

# Grid search
lgb_CV = GridSearchCV(estimator=lgb.LGBMClassifier(), param_grid=search_params,
                                            scoring='r2', cv=5,verbose=10)
lgb_CV.fit(X_train, Y_train)
GridSearchCV_df=pd.DataFrame(lgb_CV.cv_results_).sort_values('rank_test_score')




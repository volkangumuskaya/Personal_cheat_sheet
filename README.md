# Personal_cheat_sheet
Some code blocks I have used over time

# Initialize script

```python
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

```

# Import datasets from ```sklearn```

```python
from sklearn import datasets

# Load the iris dataset
iris = datasets.load_iris()
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
diabetes = pd.DataFrame(data= np.c_[diabetes['data'], diabetes['target']],
                     columns= diabetes['feature_names'] + ['target'])
```

# Load file from local

```python
#The name of the raw data file
file_name='air_quality.csv'

# Path of directory, used to save/load files in different sub folders based on type
dir_path = os.path.dirname(os.path.realpath(__file__))

#Load dataset from subfolder 'Raw_data'
air_q=pd.read_csv(dir_path + '\\raw_files\\' + file_name)

```

# Create ```df``` and track changes through a list

```python

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

```

# Create some columns randomly
```python
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
```

# crosstab
```python
pd.crosstab(df.Orig_cntry,df.Dest_cntry,dropna=False)
```

# value_counts
```python
df.Orig_cntry.value_counts(dropna=False)
df[['Orig_cntry','Dest_cntry']].value_counts(dropna=False).sort_index()
```


# change certain column names
```python
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
```

# Summarize
```python
df.info(show_counts=True,memory_usage=True)

df.describe(percentiles=[0,0.1,0.25,0.5,0.75,0.9,1],include='all')
df.describe(percentiles=[0,0.1,0.25,0.5,0.75,0.9,1],include=['float','int'])
df.describe(include=['object','string'])
df.describe(include=[np.number],percentiles=[0,0.1,0.25,0.5,0.75,0.9,1])
df.describe(exclude=[np.number])
```

# drop_duplicates and duplicated
```python
df.drop_duplicates(subset=['int_class'])
df[df.duplicated(subset=['int_class'],keep=False)]
df[~df.duplicated(subset=['int_class'],keep='first')]
df[~df.duplicated(subset=['int_class'],keep='last')]
```

## count per group with a condition (check non NL in 'Dest-cntry' group by 'Orig-cntry'
```python
df['nonNL']=(df['Dest-cntry'] != 'NL').groupby(df['Orig-cntry']).transform('sum')
```

# find max per group row-wise based on simple arithmetics
```python
df['max_total'] = pd.concat([df['sepal_length_cm'] + df['sepal_width_cm'], 
                             df['petal_length_cm'] + df['petal_width_cm']],
                                    axis=1).max(axis=1)
```

# compare floats
```python
df['Compare_lengths_tol_0.1']=np.where(np.isclose(df.sepal_length_cm,df.petal_length_cm,atol=0.1),1,0)
df['Compare_lengths_tol_3']=np.where(np.isclose(df.sepal_length_cm,df.petal_length_cm,atol=3),1,0)
```
# rename columns
```python
df=df.rename(columns={'Orig-cntry': "origin_country", 'Dest-cntry': "dest_country"})
```

# groupby and create multiple columns of sum
```python

cond_positive_target=df.target>0
df_positive_target_totals = df[cond_positive_target].groupby('int_class',observed=True).\
    agg({'sepal_length_cm': 'sum',
         'petal_length_cm': 'sum',}).\
    reset_index().\
    rename(columns={'sepal_length_cm':'sum_sepal',
                    'petal_length_cm':'sum_petal'})

#groupby sums on condition
df['sum_sepal_lengths_per_int_class'] = df.groupby('int_class')['sepal_length_cm'].transform('sum')
df['sum_sepal_lengths_per_int_class_non_zero_target'] = df['sepal_length_cm'].where(df['target'] >0).groupby(df['int_class']).transform(
    'sum')

```

# get cuts based on selected thresholds
```python
df['cut']=pd.cut(df.sepal_length_cm,bins=[0,1,2,3,4,5,6,7,np.inf])
df['cut_lower']=pd.cut(df.sepal_length_cm,bins=[0,1,2,3,4,5,6,7,np.inf]).map(lambda x: x.left)
df['cut_upper']=pd.cut(df.sepal_length_cm,bins=[0,1,2,3,4,5,6,7,np.inf]).map(lambda x: x.right)
df['sepal_length_cut'].value_counts()
```





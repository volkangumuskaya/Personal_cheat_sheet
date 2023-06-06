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
iris.describe()
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
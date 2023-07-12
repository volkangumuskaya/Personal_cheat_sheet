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

# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Parameters to fit
lgbm_params = {
    'n_estimators': 10,  # 100, opt 1000
    'max_depth': 6,  # 6, opt 14
    'learning_rate': 0.5,  # 0.3
    'reg_alpha': 0.5,  # none
    'reg_lambda': 0,  # none,
    # 'monotone_constraints': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #This is used when we want monotonic constraints for example for regression wrt a feature
}

# Define features and target
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
target = "target"

# Change string and object type columns to category for LGBM
for col in df.columns:
    col_type = df[col].dtype
    if col_type == 'object' or col_type.name == 'string':
        df[col] = df[col].astype('category')
df.dtypes

# Create X and y
X = df[features].copy()  # Features table
y = df[target]  # Target table (Natural logarithm of the target is used!)

# Split X and y randomly
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.20,random_state=42)

# Fit model using predetermined parameters
# lgbr = lgb.LGBMRegressor(**lgbm_params)  # lgbr.get_params()
lgbr = lgb.LGBMClassifier(**lgbm_params)  # lgbr.get_params()
lgbr.fit(X_train, Y_train, eval_set=(X_test, Y_test), feature_name='auto', categorical_feature='auto',verbose=1)

#classification_model = lgb.LGBMClassifier(n_estimators=500,max_depth=-1,max_bin=20,learning_rate=0.05,
                              min_data_in_leaf=20,num_leaves=10,scale_pos_weight=scale_pos_weight)
#classification_model.fit(X_train, y_train, eval_set=(X_validate, y_validate), feature_name='auto',
                                categorical_feature = 'auto',callbacks=[lgb.log_evaluation(0),lgb.early_stopping(early_stop_rounds_reg)])

print('feature importance by gain')
lgb.plot_importance(lgbr,importance_type='gain',figsize=(6,20),max_num_features=55)
lgbr.feature_importances_

print('feature importance by split')
lgb.plot_importance(lgbr,importance_type='split',figsize=(6,20),max_num_features=55)
lgbr.feature_importances_

# make predictions 
pred_test = lgbr.predict(X_test)
pred_train = lgbr.predict(X_train)

# make predcitions using customized threshold
pred_test_customized_threshold=np.where((lgbr.predict_proba(X_test)[:,1] >= 0.5),1,0)
pred_test_customized_threshold=np.where((lgbr.predict_proba(X_test)[:,1] >= 0.2),1,0)
pred_test_customized_threshold=np.where((lgbr.predict_proba(X_test)[:,1] >= 0.01),1,0)

# predictions as df using index of X_test
pred_test_df = pd.DataFrame(pred_test, index=X_test.index)

#Accuracy on training and test set
acc_test=(pred_test==Y_test).sum()/len(Y_test)
acc_train=(pred_train==Y_train).sum()/len(Y_train)

#create confusion matrix (as pandas dataframe)
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix,confusion_matrix
# LOG Combined_confusion_table regression

classificationReport = classification_report(Y_test,
                                             pred_test, output_dict=True,
                                             zero_division=1)
tmp = pd.DataFrame(classificationReport).transpose()

#Plot single heatmap
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

#plot Multiple heatmap
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

# Collect Performance metrics on the trained model (not meaning for classification but anyway))
# R2_test = lgbr.score(X_test, Y_test)  # R2 on test set
# MAPE = np.mean((Y_test - pred_test) / Y_test)  # MAPE on original values
# MAE = (np.mean((Y_test - pred_test)))  # MAE on test values in percentage

# Hyperparameter tuning  with RandomizedSearchCV and GridSearchCV
search_params = {
    'n_estimators' : [5,10,50],  # 100
    'max_depth' : [6,10,15],  # 6
    'learning_rate' : [0.3,0.5],  # 0.3   
    }

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

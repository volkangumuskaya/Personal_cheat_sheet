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

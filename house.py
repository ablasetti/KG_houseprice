
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb




train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.info())

y_train = np.log1p(train['SalePrice']) # da fare il log...
print("Decribe saleprice...", y_train.describe())

print(y_train)

train.drop('Id', axis=1, inplace=True)
ID = test['Id']
test.drop('Id', axis=1, inplace=True)
train.drop('SalePrice', axis=1, inplace=True)


for col in train.columns:
    print(str(col))
    if train[col].isnull().sum() > 0:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
        print('dropping ...' , col)
        

print(train.info())
print(test.info())

 #fil missing data in test
for col in test.columns:
    print(str(col))
    if test[col].isnull().sum() > 0:
        test[col].fillna(test[col].dropna().mode()[0], inplace=True)
       

# Encoding object type...
labelEnc=LabelEncoder()
for col in train.columns:
   if train[col].dtype == 'object' :
        train[col]=labelEnc.fit_transform(train[col])
        test[col]=labelEnc.fit_transform(test[col])
#
       
       
  
# Normalize
'''
col = train.columns
std_scaler = preprocessing.StandardScaler()
train = std_scaler.fit_transform(train) # train diventa un oggetto numpy....
#X_test = std_scaler.transform(X_test)
train = pd.DataFrame(train[1:,:], columns=col)  # 1st row as the column names
train.to_csv('scaledtrainset.csv', index=False) 
print(train.info())
'''


# LB 0.12682 senza scaling
'''
col = train.columns
std_scale = preprocessing.StandardScaler().fit(train[col])
train[col] = std_scale.transform(train[col])
    
std_scale = preprocessing.StandardScaler().fit(test[col])
test[col] = std_scale.transform(test[col])
'''

train.to_csv('scaledtrainset.csv', index=False)
test.to_csv('scaledtestset.csv', index=False)

################################################### START MODELING

'''
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(train, y_train)
print("LASSO RMSE: ",rmse_cv(model_lasso).mean())
coef = pd.Series(model_lasso.coef_, index =train.columns)
print(coef.sort_values())


'''
    
    
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

print("Random forest REGRESSOR...")
model_rfr = RandomForestRegressor(n_estimators=1000) # LB 0.14427
model_rfr.fit(train, y_train)
rfr_train_pred = model_rfr.predict(train)
print(rmsle(y_train, rfr_train_pred))
#Prediction
rfr_pred = np.expm1(model_rfr.predict(test))  



    
    

print("XGB REGRESSOR")

#model_xgb = xgb.XGBRegressor(n_estimators=1000) # LB 0.139
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)  # LB 0.129

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
print(rmsle(y_train, xgb_train_pred))
#Prediction
xgb_pred = np.expm1(model_xgb.predict(test))

#DATA SUBMISSION
submission = pd.DataFrame({ "Id": ID, "SalePrice": xgb_pred })
submission.to_csv('alex_sales_sub.csv', index=False)




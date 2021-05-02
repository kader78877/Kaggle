import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
pd.set_option('display.max_columns', None)

gender_submission = pd.read_csv("gender_submission.csv")
train = pd.read_csv("train.csv")
test_original = pd.read_csv("test.csv")



###### 1) Missing values ######
test = test_original
"""
print(train.info())
print(test.info())

print(train.describe())
print(test.describe())

print("Missing Values")
print(train.info())
print("pourcentage null value")
print(test.isnull().sum()/test.shape[0]*100.00)

print("pourcentage null value")
print(train.isnull().sum()/train.shape[0]*100.00)

print("pourcentage null value")
print(test.isnull().sum()/test.shape[0]*100.00)
"""
#### drop the column Cabin

train = train.drop(["Cabin"],axis=1)
test = test.drop(["Cabin"],axis=1)

#### Possible improvements

train = train.drop(["Name","Ticket"],axis=1)
test = test.drop(["Name","Ticket"],axis=1)

#### interpolate with linear approach

train.Age = train.Age.interpolate().bfill()
train.Embarked = train.Embarked.interpolate().bfill()
test.Age = test.Age.interpolate().bfill()
test.Fare = test.Fare.interpolate().bfill()

#### 2) From categorical to numerical value ####

## Column to encode : Embarked, Sex
cleanup_nums = {"Sex":{"male":1,"female":2},
                "Embarked":{"C":1,"S":2,"Q":3}}
col_cat_train = train.select_dtypes(include=['object']).copy()
col_cat_train = col_cat_train.replace(cleanup_nums)
col_cat_test = test.select_dtypes(include=['object']).copy()
col_cat_test = col_cat_test.replace(cleanup_nums)

train.Sex,train.Embarked =col_cat_train.Sex,col_cat_train.Embarked 
test.Sex,test.Embarked =col_cat_test.Sex,col_cat_test.Embarked

#### Apply xgboost algorithm
train.set_index('Survived',inplace=True)
train.reset_index(inplace=True)
X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

# Data Interface
data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
X_test = xgb.DMatrix(data=test)

# Setting parameters
params = {"objective":"reg:squarederror",'colsample_bytree': 1,'learning_rate': 0.1,
                'max_depth': 3, 'alpha': 10}

# Training

# First approach
"""
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=50)
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()
y_pred = xg_reg.predict(X_test)
"""
# Second approach
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
test = test.to_numpy()
model = xgb.XGBClassifier(use_label_encoder=False,n_estimators=100,max_depth=8,learning_rate=0.1,subsample=0.5)
train_model = model.fit(X_train,y_train)
pred = train_model.predict(test)


df_submission = pd.DataFrame(test_original.PassengerId)
df_submission["Survived"] = pd.DataFrame(pred,columns=['Survived'])
df_submission.to_csv('submission',index=False)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('eda_data.csv')

#Choose Relevent columns
print(data.columns)
data_model = data[['Rating', 'Type of ownership', 'Industry', 'Sector', 'Revenue','num_comp',
               'hourly', 'employer_provided','job_state', 'same_state', 'age', 'python_yn',
               'spark', 'aws', 'job_simp', 'seniority', 'desc_len', 'avg_salary']]

print(data_model.head(5))

#get dummy data
data_dummy = pd.get_dummies(data_model)
print(data_dummy.head(5))

#Splitting data into Training, testing, validation set
from sklearn.model_selection import train_test_split
X = data_dummy.drop('avg_salary', axis=1)
y = data_dummy.avg_salary.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=60)

#Model -1 : Multi-linear Regression
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
print(model.fit().summary())

#Linear Regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
lm = LinearRegression()
lm.fit(X_train, y_train)

lm_mean = np.mean(cross_val_score(lm,X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
print("Mean of Linear regression: ", lm_mean)
#Model -2 : Least Absolute Shrinkage and Selection Operator(LASSO) Regression
lm_1 = Lasso()
lm_1_mean = np.mean(cross_val_score(lm_1,X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
lm_1.fit(X_train,y_train)
print("Mean of Lasso Regression: ", lm_1_mean)

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/10)
    lml = Lasso(alpha=(i/10))
    error.append(lm_1_mean)

plt.plot(alpha, error)

err = tuple(zip(alpha,error))
data_err = pd.DataFrame(err, columns=['alpha', 'error'])
data_err[data_err.error == max(data_err.error)]

#Model -3 : Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf_mean = np.mean(cross_val_score(rf,X_train, y_train, scoring='neg_mean_absolute_error', cv=3))
print("Random Forest Regressor Mean: ", rf_mean)
#Model -4 : GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf,parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)
gs.best_score_
gs.best_estimator_

#Test Ensembles
lm_pred = lm.predict(X_test)
lm_1_pred = lm_1.predict(X_test)
rf_pred = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,lm_pred)
mean_absolute_error(y_test,lm_1_pred)
mean_absolute_error(y_test,rf_pred)

mae = mean_absolute_error(y_test,(lm_pred+rf_pred)/2)
print(mae)

import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(X_test.iloc[1,:])).reshape(1,-1))[0]

list(X_test.iloc[1,:])
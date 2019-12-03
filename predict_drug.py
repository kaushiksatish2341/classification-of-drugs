import numpy as np 
import pandas as pd
my_data = pd.read_csv("drug200.csv")

X=my_data[['Age','Sex','BP','Cholesterol','Na_to_K']].values
y=my_data['Drug'].values
print('***********')
print(X[:5])
print('***********')
print(y[:5])
print('***********')

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
#le_sex.fit(['F','M'])
X[:,1] = le_sex.fit_transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])
print('***********')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=4)
print(X_train.shape)
print('***********')

from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(max_depth=4)
dc.fit(X_train,y_train)
pred_X_test=dc.predict(X_test)
print(pred_X_test[:5])
print('***********')
print(y_test[:5])
print('***********')
#print('Accuracy of train : ', np.sqrt(((y_test - pred_X_test)*(y_test , pred_X_test))/X_test.shape))
from sklearn import metrics
print('Accuracy of test : ',metrics.accuracy_score(y_test , pred_X_test))
print('Accuracy of train : ',metrics.accuracy_score(y_train , dc.predict(X_train)))

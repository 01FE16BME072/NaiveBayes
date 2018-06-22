import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv('cancer.csv')

dataframe.replace('?',-99999,inplace = True)
dataframe.drop(['id'],1)

X = dataframe.drop(['CLass'],1)
Y = dataframe['CLass']

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size = 0.25)

classifier = GaussianNB()

classifier.fit(X_train,Y_train)
print(classifier.score(X_test,Y_test))
# print(20*'#')
# print(Y_test)
print(confusion_matrix(Y_test,classifier.predict(X_test)))

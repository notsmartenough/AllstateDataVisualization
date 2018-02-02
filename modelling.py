import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#collecting the 14 cont column names and 116 cat column names
cols_cat = train.iloc[:,1:117].columns
cols_cont = train.iloc[:,117:131].columns

#train_skew = train.skew()
#ntrain_skew = np.log1p(train['loss'])
#ptrain_skew = np.power(train['loss'], 1/4)

#first make the binary cat columns into 0 and 1
#first 72 cat columns have only 'A' or 'B' . Let us put 1 for A and 0 for B
for i in range(71):
    train[cols_cat[i]] = np.where(train[cols_cat[i]]=='A',1,0)
    
for i in range(71):
    test[cols_cat[i]] = np.where(test[cols_cat[i]]=='A',1,0)
    
#store the loss label into another target so that we can make both test and train of same dimensions
target = pd.DataFrame()
target['loss'] = train['loss']
#drop the loss column
train = train.drop('loss', axis=1)

#since it's only going to get uglier, I will combine test and train for these steps
combined = train.append(test)




#dummy encoding all categorical values
for i in range(44):
    dummies1 = pd.get_dummies(combined[cols_cat[i+71]], prefix=cols_cat[i+72])
    combined = pd.concat([combined,dummies1], axis=1)
    combined = combined.drop(cols_cat[i+71], axis=1)
 
#looks like I missed one column, no issue
combined = combined.drop(cols_cat[115], axis=1)   

#817 columns is very big and deleting one dummy variable per encoding to prevent dummy variable trap is not nearly enough
#instead I will try to look at feature importances, this is a regression problem so lets first try random forest regressor
#split train and test back, first 188318 rows are the training set
train = combined.iloc[:188318]
test = combined.iloc[188318:]

#I plan to determine the feature importances of the dummies created and handle the cont features manually or otherwise
train_cat = train.iloc[:,1:72]
train_cat2 = train.iloc[:,86:]
#pd.options.display.max_columns = 200
train_cat = pd.concat([train_cat, train_cat2], axis=1)

#I plan to determine the feature importances of the dummies created and handle the cont features manually or otherwise
#not enough RAM, sad. So I have to try other ways for these regressors. Maybe try the xgboost I have heard so much about

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(train)
x_test = sc.transform(test)

#I'll just throw it into an ANN first
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initialize the ANN
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer="he_normal", activation="elu", input_dim=771))
#dropout to reduce overfitting
classifier.add(Dropout(rate = 0.3))
classifier.add(Dense(units=6, kernel_initializer="he_normal", activation="elu"))
classifier.add(Dropout(rate = 0.2))
#output layer
classifier.add(Dense(units=1, kernel_initializer="he_normal", input_dim=771))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#fitting ANN
#batch size and epoch needs tuning as gridsearch would take forever in my machine
classifier.fit(train, target, batch_size = 10, epochs = 100)

#correct the skew and reduce features



#prediction
#y_pred = classifier.predict(x_test)
#y_pred = (y_pred > 0.5) #this is done for confusion matrix below

#confusion matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test,y_pred)


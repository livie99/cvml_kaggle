
import logging
import numpy as np
import pandas as pd
from sklearn import svm

from sklearn.metrics import classification_report

train_data = pd.read_excel(r'../au-ece-cvml2022/Train_data.xls')
validation_data = pd.read_excel(r'../au-ece-cvml2022/Validation_data.xls')
test_data = pd.read_excel(r'../au-ece-cvml2022/Test_data.xls')

train_mean_0 = round(train_data[train_data['MIG_group']==0].mean())
train_mean_1 = round(train_data[train_data['MIG_group']==1].mean())

validation_mean_0 = round(validation_data[validation_data['MIG_group']==0].mean())
validation_mean_1 = round(validation_data[validation_data['MIG_group']==1].mean())

test_mean = round(test_data.mean())


for col in train_data.columns:
    if col == 'Number':
        continue
    train_data[col][train_data['MIG_group']==True] = train_data[col][train_data['MIG_group']==True].replace(np.nan,train_mean_0[col])
    train_data[col][train_data['MIG_group']==False] = train_data[col][train_data['MIG_group']==False].replace(np.nan,train_mean_1[col])    
    # train_data[col] = train_data[col].replace(np.nan,train_mean_0[col])
for col in validation_data.columns:
    if col == 'MIG_group':
        continue
    validation_data[col][validation_data['MIG_group']==True] = validation_data[col][validation_data['MIG_group']==True].replace(np.nan,validation_mean_0[col])
    validation_data[col][validation_data['MIG_group']==False] = validation_data[col][validation_data['MIG_group']==False].replace(np.nan,validation_mean_1[col])
    # validation_data[col] = validation_data[col].replace(np.nan,validation_mean_0[col])
for col in test_data.columns:
    test_data[col] = test_data[col].replace(np.nan,test_mean[col])




train_features = train_data[train_data.columns[:-1]]
train_labels = train_data['MIG_group']
validation_features = validation_data[validation_data.columns[:-1]]
validation_labels = validation_data['MIG_group']

    
# print(train_features[col])
model = svm.SVC(kernel='rbf')
model.fit(train_features, train_labels)
FORMAT = '%(message)s'
logging.basicConfig(filename='./log.txt', format = FORMAT, level=logging.INFO)
# logging.info(f"Training Set Score : {model.score(train_feature, train_labels) * 100} %")
# logging.info(f"Validation Set Score : {model.score(validation_feature, validation_labels) * 100} %")
# # Printing classification report of classifier on the test set set data
logging.info(f"Model Classification Report : \n{classification_report(train_labels, model.predict(train_features))}")
lables = model.predict(test_data)


result = pd.DataFrame({
    'ID':test_data['Number'],
    'Label':lables})
result.to_csv('test_result.csv', index=False)
print(result)


from operator import index
import numpy as np
import pandas as pd
import logging
from imblearn.over_sampling import SMOTE
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

FORMAT = '%(message)s'
logging.basicConfig(filename='./preprocessing_log.txt', format = FORMAT, level=logging.INFO)
pd.set_option('display.max_columns', None)

train_data = pd.read_excel(r'../au-ece-cvml2022/Train_data.xls')
validation_data = pd.read_excel(r'../au-ece-cvml2022/Validation_data.xls')
test_data = pd.read_excel(r'../au-ece-cvml2022/Test_data.xls')

train_mean_0 = round(train_data[train_data['MIG_group']==0].mean())
train_mean_1 = round(train_data[train_data['MIG_group']==1].mean())

train_mean = round(train_data.mean())


for col in train_data.columns:
    if col == 'Number':
        continue
    train_data[col][train_data['MIG_group']==True] = train_data[col][train_data['MIG_group']==True].replace(np.nan,train_mean_0[col])
    train_data[col][train_data['MIG_group']==False] = train_data[col][train_data['MIG_group']==False].replace(np.nan,train_mean_1[col])    
    # train_data[col] = train_data[col].replace(np.nan,train_mean_0[col])
for col in validation_data.columns:
    if col == 'MIG_group':
        continue
    validation_data[col][validation_data['MIG_group']==True] = validation_data[col][validation_data['MIG_group']==True].replace(np.nan,train_mean_0[col])
    validation_data[col][validation_data['MIG_group']==False] = validation_data[col][validation_data['MIG_group']==False].replace(np.nan,train_mean_1[col])
    # validation_data[col] = validation_data[col].replace(np.nan,validation_mean_0[col])
for col in test_data.columns:
    test_data[col] = test_data[col].replace(np.nan,train_mean[col])


def smote(data_set):
    X = data_set.loc[:, train_data.columns != 'MIG_group']
    y = data_set.loc[:, train_data.columns == 'MIG_group']

    os = SMOTE(random_state=0)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    columns = X.columns
    os_data_X,os_data_y=os.fit_resample(X, y)
    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
    os_data_y= pd.DataFrame(data=os_data_y,columns=['MIG_group'])
    # we can Check the numbers of our data
    logging.info("length of oversampled data:\n%s",len(os_data_X))
    logging.info("Number of no subscription in oversampled data:\n%s",len(os_data_y[os_data_y['MIG_group']==0]))
    logging.info("Number of subscription:\n%s",len(os_data_y[os_data_y['MIG_group']==1]))
    logging.info("Proportion of no subscription data in oversampled data:\n%s",len(os_data_y[os_data_y['MIG_group']==0])/len(os_data_X))
    logging.info("Proportion of subscription data in oversampled data:\n%s",len(os_data_y[os_data_y['MIG_group']==1])/len(os_data_X))
    return os_data_X, os_data_y

train_features, train_labels = smote(train_data)
validation_features, validation_labels = smote(validation_data)

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=8)
rfe = rfe.fit(train_features, train_labels.values.ravel())
logging.info(rfe.support_)
logging.info(rfe.ranking_)

columns = ['Sex', 'Tscore', 'Cem_ucem', 'TKA', 'side', 'former_alcoholabuse', 'smoker', 'former_smoker'] 
train_features = train_features[columns]
validation_features = validation_features[columns]


logit_model=sm.Logit(train_labels,train_features)
result=logit_model.fit()
logging.info(result.summary2())

# columns = ['Tscore', 'side', 'former_alcoholabuse', 'smoker'] 
# train_features = train_features[columns]
# validation_features = validation_features[columns]

model = svm.SVC(kernel='rbf')
model.fit(train_features, train_labels)

logging.info(f"Model Classification Report : \n{classification_report(train_labels, model.predict(train_features))}")
logging.info(f"Model Classification Report : \n{classification_report(validation_labels, model.predict(validation_features))}")
lables = model.predict(test_data[columns])


result = pd.DataFrame({
    'ID':test_data['Number'],
    'Label':lables})
result.to_csv('test_result.csv', index=False)

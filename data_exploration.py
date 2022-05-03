import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
FORMAT = '%(message)s'
logging.basicConfig(filename='./exploring_log.txt', format = FORMAT, level=logging.INFO)
pd.set_option('display.max_columns', None)

train_data = pd.read_excel(r'../au-ece-cvml2022/Train_data.xls')
validation_data = pd.read_excel(r'../au-ece-cvml2022/Validation_data.xls')
test_data = pd.read_excel(r'../au-ece-cvml2022/Test_data.xls')

general_distribution = train_data['MIG_group'].value_counts()
sns.countplot(x = 'MIG_group', data=train_data, palette='hls')
plt.savefig('general_distribution')
# plt.show()
logging.info('general_distribution: \n%s',general_distribution)

group_mean = train_data.groupby('MIG_group').mean()
logging.info('group_mean: \n%s',group_mean)

pd.crosstab(train_data.Age,train_data.MIG_group).plot(kind='line')
plt.title('Line Graph of Age')
plt.xlabel('Age')
plt.ylabel('Number of people')
plt.savefig('lg_age')
# plt.show()

pd.crosstab(train_data.Sex,train_data.MIG_group).plot(kind='bar')
plt.title('Histogram of Sex')
plt.xlabel('Sex')
plt.ylabel('Number of people')
plt.savefig('hist_sex')
# plt.show()

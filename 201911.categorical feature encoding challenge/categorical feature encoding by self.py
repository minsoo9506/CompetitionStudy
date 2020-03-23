# binary, norminal, ordinal , time data 를 encoding 하는 방법
# 2019.12.21~ 시작
# 참고 커널
# 1. https://www.kaggle.com/c/cat-in-the-dat/notebooks?sortBy=voteCount&group=everyone&pageSize=20&competitionId=14999
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import base

df_train = pd.read_csv('C:/Users/송민수/Desktop/kaggle/categorical feature encoding challenge/cat-in-the-dat/train.csv')
df_test = pd.read_csv('C:/Users/송민수/Desktop/kaggle/categorical feature encoding challenge/cat-in-the-dat/test.csv')

print('train {} rows {} columns'.format(df_train.shape[0], df_train.shape[1]))
print('test {} rows {} columns'.format(df_test.shape[0], df_test.shape[1]))

df_train.info()

X = df_train.drop(['target'], axis = 1)
y = df_train['target']

y.value_counts().plot.bar()

# Method1 : Label encoding
# 그냥 전부 다 숫자로 바꾸는 것

from sklearn.preprocessing import LabelEncoder # 불러오기

%%time
train = pd.DataFrame()
label = LabelEncoder()
for col in X.columns : 
    if X[col].dtype == 'object' :
        train[col]=label.fit_transform(X[col]) # fit 하고 transform
    else :
        train[col] = X[col]

train.head(3)

# logistic fitting
def logistic(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pre = lr.predict(X_test)
    print('Accuracy : ', accuracy_score(y_test, y_pre))

logistic(train, y)

# Method2 : One hot encoding
from sklearn.preprocessing import OneHotEncoder

%%time
one = OneHotEncoder()
one.fit(X)
train = one.transform(X)
print(train.shape) # column이 엄청 커진다

logistic(train,y)

# Method3 : Feature hashing(aka the hashing trick)
# 원리는 잘 모르겠지만 차원을 축소해서 coding 한다고 한다
from sklearn.feature_extraction import FeatureHasher

%%time
X_train_hash = X.copy()
for col in X.columns : 
    X_train_hash[col] = X[col].astype('str')
hashing = FeatureHasher(input_type = 'string')
train = hashing.transform(X_train_hash.values)

print(train.shape)

logistic(train,y)

# Method 4 : Encoding categories with dataset statistics
# 잘 모르겠음
# 각 category들을 숫자로 encoding 할 때, 해당 category가 나온 빈도수를
# 이용하여 encoding 한다.

%%time

X_train_stat = X.copy()
for col in X_train_stat.columns:
    if X_train_stat[col].dtype == 'object':
        X_train_stat[col] = X_train_stat[col].astype('category')
        counts = X_train_stat[col].value_counts()
        counts = counts.sort_index()
        counts = counts.fillna(0)
        counts += np.random.rand(len(counts)) / 1000
        X_train_stat[col].cat.categories = counts


X_train_stat.head()

# Encoding cyclic features
# feature가 cyclic한 경우 (month, day ..)
# sin, cosine을 이용하여 transform
%%time

X_train_cyclic = X.copy()
columns = ['day','month']
for col in columns : 
    X_train_cyclic[col+'_sin'] = np.sin((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
    X_train_cyclic[col+'_cos'] = np.cos((2*np.pi*X_train_cyclic[col])/max(X_train_cyclic[col]))
X_train_cyclic = X_train_cyclic.drop(columns, axis = 1)
 
X_train_cyclic[['day_sin','day_cos']].head()

one = OneHotEncoder()
one.fit(X_train_cyclic)
train = one.transform(X_train_cyclic)

logistic(train,y)

# Method 5 : Target encoding
# target이 예를 들어, male, female으로 이루어져 있는 경우
# 해당 target의 비율대로 encoding
%%time
X_target = df_train.copy()
X_target['day'] = X_target['day'].astype('object')
X_target['month'] = X_target['month'].astype('object')
for col in X_target.columns : 
    if X_target[col].dtype == 'object' : 
        target = dict(X_target.groupby(col)['target'].agg('sum') /X_target.groupby(col)['target'].agg('count') )
        X_target[col] = X_target[col].replace(target).values
import numpy as np
import pandas as pd
import math
from inspect import currentframe, getframeinfo

print((getframeinfo(currentframe())).lineno)
df = pd.read_csv("parkersburg_all_parts", header=None)
print(getframeinfo(currentframe()).lineno)
df = df.sort_values([0,1], ascending=True)
print(getframeinfo(currentframe()).lineno)
df = df.drop([1], axis=1)
print(getframeinfo(currentframe()).lineno)
#df_counts = df.apply(pd.value_counts)
print(getframeinfo(currentframe()).lineno)
print(getframeinfo(currentframe()).lineno)
df.head()
print(getframeinfo(currentframe()).lineno)
df2 = df.iloc[:,1:].astype('category')
df2 = df2.apply(lambda x: x.cat.codes)
df3 = pd.concat((df.iloc[:,:1],df2),axis=1)
users = df3[0].drop_duplicates()
users = users.sample(frac=1)
first_80 = len(users)
first_80 = math.ceil(first_80*0.8)
train = df3
print(train.shape)
print('Train number of groups', train.shape)
print(train.tail())
X=train.groupby([0]).apply(lambda x: x.iloc[:,2:].values.flatten()) 
print(getframeinfo(currentframe()).lineno)
print("X.shape is", X.shape)
indexer = np.arange(95)[None, :] + 1*np.arange(15675-94)[:, None]
X.apply(lambda x: print(x.shape))
print("Indexer shape is", indexer.shape)
X = X.apply(lambda x: pd.DataFrame(x[indexer]))
X_lists = []
Big_Data = pd.DataFrame()
for i in range(X.shape[0]):
    if(i % 50 == 0):
        print(i)
    X_lists.append(X.values[i])
    #Big_Data = pd.concat((Big_Data,X.values[i]))
Big_Data = pd.concat(X_lists)
print(Big_Data.shape)
print(Big_Data.head())
print(getframeinfo(currentframe()).lineno)
print(int(Big_Data.shape[0]*0.8))
print(Big_Data[:int(Big_Data.shape[0]*0.8)].shape)
Big_Data[:int(Big_Data.shape[0]*0.8)].to_csv('/media/dready/Data/BigData/Big_Data_without_mapping_parkersburg_train.csv')

test = Big_Data[int(Big_Data.shape[0]*0.8):]
print(test.shape)
test.to_csv('/media/dready/Data/BigData/Big_Data_without_mapping_parkersburg_test.csv')
input('input')
print(getframeinfo(currentframe()).lineno)

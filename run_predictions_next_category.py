import numpy as np
np.set_printoptions(threshold=np.nan, linewidth=200)
import pandas as pd
from inspect import currentframe, getframeinfo
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Dropout, LSTM, CuDNNLSTM
print(getframeinfo(currentframe()).lineno)
chunksize = 10 ** 6
small_chunksize = 0.2 * chunksize
number_of_chunks = 100
index = 0
df_list = []
for chunk in pd.read_csv('/media/dready/Data/BigData/Big_Data_without_mapping_parkersburg_train.csv', chunksize=chunksize, index_col=0):
    print("Train Working on chunk number - Start", index)
    print("Train Working on chunk number - Start", index)
    df_list.append(chunk.copy())
    print("Train Working on chunk number - End", index)
    index +=1
    if(index > number_of_chunks):
        break
train = pd.concat(df_list)
print(train.head())
print(train.shape)


print(getframeinfo(currentframe()).lineno)

import numpy as np
from collections import Counter

x_train = train.iloc[:,:-1].values
y_train = train.iloc[:,-1].values

max_value = x_train.max()
x_train = x_train/(max_value+1)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
model = tf.keras.models.Sequential()

print(x_train.shape)
print(x_train.shape[1:])
model.add(CuDNNLSTM(256, return_sequences=True, input_shape=x_train.shape[1:]))
model.add(Dropout(0.2))
#
model.add(CuDNNLSTM(256, return_sequences=True))
model.add(Dropout(0.2))
#
model.add(CuDNNLSTM(256))
model.add(Dropout(0.1))
#
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
#
model.add(Dense((max_value+1), activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=1e-6),
              loss='sparse_categorical_crossentropy',
              metrics=['categorical_accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=2048, epochs=2) #, class_weight=class_weights)#, steps_per_epoch = 3)
model.save('little_rock.model')

index = 0
df_list = []
for chunk in pd.read_csv('/media/dready/Data/BigData/Big_Data_without_mapping_parkersburg_test.csv', chunksize=small_chunksize, index_col=0):
    print("Test Working on chunk number - Start", index)
    df_list.append(chunk.copy())
    print("Test Working on chunk number - End", index)
    index +=1
    if(index > number_of_chunks):
        break
test = pd.concat(df_list)
print(test.head())
print(test.shape)

x_test = test.iloc[:,:-1].values
x_test = x_test/(max_value+1)
x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1)) #))
y_test = test.iloc[:,-1].values
val_loss_val_acc = model.evaluate(x_test, y_test)
print(val_loss_val_acc)
print(getframeinfo(currentframe()).lineno)
classes = model.predict(x_test)
print(getframeinfo(currentframe()).lineno)
y_pred = np.argmax(classes, axis=1)
print(getframeinfo(currentframe()).lineno)
from sklearn.metrics import confusion_matrix
print(getframeinfo(currentframe()).lineno)
if(len(y_test) < 20):
    print(confusion_matrix(y_test, y_pred))
print(getframeinfo(currentframe()).lineno)
from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred, average='micro'))
print(getframeinfo(currentframe()).lineno)
#precision@1


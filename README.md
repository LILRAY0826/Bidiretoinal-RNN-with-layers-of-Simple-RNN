# Deep Learning - Bidiretoinal RNN with layers of Simple RNN

> #### **My Github with open source (Fully code):**
> https://github.com/LILRAY0826/Bidiretoinal-RNN-with-layers-of-Simple-RNN.git

> Dataset download link: http://134.208.3.118/~ccchiang/DLFA/dataset.zip


*I . Data Preprocessing*
---

***Dealing with txt file with batch sixe to get 4 outputs.***
***X = (len/batch_size, batch_size, 2, timesteps=39)***
***Y = (len/batch_size, batch_size, timesteps=39)***
```python
def get_all_data(df, batch_size):
    dfN = df.to_numpy()
    X, Y = [], []
    for i in range(math.ceil(len(dfN)/batch_size)):
        print("Getting {}/{} batch.".format(i, math.ceil(len(dfN)/batch_size)))
        X_batch, Y_batch = [], []
        for j in range(min(batch_size, len(dfN)-i*batch_size)):
            str1, str2, str3 = dfN[i*batch_size+1]
            strX1, strX2, strY = [0], [0], [] #使A,B的長度等於C的長度(steps = 39)
            for char in str1:
                strX1.append(int(char))
            for char in str2:
                strX2.append(int(char))
            for char in str3:
                strY.append(int(char))
            strX = (strX1, strX2)
            X_batch.append(strX)
            Y_batch.append(strY)
        X.append(X_batch)
        Y.append(Y_batch)
    steps = len(strX1)
    Xlen = len(X)
    X = np.array(X)
    Y = np.array(Y)

    return X, Y, steps, Xlen
```
***Return X = (batch_size, timestep, vocab_size*2)***
***Return Y = (batch_size, timestep)***
```python
# Convert array to tensor
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))

# Generate data for training
def generator(data):
    i = 0
    iteator = iter(data)
    while True:
        try:
            X, Y = next(iteator)
            i += 1
        except:
            iteator = iter(data)
            X, Y = next(iteator)
            i = 1
        X = tf.one_hot(X, vocab_size)
        X = np.concatenate((X[:,0], X[:, 1]), axis=2)
        yield (X, Y)
```

***Loading train dataset***
```python
# Loading train dataset
train_df = pd.read_csv("dataset/train.txt", header=None, delimiter=",")
X_train, Y_train, steps, Xlen_train = get_all_data(train_df, batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_gen = generator(train_dataset)
```

*II . Model Construction*
---
```python
# Hyper Parameters
batch_size = 100
vocab_size = 10
epochs = 100
LR = 0.001

# Structure
model = tf.keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(64, return_sequences=True), merge_mode='concat'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(128, return_sequences=True), merge_mode='concat'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(256, return_sequences=True), merge_mode='concat'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(vocab_size))

optimizer = keras.optimizers.Adam(learning_rate=LR)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# Training
history = model.fit(x=train_gen, epochs=epochs, steps_per_epoch=64, verbose=1)
```
***The accuracy and loss of model in epochs = 100.***
```
Epoch 1/100
64/64 [==============================] - 12s 184ms/step - loss: 0.3671 - accuracy: 0.8960
Epoch 2/100
64/64 [==============================] - 12s 182ms/step - loss: 0.2931 - accuracy: 0.9147
Epoch 3/100
64/64 [==============================] - 12s 180ms/step - loss: 0.2879 - accuracy: 0.9151
Epoch 4/100
64/64 [==============================] - 12s 187ms/step - loss: 0.2686 - accuracy: 0.9144
Epoch 5/100
64/64 [==============================] - 12s 183ms/step - loss: 0.2641 - accuracy: 0.9166
Epoch 6/100
64/64 [==============================] - 11s 179ms/step - loss: 0.2439 - accuracy: 0.9187
Epoch 7/100
64/64 [==============================] - 12s 182ms/step - loss: 0.2346 - accuracy: 0.9230
Epoch 8/100
64/64 [==============================] - 11s 178ms/step - loss: 0.2293 - accuracy: 0.9240
Epoch 9/100
64/64 [==============================] - 11s 178ms/step - loss: 0.2196 - accuracy: 0.9244
Epoch 10/100
64/64 [==============================] - 11s 179ms/step - loss: 0.2213 - accuracy: 0.9262
Epoch 11/100
64/64 [==============================] - 11s 178ms/step - loss: 0.2174 - accuracy: 0.9257
Epoch 12/100
64/64 [==============================] - 12s 191ms/step - loss: 0.2135 - accuracy: 0.9222
Epoch 13/100
64/64 [==============================] - 12s 187ms/step - loss: 0.2126 - accuracy: 0.9254
Epoch 14/100
64/64 [==============================] - 12s 183ms/step - loss: 0.2119 - accuracy: 0.9247
Epoch 15/100
64/64 [==============================] - 12s 184ms/step - loss: 0.2171 - accuracy: 0.9267
Epoch 16/100
64/64 [==============================] - 12s 186ms/step - loss: 0.2156 - accuracy: 0.9252
Epoch 17/100
64/64 [==============================] - 12s 183ms/step - loss: 0.1992 - accuracy: 0.9259
Epoch 18/100
64/64 [==============================] - 12s 192ms/step - loss: 0.1974 - accuracy: 0.9276
Epoch 19/100
64/64 [==============================] - 12s 184ms/step - loss: 0.2038 - accuracy: 0.9232
Epoch 20/100
64/64 [==============================] - 12s 185ms/step - loss: 0.1912 - accuracy: 0.9296
Epoch 21/100
64/64 [==============================] - 12s 180ms/step - loss: 0.2079 - accuracy: 0.9262
Epoch 22/100
64/64 [==============================] - 14s 211ms/step - loss: 0.1930 - accuracy: 0.9292
Epoch 23/100
64/64 [==============================] - 13s 195ms/step - loss: 0.1916 - accuracy: 0.9299
Epoch 24/100
64/64 [==============================] - 12s 191ms/step - loss: 0.2000 - accuracy: 0.9245
Epoch 25/100
64/64 [==============================] - 12s 184ms/step - loss: 0.1880 - accuracy: 0.9298
Epoch 26/100
64/64 [==============================] - 13s 199ms/step - loss: 0.1937 - accuracy: 0.9267
Epoch 27/100
64/64 [==============================] - 12s 190ms/step - loss: 0.1925 - accuracy: 0.9296
Epoch 28/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1928 - accuracy: 0.9262
Epoch 29/100
64/64 [==============================] - 12s 187ms/step - loss: 0.2001 - accuracy: 0.9237
Epoch 30/100
64/64 [==============================] - 12s 187ms/step - loss: 0.1963 - accuracy: 0.9271
Epoch 31/100
64/64 [==============================] - 12s 184ms/step - loss: 0.1937 - accuracy: 0.9272
Epoch 32/100
64/64 [==============================] - 12s 186ms/step - loss: 0.1838 - accuracy: 0.9298
Epoch 33/100
64/64 [==============================] - 12s 189ms/step - loss: 0.1945 - accuracy: 0.9257
Epoch 34/100
64/64 [==============================] - 13s 197ms/step - loss: 0.1854 - accuracy: 0.9293
Epoch 35/100
64/64 [==============================] - 12s 190ms/step - loss: 0.1766 - accuracy: 0.9356
Epoch 36/100
64/64 [==============================] - 12s 185ms/step - loss: 0.1908 - accuracy: 0.9266
Epoch 37/100
64/64 [==============================] - 12s 186ms/step - loss: 0.1898 - accuracy: 0.9292
Epoch 38/100
64/64 [==============================] - 12s 194ms/step - loss: 0.1922 - accuracy: 0.9290
Epoch 39/100
64/64 [==============================] - 12s 187ms/step - loss: 0.1851 - accuracy: 0.9303
Epoch 40/100
64/64 [==============================] - 12s 192ms/step - loss: 0.1854 - accuracy: 0.9311
Epoch 41/100
64/64 [==============================] - 13s 199ms/step - loss: 0.1781 - accuracy: 0.9315
Epoch 42/100
64/64 [==============================] - 13s 198ms/step - loss: 0.1889 - accuracy: 0.9306
Epoch 43/100
64/64 [==============================] - 13s 203ms/step - loss: 0.1795 - accuracy: 0.9356
Epoch 44/100
64/64 [==============================] - 13s 196ms/step - loss: 0.1839 - accuracy: 0.9314
Epoch 45/100
64/64 [==============================] - 14s 213ms/step - loss: 0.1818 - accuracy: 0.9321
Epoch 46/100
64/64 [==============================] - 13s 203ms/step - loss: 0.1825 - accuracy: 0.9326
Epoch 47/100
64/64 [==============================] - 12s 190ms/step - loss: 0.1785 - accuracy: 0.9341
Epoch 48/100
64/64 [==============================] - 13s 196ms/step - loss: 0.1833 - accuracy: 0.9304
Epoch 49/100
64/64 [==============================] - 13s 203ms/step - loss: 0.1861 - accuracy: 0.9318
Epoch 50/100
64/64 [==============================] - 13s 200ms/step - loss: 0.1766 - accuracy: 0.9369
Epoch 51/100
64/64 [==============================] - 12s 189ms/step - loss: 0.1776 - accuracy: 0.9313
Epoch 52/100
64/64 [==============================] - 12s 193ms/step - loss: 0.1803 - accuracy: 0.9332
Epoch 53/100
64/64 [==============================] - 12s 190ms/step - loss: 0.1777 - accuracy: 0.9355
Epoch 54/100
64/64 [==============================] - 12s 195ms/step - loss: 0.1812 - accuracy: 0.9344
Epoch 55/100
64/64 [==============================] - 13s 205ms/step - loss: 0.1872 - accuracy: 0.9323
Epoch 56/100
64/64 [==============================] - 12s 193ms/step - loss: 0.1758 - accuracy: 0.9346
Epoch 57/100
64/64 [==============================] - 12s 193ms/step - loss: 0.1696 - accuracy: 0.9369
Epoch 58/100
64/64 [==============================] - 13s 198ms/step - loss: 0.1778 - accuracy: 0.9362
Epoch 59/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1639 - accuracy: 0.9399
Epoch 60/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1803 - accuracy: 0.9377
Epoch 61/100
64/64 [==============================] - 12s 194ms/step - loss: 0.1593 - accuracy: 0.9424
Epoch 62/100
64/64 [==============================] - 13s 195ms/step - loss: 0.1671 - accuracy: 0.9430
Epoch 63/100
64/64 [==============================] - 13s 197ms/step - loss: 0.1716 - accuracy: 0.9397
Epoch 64/100
64/64 [==============================] - 13s 202ms/step - loss: 0.1632 - accuracy: 0.9426
Epoch 65/100
64/64 [==============================] - 12s 190ms/step - loss: 0.1669 - accuracy: 0.9428
Epoch 66/100
64/64 [==============================] - 13s 200ms/step - loss: 0.1667 - accuracy: 0.9431
Epoch 67/100
64/64 [==============================] - 13s 198ms/step - loss: 0.1645 - accuracy: 0.9435
Epoch 68/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1679 - accuracy: 0.9389
Epoch 69/100
64/64 [==============================] - 12s 192ms/step - loss: 0.1669 - accuracy: 0.9427
Epoch 70/100
64/64 [==============================] - 13s 205ms/step - loss: 0.1621 - accuracy: 0.9454
Epoch 71/100
64/64 [==============================] - 13s 206ms/step - loss: 0.1592 - accuracy: 0.9461
Epoch 72/100
64/64 [==============================] - 14s 221ms/step - loss: 0.1625 - accuracy: 0.9453
Epoch 73/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1629 - accuracy: 0.9433
Epoch 74/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1540 - accuracy: 0.9456
Epoch 75/100
64/64 [==============================] - 12s 194ms/step - loss: 0.1584 - accuracy: 0.9446
Epoch 76/100
64/64 [==============================] - 12s 194ms/step - loss: 0.1613 - accuracy: 0.9457
Epoch 77/100
64/64 [==============================] - 12s 193ms/step - loss: 0.1618 - accuracy: 0.9452
Epoch 78/100
64/64 [==============================] - 13s 204ms/step - loss: 0.1540 - accuracy: 0.9479
Epoch 79/100
64/64 [==============================] - 13s 208ms/step - loss: 0.1558 - accuracy: 0.9469
Epoch 80/100
64/64 [==============================] - 13s 199ms/step - loss: 0.1583 - accuracy: 0.9460
Epoch 81/100
64/64 [==============================] - 13s 210ms/step - loss: 0.1638 - accuracy: 0.9445
Epoch 82/100
64/64 [==============================] - 13s 199ms/step - loss: 0.1564 - accuracy: 0.9465
Epoch 83/100
64/64 [==============================] - 12s 193ms/step - loss: 0.1612 - accuracy: 0.9453
Epoch 84/100
64/64 [==============================] - 12s 194ms/step - loss: 0.1562 - accuracy: 0.9469
Epoch 85/100
64/64 [==============================] - 13s 203ms/step - loss: 0.1661 - accuracy: 0.9416
Epoch 86/100
64/64 [==============================] - 14s 211ms/step - loss: 0.1573 - accuracy: 0.9448
Epoch 87/100
64/64 [==============================] - 13s 204ms/step - loss: 0.1539 - accuracy: 0.9458
Epoch 88/100
64/64 [==============================] - 12s 193ms/step - loss: 0.1648 - accuracy: 0.9456
Epoch 89/100
64/64 [==============================] - 12s 187ms/step - loss: 0.1524 - accuracy: 0.9489
Epoch 90/100
64/64 [==============================] - 13s 203ms/step - loss: 0.1501 - accuracy: 0.9470
Epoch 91/100
64/64 [==============================] - 13s 199ms/step - loss: 0.1577 - accuracy: 0.9459
Epoch 92/100
64/64 [==============================] - 12s 186ms/step - loss: 0.1592 - accuracy: 0.9457
Epoch 93/100
64/64 [==============================] - 12s 191ms/step - loss: 0.1687 - accuracy: 0.9425
Epoch 94/100
64/64 [==============================] - 13s 200ms/step - loss: 0.1639 - accuracy: 0.9440
Epoch 95/100
64/64 [==============================] - 12s 189ms/step - loss: 0.1590 - accuracy: 0.9438
Epoch 96/100
64/64 [==============================] - 11s 177ms/step - loss: 0.1539 - accuracy: 0.9464
Epoch 97/100
64/64 [==============================] - 11s 173ms/step - loss: 0.1618 - accuracy: 0.9444
Epoch 98/100
64/64 [==============================] - 11s 172ms/step - loss: 0.1510 - accuracy: 0.9479
Epoch 99/100
64/64 [==============================] - 11s 179ms/step - loss: 0.1573 - accuracy: 0.9476
Epoch 100/100
64/64 [==============================] - 12s 190ms/step - loss: 0.1461 - accuracy: 0.9496
```
***The final loss = 0.1461, accuracy = 0.9496.***

![](https://i.imgur.com/kUVr5m3.png)
![](https://i.imgur.com/SWwC0Z2.png)



*III . Predicting*
---
***Predict the test.txt and output a C.txt file.***
```python
# Loading test dataset
df = pd.read_csv("dataset/test.txt", header=None, delimiter=",")
X_test, Xlen_test = get_test_data(df, steps)
path = 'C.txt'
f = open(path, "w")
output = []

# Predicting
Y = model.predict(X_test)
Y = tf.argmax(Y, axis=-1)
Y = np.array(Y)
for i in range(len(Y)):
    Y1 = Y[i].tolist()
    Y1 = [str(p) for p in Y1]
    Y1 = ''.join(Y1)
    print(Y1, file=f)
f.close()
```

*IV . Full Code*
---
```python
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# Hyper Parameters
batch_size = 100
vocab_size = 10
epochs = 100
LR = 0.001


def get_all_data(df, batch_size):
    dfN = df.to_numpy()
    X, Y = [], []
    for i in range(math.ceil(len(dfN)/batch_size)):
        print("Getting {}/{} batch.".format(i, math.ceil(len(dfN)/batch_size)))
        X_batch, Y_batch = [], []
        for j in range(min(batch_size, len(dfN)-i*batch_size)):
            str1, str2, str3 = dfN[i*batch_size+1]
            strX1, strX2, strY = [0], [0], []
            for char in str1:
                strX1.append(int(char))
            for char in str2:
                strX2.append(int(char))
            for char in str3:
                strY.append(int(char))
            strX = (strX1, strX2)
            X_batch.append(strX)
            Y_batch.append(strY)
        X.append(X_batch)
        Y.append(Y_batch)
    steps = len(strX1)
    Xlen = len(X)
    X = np.array(X)
    Y = np.array(Y)

    return X, Y, steps, Xlen


def generator(data):
    i = 0
    iteator = iter(data)
    while True:
        try:
            X, Y = next(iteator)
            i += 1
        except:
            iteator = iter(data)
            X, Y = next(iteator)
            i = 1
        X = tf.one_hot(X, vocab_size)
        X = np.concatenate((X[:,0], X[:, 1]), axis=2)
        yield (X, Y)


def get_test_data(df, steps):
    dfN = df.to_numpy()
    X = []
    for i in range(len(dfN)):
        print("Getting {}/{} test data.".format(i, len(dfN)))
        str1, str2 = dfN[i]
        strX1, strX2 = [], []
        for j in range(steps-len(str1)):
            strX1.append(0)
            strX2.append(0)
        for char in str1:
            strX1.append(int(char))
        for char in str2:
            strX2.append(int(char))
        strX = (strX1, strX2)
        X.append(strX)
    steps = len(strX1)
    Xlen_test = len(X)
    X = np.array(X)
    X = tf.one_hot(X, vocab_size)
    X = np.concatenate((X[:,0], X[:, 1]), axis=2)
    return X, Xlen_test


# Loading train dataset
train_df = pd.read_csv("dataset/train.txt", header=None, delimiter=",")
X_train, Y_train, steps, Xlen_train = get_all_data(train_df, batch_size)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_gen = generator(train_dataset)

# Loading test dataset
df = pd.read_csv("dataset/test.txt", header=None, delimiter=",")
X_test, Xlen_test = get_test_data(df, steps)
path = 'C.txt'
f = open(path, "w")
output = []

# Model Training
model = tf.keras.Sequential()
model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(64, return_sequences=True), merge_mode='concat'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(128, return_sequences=True), merge_mode='concat'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Bidirectional(keras.layers.SimpleRNN(256, return_sequences=True), merge_mode='concat'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(vocab_size))

optimizer = keras.optimizers.Adam(learning_rate=LR)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

history = model.fit(x=train_gen, epochs=epochs, steps_per_epoch=64, verbose=1)

# Predicting
Y = model.predict(X_test)
Y = tf.argmax(Y, axis=-1)
Y = np.array(Y)
for i in range(len(Y)):
    Y1 = Y[i].tolist()
    Y1 = [str(p) for p in Y1]
    Y1 = ''.join(Y1)
    print(Y1, file=f)
f.close()

# summarize history for accuracy
plt.plot(history.history['accuracy'], "b-")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'], "y-")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
```

import pandas as pd
import numpy as np
from keras.models import Sequential  # import the model
from keras.layers.core import Dense, Activation  # import frequently used layers


x_train = pd.read_csv('train_data.txt', sep='\t')
y_train = pd.read_csv('train_truth.txt', sep='\t')
x_test = pd.read_csv('test_data.txt', sep='\t')
model=Sequential()  # initialize the model
model.add(Dense(4,input_shape=(3,),activation='relu'))  # 3 inputs，4 hidden neurons
model.add(Dense(4,activation='relu'))  # 4 hidden neurons
model.add(Dense(1))  # one output
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])  # compile, specify objective function and optimization method
hur = model.fit(x_train, y_train, nb_epoch=50, batch_size=64)
y_pred = model.predict(x_test)
print("y_pred:", y_pred)
pd.DataFrame(y_pred.reshape((-1)), columns=['y']).to_csv("test_predicted.txt", index=False)
import pandas as pd 
import pdb
import numpy as np
import random
import tensorflow as tf 
import keras.backend as kb
from keras import losses
from keras import models
from keras import regularizers
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import concatenate
from keras.layers.merge import *
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from keras.optimizers import SGD
from sklearn.metrics import f1_score

ALL_COLUMNS = ['x0', 'x1', 'x2', 'x3', 'x4']

# the function we want to predict. We want to experiment and see what kind of functions
# can be solved by what kind of network. All the functions return x,y as data and label.
def funcCond(seed, start, count):
    x = {k: np.zeros((count, 2)) for k in ALL_COLUMNS}
    y = np.zeros(count)

    for i in range(count):
        x['x0'][i][0] = start+i
        x['x1'][i][0] = random.randint(1, 10)
        if i >= 1:
            y[i] = x['x1'][i-1][0]
            x['x1'][i][1] = x['x1'][i-1][0]
            
            if i % 10 == 0:
                y[i] = x['x1'][i][0]
        else:
            y[i] = seed
            x['x1'][i][1] = seed
    return x,y

def buildDenseModel():
    inputs = []
    for k in ALL_COLUMNS:
        inputs.append(Input(shape=(2,), dtype='float32', name=k))

    x = concatenate(inputs)

    x = Dense(100, activation='relu', name='dense1', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(50, activation='relu', name='dense2', kernel_regularizer=regularizers.l2(0.01))(x)
    prediction = Dense(1, name='dense3')(x)

    model = models.Model(inputs=inputs, outputs=[prediction])
    model.compile(loss='mse', optimizer=adam(lr=0.0001), metrics=['accuracy'])

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    return model, board
    
def testCond(model, board):
    x,y = funcCond(0, 0, 1000)
    evalX, evalY = funcCond(x['x1'][999][0], 1000, 100)
    history = model.fit(x=x, y=y, validation_data=(evalX, evalY), batch_size=100, epochs=400, callbacks=[board])

    return
    loss, metrics = model.evaluate(x=evalX, y=evalY)
    print(loss)
    print(metrics)


def main():
    pdb.set_trace()
    model, board = buildDenseModel()
    testCond(model, board)
    return

if __name__ == '__main__':
    main()
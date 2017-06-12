import pandas as pd 
import pdb
import numpy as np
import random
import math
import tensorflow as tf 
import keras.backend as kb
from keras import losses
from keras import models
from keras import regularizers
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.merge import *
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU

INPUT_COUNT = 20

# generate a sequence of data, using x = sin(a*i). Passing in a and count
def getSineData(a, count):
    x = np.zeros((count,))
    for i in range(0, count):
        x[i] = math.sin(a * i)
    return x

def getXY(data, total):
    x = data[0: total - INPUT_COUNT].reshape((total - INPUT_COUNT, 1))
    for i in range(1, INPUT_COUNT):
        x = np.concatenate((x, data[i: total - INPUT_COUNT + i].reshape((total - INPUT_COUNT, 1))), axis=1)
    
    y = data[INPUT_COUNT : total]
    return x,y

def getSeries(a, count):
    data = getSineData(a, count)
    trainCount = int(count * 0.9)
    testCount = count - trainCount
    x,y = getXY(data, trainCount)
    evalX, evalY = getXY(data[trainCount:], testCount)
    return x, y, evalX, evalY

def buildDenseModel():
    model = models.Sequential()
    model.add(Dense(100, input_shape=(INPUT_COUNT,)))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    return model, board
    
def testSingleParam(model, board):
    total = 10000
    allX, allY, allEX, allEY = getSeries(0.06, total)
    
    for i in range(1, 10):
        x, y, evalX, evalY = getSeries(0.06 + i * 0.006, total)
        allX = np.concatenate((allX, x))
        allY = np.concatenate((allY, y))
        allEX = np.concatenate((allEX, evalX))
        allEY = np.concatenate((allEY, evalY))

    testX, testY, allTX, allTY = getSeries(0.03, total)
    
    history = model.fit(x=allX, y=allY, validation_data=(allTX, allTY), batch_size=100, epochs=100, callbacks=[board])

    return

def main():
    pdb.set_trace()
    model, board = buildDenseModel()
    testSingleParam(model, board)
    return

if __name__ == '__main__':
    main()
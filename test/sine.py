import pandas as pd 
import pdb
import numpy as np
import random
import math
import tensorflow as tf 
import matplotlib.pyplot as plt
import keras.backend as kb
from keras import losses
from keras import models
from keras import regularizers
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.merge import *
from keras.layers.convolutional import Conv1D
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

def getCNNSeries(a, count):
    x, y, evalX, evalY = getSeries(a, count)
    return x.reshape(x.shape[0], x.shape[1], 1), y, evalX.reshape(evalX.shape[0], evalX.shape[1], 1), evalY

def buildDenseModel():
    model = models.Sequential()
    model.add(Dense(1000, input_shape=(INPUT_COUNT,)))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    return model, board

def buildCNNModel():
    model = models.Sequential()
    model.add(Conv1D(100, 3, strides=1, input_shape=(INPUT_COUNT, 1)))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Conv1D(100, 3, strides=1))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    return model, board

def buildLSTMModel():
    model = models.Sequential()
    model.add(LSTM(100, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    return model, board
    
def testFC():
    model, board = buildDenseModel()

    total = 10000
    allX, allY, allEX, allEY = getSeries(0.06, total)
    
    for i in range(1, 10):
        x, y, evalX, evalY = getSeries(0.06 + i * 0.006, total)
        allX = np.concatenate((allX, x))
        allY = np.concatenate((allY, y))
        allEX = np.concatenate((allEX, evalX))
        allEY = np.concatenate((allEY, evalY))

    testX, testY, allTX, allTY = getSeries(0.03, total)
    
    history = model.fit(x=allX, y=allY, validation_data=(allTX, allTY), batch_size=100, epochs=30, callbacks=[board])

    plotGenerated(model, 0.083)
    plotGenerated(model, 0.03)
    return

def testCNN():
    model, board = buildCNNModel()

    total = 10000
    allX, allY, allEX, allEY = getCNNSeries(0.06, total)
    
    for i in range(1, 10):
        x, y, evalX, evalY = getCNNSeries(0.06 + i * 0.006, total)
        allX = np.concatenate((allX, x))
        allY = np.concatenate((allY, y))
        allEX = np.concatenate((allEX, evalX))
        allEY = np.concatenate((allEY, evalY))

    testX, testY, allTX, allTY = getCNNSeries(0.03, total)
    
    history = model.fit(x=allX, y=allY, validation_data=(allTX, allTY), batch_size=100, epochs=30, callbacks=[board])

    plotGeneratedCNN(model, 0.063)
    plotGeneratedCNN(model, 0.03)
    return

def testLSTM():
    model, board = buildLSTMModel()

    total = 1001
    trainSize = int(total * 0.9)
    
    for epoch in range(0, 5):
        for i in range(0, 10):
            data = getSineData(0.06 + i * 0.006, total)
            x = data[0:trainSize].reshape((trainSize, 1, 1))
            y = data[1:trainSize+1]
            evalX = data[trainSize:total-1].reshape((total-1-trainSize, 1, 1))
            evalY = data[trainSize+1:total]
            model.fit(x=x, y=y, validation_data=(evalX, evalY), batch_size=1, epochs=1, shuffle=False)
            model.reset_states()
    
    plotGeneratedLSTM(model, 0.083)
    plotGeneratedLSTM(model, 0.043)
    return

def plotPredicted(model, a):
    model.reset_states()
    x, y, eX, eY = getSeries(a, 200)
    predict = model.predict(x=x)
    plt.plot(y[0:100])
    plt.plot(predict[0:100])
    plt.show()

def plotGenerated(model, a):
    model.reset_states()
    x, y, eX, eY = getSeries(a, 200)
    predict = np.zeros((100,))
    gx = x[0].copy()
    for i in range(0, 100):
        predict[i] = model.predict(x=np.array([gx]))
        gx = np.append(gx[1:], predict[i])
    
    plt.plot(y[0:100])
    plt.plot(predict[0:100])
    plt.show()

def plotGeneratedCNN(model, a):
    model.reset_states()
    x, y, eX, eY = getCNNSeries(a, 200)
    predict = np.zeros((100,))
    gx = x[0].copy()
    for i in range(0, 100):
        predict[i] = model.predict(x=np.array([gx]))
        gx = np.append(gx[1:], np.array([[predict[i]]]), axis=0)
    
    plt.plot(y[0:100])
    plt.plot(predict[0:100])
    plt.show()

def plotGeneratedLSTM(model, a):
    model.reset_states()
    data = getSineData(a, 300)
    x = data[0:100]
    # Build up LSTM state
    model.predict(x=x.reshape((100, 1, 1)), batch_size=1)

    y = data[101:201]
    predicted = np.zeros((100,))
    gx = data[0]
    for i in range(100, 200):
        predicted[i-100] = model.predict(x=gx.reshape(1, 1, 1))
        gx = predicted[i-100]

    plt.plot(y)
    plt.plot(predicted)
    plt.show()

def main():
    pdb.set_trace()
    testLSTM()
    return

if __name__ == '__main__':
    main()
import pandas as pd 
import pdb
import numpy as np
import random
import math
import argparse
import tensorflow as tf 
import matplotlib.pyplot as plt
from keras import losses
from keras import models
from keras import regularizers
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Conv1D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from keras.layers.advanced_activations import LeakyReLU

INPUT_COUNT = 20

# generate a sequence of data, using x = sin(a*i). Passing in a and count
def getSineData(a, count):
    return np.array([math.sin(a * i) for i in range(0, count)])

# Given the time series data and data size, return the x,y data suitable for training
# x - time lagged by INPUT_COUNT, so for each time step we have INPUT_COUNT number of data points
# y - for each data point x, the data at the next time step
def getXY(data, total):
    x = data[0: total - INPUT_COUNT].reshape((total - INPUT_COUNT, 1))
    for i in range(1, INPUT_COUNT):
        x = np.concatenate((x, data[i: total - INPUT_COUNT + i].reshape((total - INPUT_COUNT, 1))), axis=1)
    
    y = data[INPUT_COUNT : total]
    return x,y

# Generate the training and evaluating data sets for data series sin(a*x)
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

def buildFCModel():
    model = models.Sequential()
    model.add(Dense(100, input_shape=(INPUT_COUNT,)))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(100))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    board.set_model(model)
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
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board

def buildLSTMModel():
    model = models.Sequential()
    model.add(LSTM(100, batch_input_shape=(1, 1, 1), return_sequences=False, stateful=True))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=adam(lr=0.0001))

    board = TensorBoard(log_dir='model', histogram_freq=1, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board
    
def testFC():
    model, board = buildFCModel()

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
    plotGenerated(model, 0.163)
    plotGenerated(model, 0.033)
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

    plotGeneratedCNN(model, 0.083)
    plotGeneratedCNN(model, 0.163)
    plotGeneratedCNN(model, 0.033)
    return

def testLSTM():
    model, board = buildLSTMModel()

    total = 1001
    trainSize = int(total * 0.9)
    testData = getSineData(0.05, total)

    # we are not really validating, just use this to make tensorboard work with train_on_batch
    board.validation_data = None
    
    for epoch in range(0, 10):
        for i in range(0, 10):
            data = getSineData(0.06 + i * 0.006, total)
            x = data[0:trainSize].reshape((trainSize, 1, 1))
            y = data[1:trainSize+1]
            evalX = data[trainSize:total-1].reshape((total-1-trainSize, 1, 1))
            evalY = data[trainSize+1:total]
            model.fit(x=x, y=y, validation_data=(evalX, evalY), batch_size=1, epochs=1, shuffle=False)
            model.reset_states()

        board.on_epoch_end(epoch)
    
    plotGeneratedLSTM(model, 0.083)
    plotGeneratedLSTM(model, 0.163)
    plotGeneratedLSTM(model, 0.033)
    return

def render(y, predicted, a, count):
    plt.grid()
    plt.plot(y[0:count])
    plt.plot(predicted[0:count], 'go')
    plt.title("y=sin({}*x)".format(a))
    plt.savefig("plt-{}.png".format(a))
    plt.show()

# Create a series of data points using parameter a (as in sin(a*i)). Predict y using the real
# values of x for each data point. Plot out y against predicted data points.
def plotPredicted(model, a):
    model.reset_states()
    x, y, eX, eY = getSeries(a, 200)
    predicted = model.predict(x=x)
    render(y, predicted, a, 100)

# Start a series of data using parameter a (as in sin(a*i)). Using the data as starting point,
# predict the next data in the time series. Then predict more using the generated data points
# form earlier predictions.
def plotGenerated(model, a):
    x, y, eX, eY = getSeries(a, 200)
    predicted = np.zeros((100,))
    gx = x[0].copy()
    for i in range(0, 100):
        predicted[i] = model.predict(x=np.array([gx]))
        gx = np.append(gx[1:], predicted[i])
    
    render(y, predicted, a, 100)

def plotGeneratedCNN(model, a):
    x, y, eX, eY = getCNNSeries(a, 200)
    predicted = np.zeros((100,))
    gx = x[0].copy()
    for i in range(0, 100):
        predicted[i] = model.predict(x=np.array([gx]))
        gx = np.append(gx[1:], np.array([[predicted[i]]]), axis=0)
    
    render(y, predicted, a, 100)

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

    render(y, predicted, a, 100)

def main():
    parser = argparse.ArgumentParser(description='Trying to model unseen sine function')                                         
    parser.add_argument("-m", "--model", help="fc/cnn/lstm")

    args = parser.parse_args()
    if args.model == 'fc':
        testFC()
    elif args.model == 'cnn':
        testCNN()
    elif args.model == 'lstm':
        testLSTM()
    else:
        print('unknown model')
    return

if __name__ == '__main__':
    main()
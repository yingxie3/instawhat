# Give past orders different weight. Try to learn the proper weight.
# The NN should be: given the current (to be predicted) order time,
# the past order day and time distance, output a weight.
#
# During training, the loss function can be the sme of the generated weight
# and the perfect weight. The perfect weight can be solved mathematically.

import pandas as pd 
import pdb
import argparse
import numpy as np
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
from sklearn.metrics import f1_score
from instadata import *

COLUMNS = ['product_id', 'order_dow', 'order_hour_of_day', 'order_number', 'days_since_prior_order']
# column index
COL_ORDER_ID = 0
COL_USER_ID = 1
COL_EVAL_SET = 2
COL_ORDER_NUMBER = 3
COL_ORDER_DOW = 4
COL_ORDER_HOUR = 5
COL_DAYS_SINCE = 6
COL_PRODUCT_ID = 7
COL_REORDERED = 8

# training dataset columns. 
TCOL_NUMBER_DIFF = 0
TCOL_OLD_DOW = 1
TCOL_DOW = 2
TCOL_OLD_HOUR = 3
TCOL_HOUR = 4
TCOL_OLD_DAYS_SINCE = 5
TCOL_DAYS_SINCE = 6
TCOL_SCORE = 7

LABEL = 'label'
MODEL_DIR = './model'

# training parameters
TRAIN_STEPS = 50
trainCount = 0
trainF1 = 0

# generate the data in sequence, we will randomize during training
def generateTrainingData(userOrders, userPrior):
    priorOrders = userOrders[0:len(userOrders)-1]

    # order to product_id array mapping
    productHash = {k[COL_ORDER_ID] : [] for k in priorOrders}
    for entry in userPrior:
        hashEntry = productHash.get(entry[COL_ORDER_ID])
        if hashEntry == None:
            # we reached the last order entry, which we don't use for training
            break
        hashEntry.append(entry[COL_PRODUCT_ID])

    samples = []
    for i1 in range(0, len(priorOrders)-1):
        for i2 in range(i1+1, len(priorOrders)):
            old = priorOrders[i1]
            new = priorOrders[i2]

            f1 = f1Score(hashEntry[o1[COL_ORDER_ID]], hashEntry[o2[COL_ORDER_ID]])
            entry = [new[COL_ORDER_NUMBER] - old[COL_ORDER_NUMBER], old[COL_ORDER_DOW], new[COL_ORDER_DOW],
                old[COL_ORDER_HOUR], new[COL_ORDER_HOUR], old[COL_DAYS_SINCE], new[COL_DAYS_SINCE], f1]
            samples.append(entry)
    return samples
    
def buildModel():
    productIdInput = Input(shape=(1,), dtype='int32', name='product_id')
    productIdEmbedding = Embedding(input_dim=50000, output_dim=4)(productIdInput)
    productIdFlatten = Flatten()(productIdEmbedding)

    dowInput = Input(shape=(1,), dtype='float32', name='order_dow')
    hourInput = Input(shape=(1,), dtype='float32', name='order_hour_of_day')
    orderNumberInput = Input(shape=(1,), dtype='float32', name='order_number')
    daysSinceInput = Input(shape=(1,), dtype='float32', name='days_since_prior_order')
    x = concatenate([productIdFlatten, dowInput, hourInput, orderNumberInput, daysSinceInput])

    x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=[productIdInput, dowInput, hourInput, orderNumberInput, daysSinceInput], outputs=[prediction])
    model.compile(loss=losses.binary_crossentropy, optimizer=adam(), metrics=['accuracy'])

    board = TensorBoard(log_dir=MODEL_DIR, histogram_freq=2, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board

def getWeightForOrder(order, currentOrder):
    return 0

def f1Score(prediction,actual):
    correct = pd.merge(pd.DataFrame(prediction, columns=['product_id']), 
        pd.DataFrame(actual[:, COL_PRODUCT_ID], columns=['product_id']), how='inner', on='product_id')
    if len(correct) == 0:
        return 0
    precision = len(correct) / len(prediction)
    recall = len(correct) / len(actual)
    return 2 * precision * recall / (precision + recall)

def predictForUser(userOrders, userPrior):
    # for now make the last order's weight 1, others 0
    priorOrders = userOrders[0:len(userOrders)-1]
    orderHash = {k[COL_ORDER_ID] : 0 for k in priorOrders}
    orderHash[priorOrders[-1][COL_ORDER_ID]] = 1

    # assign the weights from orders to the products
    productHash = {k[COL_PRODUCT_ID] : 0 for k in userPrior}
    for entry in userPrior:
        w = orderHash[entry[COL_ORDER_ID]]
        # productHash[entry.product_id] += (w * int(entry.reordered)) # if reordered is 0 throw it out
        productHash[entry[COL_PRODUCT_ID]] += w

    # select products based on weights
    predictedProducts = []
    for productId in productHash:
        if productHash[productId] > 0.5:
            predictedProducts.append(productId)
    return predictedProducts

def trainForUser(userOrders, userPrior, userTrain):
    predictedProducts = predictForUser(userOrders, userPrior)

    # calculate F1 score
    return f1Score(predictedProducts, userTrain[:, COL_PRODUCT_ID])

def getUidRangeEnd(array, startIndex):
    uid = array[startIndex][COL_USER_ID]
    arrayLen = len(array)
    i = startIndex
    while i < arrayLen:
        if array[i][COL_USER_ID] != uid:
            break
        i += 1

    return i

# We use one user's past history to predict the last order
def weightRegression(training):
    orders = loadDataFile('orders')
    prior = loadDataFile('order_products__prior')
    del prior['add_to_cart_order']
    train = loadDataFile('order_products__train')
    del train['add_to_cart_order']

    orderPrior = pd.merge(left=orders, right=prior, how='inner', on='order_id').values
    orderTrain = pd.merge(left=orders, right=train, how='inner', on='order_id').values
    orders = orders.values

    # Everything is sorted by user_id, order_id, order_number, add_to_cart_order after the above merge.
    priorStartIndex = 0
    trainStartIndex = 0
    orderStartIndex = 0
    allpredictions = []
    nextUid = orderPrior[priorStartIndex][COL_USER_ID]
    progress = 0
    while orderStartIndex < len(orders):
        currentUid = nextUid

        priorEndIndex = getUidRangeEnd(orderPrior, priorStartIndex)
        orderEndIndex = getUidRangeEnd(orders, orderStartIndex)
        userPrior = orderPrior[priorStartIndex : priorEndIndex]
        userOrders = orders[orderStartIndex : orderEndIndex]

        priorStartIndex = priorEndIndex
        orderStartIndex = orderEndIndex

        if orderStartIndex < len(orders):
            nextUid = orders[orderStartIndex][COL_USER_ID]
            assert nextUid == orderPrior[priorStartIndex][COL_USER_ID]
            assert nextUid != currentUid

        lastOrder = userOrders[-1]
        lastOrderType = lastOrder[COL_EVAL_SET]

        if lastOrderType == 'train' and training:
            trainEndIndex = getUidRangeEnd(orderTrain, trainStartIndex)
            userTrain = orderTrain[trainStartIndex : trainEndIndex]
            trainStartIndex = trainEndIndex
            if trainStartIndex < len(orderTrain):
                assert orderTrain[trainStartIndex][COL_USER_ID] != currentUid

            f1 = trainForUser(userOrders, userPrior, userTrain)
            global trainCount, trainF1
            trainCount += 1
            trainF1 += f1

            if trainCount % 100 == 0:
                print("****************************************{} f1_score: {:.4f} avg: {:.4f}".format(trainCount, f1, trainF1/trainCount))
        elif lastOrderType == 'test' and (not training):
            predictedProducts = predictForUser(userOrders, userPrior)
            p = [lastOrder[COL_ORDER_ID], ''.join(str(e) + ' ' for e in predictedProducts)]
            allpredictions.append(p)

            progress += 1

            if progress % 100 == 0:
                print("------------------processed {}".format(progress))

    predictionDF = pd.DataFrame(allpredictions, columns=['order_id', 'products'])
    predictionDF.to_csv("submission.csv", index=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing and training')                                         
    parser.add_argument("-t", "--train", help="Train the model.", action="store_true")
                                                                                                                 
    args = parser.parse_args()                                                                                   
    
    weightRegression(args.train)
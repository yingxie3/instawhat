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

CATEGORICAL_COLUMNS = ['product_id']
CONTINUOUS_COLUMNS  = ['order_dow', 'order_hour_of_day', 'order_number', 'days_since_prior_order']
ALL_COLUMNS = ['product_id', 'order_dow', 'order_hour_of_day', 'order_number', 'days_since_prior_order']
LABEL = 'label'
MODEL_DIR = './model'

# training parameters
TRAIN_STEPS = 50
trainCount = 0
trainF1 = 0

def getWeightForOrder(order, currentOrder):
    return 0

def f1Score(prediction,actual):
    correct = pd.merge(prediction, actual, how='inner', on='product_id')
    if len(correct) == 0:
        return 0
    precision = len(correct) / len(prediction)
    recall = len(correct) / len(actual)
    return 2 * precision * recall / (precision + recall)

def predictForUser(userOrders, userPrior):
    # for now make the last order's weight 1, others 0
    priorOrders = userOrders[0:len(userOrders)-1]
    orderHash = {k : 0 for k in priorOrders.order_id}
    orderHash[priorOrders.iloc[-1].order_id] = 1

    # assign the weights from orders to the products
    productHash = {k : 0 for k in userPrior.product_id}
    for row in userPrior.iterrows():
        entry = row[1]
        w = orderHash[entry.order_id]
        # productHash[entry.product_id] += (w * int(entry.reordered)) # if reordered is 0 throw it out
        productHash[entry.product_id] += w

    # select products based on weights
    predictedProducts = []
    for productId in productHash:
        if productHash[productId] > 0.5:
            predictedProducts.append(productId)
    return predictedProducts

def trainForUser(userOrders, userPrior, userTrain):
    predictedProducts = predictForUser(userOrders, userPrior)

    # calculate F1 score
    f1 = f1Score(pd.DataFrame(predictedProducts, columns=['product_id']), pd.DataFrame(userTrain.product_id))
    return f1

# We use one user's past history to predict the last order
def weightRegression(training):
    orders = loadDataFile('orders')
    prior = loadDataFile('order_products__prior')
    del prior['add_to_cart_order']
    train = loadDataFile('order_products__train')
    del train['add_to_cart_order']

    orderPrior = pd.merge(left=orders, right=prior, how='inner', on='order_id')
    orderTrain = pd.merge(left=orders, right=train, how='inner', on='order_id')

    # Everything is sorted by user_id, order_id, order_number, add_to_cart_order after the above merge.
    priorStartIndex = 0
    trainStartIndex = 0
    orderStartIndex = 0
    allpredictions = []
    nextUid = orderPrior.iloc[priorStartIndex].user_id
    progress = 0
    while orderStartIndex < len(orders):
        currentUid = nextUid
        range = 100
        uid = currentUid
        # first find the end point for the user. Just needs to be a rough guess.
        while uid == currentUid:
            range *= 2
            priorRange = orderPrior.iloc[priorStartIndex : priorStartIndex+range]
            uid = priorRange.iloc[-1].user_id

        userPrior = priorRange.loc[lambda x: x.user_id == currentUid]
        userOrders = orders.iloc[orderStartIndex:].loc[lambda x: x.user_id == currentUid]

        priorStartIndex += len(userPrior)
        orderStartIndex += len(userOrders)

        nextUid = orderPrior.iloc[priorStartIndex].user_id
        assert nextUid != currentUid
        if orderStartIndex < len(orders):
            assert orders.iloc[orderStartIndex].user_id == nextUid

        lastOrder = userOrders.iloc[-1]
        lastOrderType = lastOrder['eval_set']

        if lastOrderType == 'train' and training:
            # no range, train set is small
            userTrain = orderTrain.iloc[trainStartIndex:].loc[lambda x: x.user_id == currentUid] 
            trainStartIndex += len(userTrain)
            if trainStartIndex < len(orderTrain):
                assert orderTrain.iloc[trainStartIndex].user_id != currentUid

            f1 = trainForUser(userOrders, userPrior, userTrain)
            global trainCount, trainF1
            trainCount += 1
            trainF1 += f1

            if trainCount % 100 == 0:
                print("****************************************{} f1_score: {:.4f} avg: {:.4f}".format(trainCount, f1, trainF1/trainCount))
        elif lastOrderType == 'test' and (not training):
            predictedProducts = predictForUser(userOrders, userPrior)
            p = [lastOrder.order_id, ''.join(str(e) + ' ' for e in predictedProducts)]
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
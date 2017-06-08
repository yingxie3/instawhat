# Give past orders different weight. Try to learn the proper weight.
# The NN should be: given the current (to be predicted) order time,
# the past order day and time distance, output a weight.
#
# During training, the loss function can be the sme of the generated weight
# and the perfect weight. The perfect weight can be solved mathematically.

import pandas as pd 
import pdb
import json
import argparse
import numpy as np
import tensorflow as tf 
import keras.backend as kb
from keras import losses
from keras import models
from keras import regularizers
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import concatenate
from keras.layers.merge import *
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from sklearn.metrics import f1_score
from shutil import copyfile
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
COL_CART_ORDER = 8
COL_REORDERED = 9

# training dataset columns. One data point is 3x of the following, to form a 3x3 block
TCOL_NUMBER_DIFF = 0
TCOL_DOW_DIFF = 1
TCOL_HOUR_DIFF = 2
TCOL_DAYS_DIFF = 3
# the score we are predicting, must be the last
TCOL_SCORE = 4

LABEL = 'label'
MODEL_DIR = './model'

# training parameters
BATCH_SIZE = 1000
TRAIN_STEPS = 500
ORDERS_TO_CONSIDER = 10
trainCount = 0
trainF1 = 0

bestWeight = 0.4
weightTotal = 0
weightCount = 0

# random weights used to reinit the NN
randomWeights = None

class ReplayMemory(object):
    def __init__(self, maxMemory=100000):
        self.maxMemory = maxMemory
        self.current = 0
        self.count = 0
        self.x = np.zeros(shape=(maxMemory, TCOL_SCORE))
        self.y = np.zeros(shape=(maxMemory,))

    def remember(self, x, y):
        assert len(x) == len(y)
        for i in range(0, len(x)):
            self.x[self.current] = x[i]
            self.y[self.current] = y[i]

            self.current += 1
            self.count += 1
            if self.current >= self.maxMemory:
                self.current = 0

    def getBatch(self, batchSize):
        validCount = min(self.count, self.maxMemory)
        x = np.empty(shape=(batchSize, TCOL_SCORE))
        y = np.empty(shape=(batchSize,))

        for i, idx in enumerate(np.random.randint(0, validCount, size=batchSize)):
            x[i] = self.x[idx]
            y[i] = self.y[idx]

        return x,y

# Generate the input to the NN from the old and new data order data point
def getInput(old, new, daysDiff):
    return np.array([new[COL_ORDER_NUMBER] - old[COL_ORDER_NUMBER], new[COL_ORDER_DOW]-old[COL_ORDER_DOW],
        new[COL_ORDER_HOUR]-old[COL_ORDER_HOUR], daysDiff])

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

    x = []
    y = []
    for i1 in range(0, len(priorOrders)-1):
        daysDiff = 0
        rangeEnd = min(len(priorOrders), i1 + ORDERS_TO_CONSIDER + 1)
        for i2 in range(i1+1, rangeEnd):
            old = priorOrders[i1]
            new = priorOrders[i2]

            daysDiff += new[COL_DAYS_SINCE]
            f1, precision, recall = f1Score(productHash[old[COL_ORDER_ID]], productHash[new[COL_ORDER_ID]])
            x.append(getInput(old, new, daysDiff))
            y.append(precision)
    return x,y

def createModel():
    model = models.Sequential()
    model.add(Dense(100, input_shape=(TCOL_SCORE,), name='d1'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(100, name='d2'))
    model.add(LeakyReLU(alpha=0.03))
    model.add(Dense(1, activation='sigmoid', name='out'))

    model.compile(loss='mse', optimizer=adam())
    global randomWeights
    randomWeights = model.get_weights().copy()

    board = TensorBoard(log_dir=MODEL_DIR, histogram_freq=2, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board

def loadModel(model):
    try:                                      
        model.load_weights("model.h5")
        print("model loaded")        
    except OSError:                           
        print("can't find model file to load")

def saveModel(model):
    print("Saving model")                         
    try:                                          
        copyfile('model.h5', 'model.h5.saved')    
        copyfile('model.json', 'model.json.saved')
    except FileNotFoundError:                     
        print("didn't find model files to save")  
                                                    
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:      
        json.dump(model.to_json(), outfile)       

def getWeightForOrder(order, currentOrder):
    return 0

def f1Score(prediction,actual):
    correct = pd.merge(pd.DataFrame(prediction, columns=['product_id']), 
        pd.DataFrame(actual, columns=['product_id']), how='inner', on='product_id')
    if len(correct) == 0:
        return 0, 0, 0
    precision = len(correct) / len(prediction)
    recall = len(correct) / len(actual)
    return 2 * precision * recall / (precision + recall), precision, recall

# given productHash that maps id to weight, and the number of desired products, return
# the list of product ids with the highest weights
def getBestProducts(productHash, desiredCount):
    if desiredCount >= len(productHash):
        return list(productHash)

    productList = [[k,v] for k,v in productHash.items()]
    sortedList = sorted(productList, key=lambda x: -x[1])
    return [x[0] for x in sortedList[0:desiredCount]]

def predictForUser(model, userOrders, userPrior):
    orderHash = {}
    # we only consider the most recent ORDERS_TO_CONSIDER orders
    start = max(0, len(userOrders) - 1 - ORDERS_TO_CONSIDER)
    daysDiff = userOrders[-1][COL_DAYS_SINCE]
    for i in reversed(range(start, len(userOrders)-1)):
        old = userOrders[i]
        x = getInput(old, userOrders[-1], daysDiff)
        # we use the predicted precision score as weight
        orderHash[old[COL_ORDER_ID]] = model.predict(np.array([x]))[0][0]

        if i != 0:
            daysDiff += old[COL_DAYS_SINCE]

    '''
    for i in range(start, len(userOrders)-1):
        print("order {}\tweight {:.4f}".format(userOrders[i][COL_ORDER_ID], orderHash[userOrders[i][COL_ORDER_ID]]))
    # force everything from the very last order
    #orderHash[userOrders[-2][COL_ORDER_ID]] = 1
    # from each order, select the percentage of products that matches the model's prediction
    i = 0
    predictedProducts = set()
    while i < len(userPrior):
        currentProduct = userPrior[i]
        assert currentProduct[COL_CART_ORDER] == 1
        nextIndex = i + next((idx for idx, x in enumerate(userPrior[i:]) if x[COL_ORDER_ID] != currentProduct[COL_ORDER_ID]),
            len(userPrior[i:]))

        w = orderHash.get(currentProduct[COL_ORDER_ID])
        if w != None:
            productsInOrder = userPrior[nextIndex-1][COL_CART_ORDER]
            count = int(productsInOrder * w)
            for j in range(i, i+count):
                predictedProducts.add(userPrior[j][COL_PRODUCT_ID])

        i = nextIndex
    
    return list(predictedProducts), {}
    '''

    # assign the weights from orders to the products
    totalOrderProducts = 0
    productHash = {}
    for entry in userPrior:
        w = orderHash.get(entry[COL_ORDER_ID])
        if w == None:
            continue

        totalOrderProducts += 1

        # Give preference to the reordered ones.
        if entry[COL_REORDERED] == 1:
            w *= 1.5
        
        if productHash.get(entry[COL_PRODUCT_ID]) == None:
            productHash[entry[COL_PRODUCT_ID]] = w
        else:
            productHash[entry[COL_PRODUCT_ID]] += w

    return getBestProducts(productHash, int(totalOrderProducts / len(orderHash) + 1)), productHash

def trainForUser(model, history, userOrders, userPrior):
    loss = 1
    global randomWeights
    model.set_weights(randomWeights)

    samplesX, samplesY = generateTrainingData(userOrders, userPrior)
    '''
    history.remember(samplesX, samplesY)
    x,y = history.getBatch(BATCH_SIZE)
    loss = model.train_on_batch(x, y)
    '''
    count = 0
    while loss > 0.01 and count < TRAIN_STEPS:
        loss = model.train_on_batch(np.array(samplesX), np.array(samplesY))
        count += 1
    return loss

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
def weightRegression(mode, startingUid):
    model, board = createModel()
    history = ReplayMemory()

    validationData = [np.ones((BATCH_SIZE, TCOL_SCORE))]
    loadModel(model)

    orders = loadDataFile('orders')
    prior = loadDataFile('order_products__prior')
    train = loadDataFile('order_products__train')

    orderPrior = pd.merge(left=orders, right=prior, how='inner', on='order_id').values
    orderTrain = pd.merge(left=orders, right=train, how='inner', on='order_id').values
    orders = orders.values

    # Everything is sorted by user_id, order_id, order_number, add_to_cart_order after the above merge.
    priorStartIndex = 0
    trainStartIndex = 0
    orderStartIndex = 0
    allpredictions = []

    if startingUid != None:
        priorStartIndex = next((idx for idx, x in enumerate(orderPrior) if x[COL_USER_ID] == startingUid), None)
        trainStartIndex = next((idx for idx, x in enumerate(orderTrain) if x[COL_USER_ID] == startingUid), None)
        orderStartIndex = next((idx for idx, x in enumerate(orders) if x[COL_USER_ID] == startingUid), None)
    
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

        if lastOrderType == 'train' and (mode == 'train' or mode == 'evaluate'):
            trainEndIndex = getUidRangeEnd(orderTrain, trainStartIndex)
            userTrain = orderTrain[trainStartIndex : trainEndIndex]
            trainStartIndex = trainEndIndex
            if trainStartIndex < len(orderTrain):
                assert orderTrain[trainStartIndex][COL_USER_ID] != currentUid

            loss = 0
            if mode == 'train':
                loss = trainForUser(model, history, userOrders, userPrior)

            predictedProducts, productHash = predictForUser(model, userOrders, userPrior)
            userTrainReorderedOnly = []
            for x in userTrain:
                if x[COL_REORDERED] == 1:
                    userTrainReorderedOnly.append(x[COL_PRODUCT_ID])
            f1, precision, recall = f1Score(predictedProducts, userTrainReorderedOnly)
            print("============== user {}\tloss {:.8f} f1 {:.8f}".format(userOrders[0][COL_USER_ID], loss, f1))
            global trainCount, trainF1
            trainCount += 1
            trainF1 += f1

            if trainCount % 100 == 0:
                # on_epoch_end requires validataData to be present. It doesn't really need
                # the data to get the histogram, so we just give is a pre-fabricated one.
                board.validation_data = validationData
                board.on_epoch_end(trainCount)

                #saveModel(model)
                print("****************************************{} f1_score: {:.4f} avg: {:.4f}".format(trainCount, f1, trainF1/trainCount))
        
        elif lastOrderType == 'test' and (mode == 'test'):
            trainForUser(model, history, userOrders, userPrior)
            predictedProducts, productHash = predictForUser(model, userOrders, userPrior)
            p = [lastOrder[COL_ORDER_ID], ''.join(str(e) + ' ' for e in predictedProducts)]
            allpredictions.append(p)

            progress += 1
            print("{},{}".format(p[0], p[1]))
            '''
            if progress % 100 == 0:
                print("------------------processed {}".format(progress))
            '''
    predictionDF = pd.DataFrame(allpredictions, columns=['order_id', 'products'])
    predictionDF.to_csv("submission.csv", index=False)
    return

def main():
    parser = argparse.ArgumentParser(description='Parsing and training')                                         
    parser.add_argument("-m", "--mode", help="train/evaluate/test")
    parser.add_argument("-u", "--uid", help="Starting user id.")
    parser.add_argument("-w", "--weight", help="The cutoff weight to use.")
                                                                                                                 
    args = parser.parse_args()                                                                                   
    
    if args.weight != None:
        global bestWeight
        bestWeight = float(args.weight)
    weightRegression(args.mode, int(args.uid))

if __name__ == '__main__':
    #pdb.set_trace()
    main()
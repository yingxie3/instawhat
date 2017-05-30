# We experiment with regression methods here

import pandas as pd 
import pdb
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

def inputFunc(df, training=True):
    columns = {k: df[k] for k in ALL_COLUMNS}

    if training:
        # Converts the label column into a constant Tensor.
        label = df[LABEL]
        # Returns the feature columns and the label.
        return columns, label
    else:
        return columns

def getUniqueOrdersProducts(userId, orderPrior, reorderedOnly=False):
    # construct data suitable for training. The X would be all the features, using all
    # the products the user has ever ordered. The Y would be 0/1 indicating whether this
    # time the user has ordered the product.

    userProducts = orderPrior.loc[lambda x: x.user_id == userId]
    poHash = {}

    if reorderedOnly:
        reorderedProducts = []
        for index,entry in userProducts.iterrows():
            if entry['reordered'] == 1:
                reorderedProducts.append(entry['product_id'])
                poHash["{}:{}".format(entry['order_id'], entry['product_id'])] = 1

        if len(reorderedProducts) == 0:
            # a hopeless situation. Just use a few products from the last order
            allUserProducts = userProducts.tail().product_id.unique()
        else:
            allUserProducts = pd.DataFrame(reorderedProducts)[0].unique()
    else:
        for index,entry in userProducts.iterrows():
            poHash["{}:{}".format(entry['order_id'], entry['product_id'])] = 1
        allUserProducts = userProducts.product_id.unique()
        
    allUserOrders = userProducts.order_id.unique()

    return poHash, allUserOrders, allUserProducts

def addLabels(orders, poHash, allUserOrders, allUserProducts):
    # not the most efficient way, but it's ok since there are not many orders per user.
    allPotentialPO = []
    for p in allUserProducts:
        for o in allUserOrders:
            allPotentialPO.append([o, p])

    podf = pd.DataFrame(allPotentialPO)
    podf.columns = ['order_id', 'product_id']

    expandedPO = pd.merge(left=orders, right=podf, how='right', on='order_id')
    expandedPO[LABEL] = 0

    # if there are known POs, set the label on them to 1
    if poHash != None:
        for index,po in expandedPO.iterrows():
            if poHash.get("{}:{}".format(po['order_id'], po['product_id'])) == 1:
                expandedPO.set_value(index, LABEL, 1)
    return expandedPO

def getEvalSet(trainPO, priorProducts):
    first = trainPO.iloc[0]
    evalArray = []
    for pid in priorProducts:
        entry = first.copy()
        entry.product_id = pid
        evalArray.append(entry)
    return pd.DataFrame(evalArray)

def getF1Score(evalX, prediction, evalProducts, trainPO, trainProducts):
    evalProductHash = {}
    for entry in evalProducts:
        evalProductHash[entry] = 1

    trainProductHash = {}
    for entry in trainProducts:
        trainProductHash[entry] = 1
    
    # what's missing is predicted as 0
    missingPrediction = []
    for entry in trainProducts:
        if evalProductHash.get(entry) != 1:
            missingPrediction.append(0)
    prediction = np.append(prediction, missingPrediction)

    # go through the products in the same order and fill in zeros and ones
    truth = np.ones(len(prediction), dtype=int)
    for index,entry in enumerate(evalProducts):
        truth[index] = int(trainProductHash.get(entry) == 1)

    return f1_score(truth, prediction)

# training and evaluating a single user's history
def trainForUser(model, board, userId, userOrders, orderPrior, orderTrain):
    trainHash, trainOrders, trainProducts = getUniqueOrdersProducts(userId, orderTrain, reorderedOnly=False)
    if len(trainProducts) == 0:
        return

    priorHash, priorOrders, priorProducts = getUniqueOrdersProducts(userId, orderPrior, reorderedOnly=True)

    '''
    productsDF = pd.concat([pd.DataFrame(priorProducts), pd.DataFrame(trainProducts)], ignore_index=True)
    allUserProducts = productsDF[0].unique()
    '''

    priorPO = addLabels(userOrders, priorHash, priorOrders, priorProducts)
    trainPO = addLabels(userOrders, trainHash, trainOrders, trainProducts)

    x,y = inputFunc(priorPO)
    model.reset_states()
    history = model.fit(x=x, y=y, batch_size=32, epochs=TRAIN_STEPS, callbacks=[board])

    # Generate a eval set using the trainPO's order number, but the priorPO's product list (we can only predict
    # products we have seen). Afterwards, we will consolidate the prediction result back to the trainPO data
    # set for a f1 score.
    evalPO = getEvalSet(trainPO, priorProducts)

    evalX, evalY = inputFunc(evalPO)
    prediction = model.predict(x=evalX)
    prediction = prediction.reshape((len(prediction)))
    prediction[prediction > 0.5] = 1
    prediction[prediction < 0.5] = 0
    print(prediction)

    f1 = getF1Score(evalX, prediction, priorProducts, trainPO, trainProducts)

    global trainCount, trainF1
    trainCount += 1
    trainF1 += f1
    print("******************************************* f1_score: {} avg: {}".format(f1, trainF1/trainCount))

# predict for a single user's history
def predictForUser(model, userId, userOrders, orderPrior): 
    return

    poHash, priorOrders, allUserProducts = getUniqueOrdersProducts(userId, orderPrior)
    priorPO = addLabels(userOrders, poHash, priorOrders, allUserProducts)

    x,y = inputFunc(priorPO)
    model.reset_states()
    history = model.fit(x=x, y=y, batch_size=32, epochs=TRAIN_STEPS)

    # now predict
    testOrders = [ userOrders['order_id'].iloc[-1] ]
    testPO = addLabels(userOrders, None, testOrders, allUserProducts)

    test_x =  inputFunc(testPO, training=False)
    prediction = model.predict(x=test_x)
    prediction = prediction.reshape((len(prediction)))
    prediction[prediction > 0.5] = 1
    prediction[prediction < 0.5] = 0
    print(prediction)

    print("done")

# We use one user's past history to predict the last order
def singleUserRegression():
    model, board = buildModel()
    orders = loadDataFile('orders')

    prior = loadDataFile('order_products__prior')
    del prior['add_to_cart_order']
    train = loadDataFile('order_products__train')
    del train['add_to_cart_order']


    orderPrior = pd.merge(left=orders, right=prior, how='inner', on='order_id')
    orderTrain = pd.merge(left=orders, right=train, how='inner', on='order_id')

    # Every time pick a single user, merge the data just for this one user, and run regressor.
    users = orders.user_id.unique()
    for uid in users:
        userOrders = orders.loc[lambda x: x.user_id == uid]
        lastOrderType = userOrders['eval_set'].iloc[-1]
        if lastOrderType == 'train':
            trainForUser(model, board, uid, userOrders, orderPrior, orderTrain)
        elif lastOrderType == 'test':
            predictForUser(model, uid, userOrders, orderPrior)

    return

if __name__ == '__main__':
    singleUserRegression()
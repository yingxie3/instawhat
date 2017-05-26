# We experiment with regression methods here

import pandas as pd 
import pdb
import numpy as np
import tensorflow as tf 
from keras import losses
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from sklearn.metrics import f1_score
from instadata import *

CATEGORICAL_COLUMNS = ['product_id']
CONTINUOUS_COLUMNS  = ['order_dow', 'order_hour_of_day', 'order_number', 'days_since_prior_order']
LABEL = 'label'
MODEL_DIR = './model'

# training parameters
TRAIN_STEPS = 1000

def buildModel():
    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2))

    model.compile(loss=losses.binary_crossentropy, optimizer=adam())

    board = TensorBoard(log_dir=MODEL_DIR, histogram_freq=2, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board

def inputFunc(df, training=True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuousColumns = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categoricalColumns = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}

    # Merges the two dictionaries into one.
    columns = dict(continuousColumns)
    columns.update(categoricalColumns)

    if training:
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL].values)
        # Returns the feature columns and the label.
        return columns, label
    else:
        return columns

def getUniqueOrdersProducts(userId, userOrders, orderPrior):
    # construct data suitable for training. The X would be all the features, using all
    # the products the user has ever ordered. The Y would be 0/1 indicating whether this
    # time the user has ordered the product.
    userProducts = orderPrior.loc(lambda x: x.user_id == userId)
    poHash = {}
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


# training and evaluating a single user's history
def trainForUser(model, userId, userOrders, orderPrior, orderTrain):
    trainHash, trainOrders, trainProducts = getUniqueOrdersProducts(userId, userOrders, orderTrain)
    if len(trainProducts) == 0:
        return

    priorHash, priorOrders, priorProducts = getUniqueOrdersProducts(userId, userOrders, orderPrior)
    poHash = priorHash
    poHash.update(trainHash)

    productsDF = pd.concat([pd.DataFrame(priorProducts), pd.DataFrame(trainProducts)], ignore_index=True)
    allUserProducts = productsDF[0].unique()

    priorPO = addLabels(userOrders, poHash, priorOrders, allUserProducts)
    trainPO = addLabels(userOrders, poHash, trainOrders, allUserProducts)

    model.fit(input_fn=lambda: inputFunc(priorPO), steps=TRAIN_STEPS)
    results = model.evaluate(input_fn=lambda: inputFunc(trainPO), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    prediction = list(model.predict(input_fn=lambda: inputFunc(trainPO, training=False)))
    print(prediction)

    truth = trainPO[LABEL].tolist()
    print("f1_score: {}".format(f1_score(truth, prediction)))

    print("done")

# predict for a single user's history
def predictForUser(model, orderPrior): 
    return

    poHash, priorOrders, allUserProducts = getUniqueOrdersProducts(orders, prior)
    priorPO = addLabels(orders, poHash, priorOrders, allUserProducts)
    model.fit(input_fn=lambda: inputFunc(priorPO), steps=TRAIN_STEPS)

    # now predict
    testOrders = [ orders['order_id'].iloc[-1] ]
    testPO = addLabels(orders, None, testOrders, allUserProducts)

    prediction = list(model.predict(input_fn=lambda: inputFunc(testPO, training=False)))
    print(prediction)

    # 

    print("done")

# We use one user's past history to predict the last order
def singleUserRegression():
    model = buildModel()
    orders = loadDataFile('orders')

    prior = loadDataFile('order_products__prior')
    del prior['add_to_cart_order']
    del prior['reordered']
    train = loadDataFile('order_products__train')
    del train['add_to_cart_order']
    del train['reordered']

    pdb.set_trace()

    orderPrior = pd.merge(left=orders, right=prior, how='inner', on='order_id')
    orderTrain = pd.merge(left=orders, right=train, how='inner', on='order_id')

    # Every time pick a single user, merge the data just for this one user, and run regressor.
    users = orders.user_id.unique()
    for uid in users:
        userOrders = orders.loc[lambda x: x.user_id == uid]
        lastOrderType = userOrders['eval_set'].iloc[-1]
        if lastOrderType == 'train':
            trainForUser(model, uid, userOrders, orderPrior, orderTrain)
        elif lastOrderType == 'test':
            predictForUser(model, orderPrior)

    return

if __name__ == '__main__':
    singleUserRegression()
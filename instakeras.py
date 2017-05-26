# We experiment with regression methods here

import pandas as pd 
import pdb
import numpy as np
import tensorflow as tf 
import keras.backend as kb
from keras import losses
from keras import models
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
TRAIN_STEPS = 100

def buildModel():
    productIdInput = Input(shape=(1,), dtype='int32', name='product_id')
    productIdEmbedding = Embedding(input_dim=50000, output_dim=4)(productIdInput)
    productIdFlatten = Flatten()(productIdEmbedding)

    dowInput = Input(shape=(1,), dtype='float32', name='order_dow')
    hourInput = Input(shape=(1,), dtype='float32', name='order_hour_of_day')
    orderNumberInput = Input(shape=(1,), dtype='float32', name='order_number')
    daysSinceInput = Input(shape=(1,), dtype='float32', name='days_since_prior_order')
    x = concatenate([productIdFlatten, dowInput, hourInput, orderNumberInput, daysSinceInput])
    '''
    x = concatenate([productIdFlatten, kb.cast(dowInput, dtype='float32'), 
        kb.cast(hourInput, dtype='float32'), kb.cast(orderNumberInput, dtype='float32'), daysSinceInput])
    '''

    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=[productIdInput, dowInput, hourInput, orderNumberInput, daysSinceInput], outputs=[prediction])
    model.compile(loss=losses.binary_crossentropy, optimizer=adam(), metrics=['accuracy'])

    board = TensorBoard(log_dir=MODEL_DIR, histogram_freq=2, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board

def inputFunc(df, training=True):
    '''
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
    '''
    columns = {k: df[k] for k in ALL_COLUMNS}

    if training:
        # Converts the label column into a constant Tensor.
        label = df[LABEL]
        # Returns the feature columns and the label.
        return columns, label
    else:
        return columns

def getUniqueOrdersProducts(userId, orderPrior):
    # construct data suitable for training. The X would be all the features, using all
    # the products the user has ever ordered. The Y would be 0/1 indicating whether this
    # time the user has ordered the product.

    userProducts = orderPrior.loc[lambda x: x.user_id == userId]
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
    trainHash, trainOrders, trainProducts = getUniqueOrdersProducts(userId, orderTrain)
    if len(trainProducts) == 0:
        return

    priorHash, priorOrders, priorProducts = getUniqueOrdersProducts(userId, orderPrior)
    poHash = priorHash
    poHash.update(trainHash)

    productsDF = pd.concat([pd.DataFrame(priorProducts), pd.DataFrame(trainProducts)], ignore_index=True)
    allUserProducts = productsDF[0].unique()

    priorPO = addLabels(userOrders, poHash, priorOrders, allUserProducts)
    trainPO = addLabels(userOrders, poHash, trainOrders, allUserProducts)

    x,y = inputFunc(priorPO)
    history = model.fit(x=x, y=y, batch_size=32, epochs=TRAIN_STEPS)

    train_x, train_y = inputFunc(trainPO)
    loss, metrics = model.evaluate(x=train_x, y=train_y)

    prediction = model.predict(x=train_x)
    prediction.reshape((len(prediction)))
    prediction[prediction > 0.5] = 1
    prediction[prediction < 0.5] = 0
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
    pdb.set_trace()
    model, board = buildModel()
    orders = loadDataFile('orders')

    prior = loadDataFile('order_products__prior')
    del prior['add_to_cart_order']
    del prior['reordered']
    train = loadDataFile('order_products__train')
    del train['add_to_cart_order']
    del train['reordered']


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
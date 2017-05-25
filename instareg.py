# We experiment with regression methods here

import pandas as pd 
import pdb
import numpy as np
import tensorflow as tf 
from instadata import *

CATEGORICAL_COLUMNS = ['product_id']
CONTINUOUS_COLUMNS  = ['order_dow', 'order_hour_of_day', 'order_number', 'days_since_prior_order']
LABEL = 'label'
MODEL_DIR = './model'

# training parameters
TRAIN_STEPS = 1000

def buildModel():
    # sparse columns
    productId = tf.contrib.layers.sparse_column_with_hash_bucket("product_id", hash_bucket_size=1000, dtype=tf.int64)

    # continuous columns
    dow = tf.contrib.layers.real_valued_column("order_dow")
    hour = tf.contrib.layers.real_valued_column("order_hour_of_day")
    orderNumber = tf.contrib.layers.real_valued_column("order_number")
    daysSince = tf.contrib.layers.real_valued_column("days_since_prior_order")

    wideColumns = [productId, dow, hour, orderNumber, daysSince]
    # m = tf.contrib.learn.LinearClassifier(model_dir=MODEL_DIR, feature_columns=wideColumns)

    deepColumns = [tf.contrib.layers.embedding_column(productId, dimension=8), dow, hour, orderNumber, daysSince]

    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=MODEL_DIR,
        linear_feature_columns=wideColumns,
        dnn_feature_columns=deepColumns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
    return m

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

def getUniqueOrdersProducts(orders, prior):
    # construct data suitable for training. The X would be all the features, using all
    # the products the user has ever ordered. The Y would be 0/1 indicating whether this
    # time the user has ordered the product.
    userProducts = pd.merge(left=orders, right=prior, how='inner', on='order_id')
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
def trainForUser(model, orders, prior, train):

    # orderProducts = pd.concat([prior, train], ignore_index=True)
    trainHash, trainOrders, trainProducts = getUniqueOrdersProducts(orders, train)
    if len(trainProducts) == 0:
        return

    priorHash, priorOrders, priorProducts = getUniqueOrdersProducts(orders, prior)
    poHash = priorHash
    poHash.update(trainHash)

    productsDF = pd.concat([pd.DataFrame(priorProducts), pd.DataFrame(trainProducts)], ignore_index=True)
    allUserProducts = productsDF[0].unique()

    priorPO = addLabels(orders, poHash, priorOrders, allUserProducts)
    trainPO = addLabels(orders, poHash, trainOrders, allUserProducts)

    model.fit(input_fn=lambda: inputFunc(priorPO), steps=TRAIN_STEPS)
    results = model.evaluate(input_fn=lambda: inputFunc(trainPO), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))

    prediction = list(model.predict(input_fn=lambda: inputFunc(trainPO, training=False)))
    print(prediction)

    print("done")

# predict for a single user's history
def predictForUser(model, orders, prior):
    poHash, priorOrders, allUserProducts = getUniqueOrdersProducts(orders, prior)
    priorPO = addLabels(orders, poHash, priorOrders, allUserProducts)
    model.fit(input_fn=lambda: inputFunc(priorPO), steps=TRAIN_STEPS)

    # now predict
    testOrders = [ orders['order_id'].iloc[-1] ]
    testPO = addLabels(orders, None, testOrders, allUserProducts)

    prediction = list(model.predict(input_fn=lambda: inputFunc(testPO, training=False)))
    print(prediction)

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
    # Every time pick a single user, merge the data just for this one user, and run regressor.
    users = orders.user_id.unique()
    for u in users:
        userOrders = orders.loc[lambda oi: oi.user_id == u]
        lastOrderType = userOrders['eval_set'].iloc[-1]
        if lastOrderType == 'train':
            trainForUser(model, userOrders, prior, train)
        elif lastOrderType == 'test':
            predictForUser(model, userOrders, prior)

    return

if __name__ == '__main__':
    singleUserRegression()
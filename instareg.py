# We experiment with regression methods here

import pandas as pd 
import pdb
import numpy as np
import tensorflow as tf 
from instadata import *

LABEL = 'label'

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
    for index,po in expandedPO.iterrows():
        if poHash.get("{}:{}".format(po['order_id'], po['product_id'])) == 1:
            expandedPO.set_value(index, LABEL, 1)
    return expandedPO


# training and evaluating a single user's history
def trainForUser(orders, prior, train):

    # orderProducts = pd.concat([prior, train], ignore_index=True)
    pdb.set_trace()

    priorHash, priorOrders, priorProducts = getUniqueOrdersProducts(orders, prior)
    trainHash, trainOrders, trainProducts = getUniqueOrdersProducts(orders, train)
    poHash = priorHash
    poHash.update(trainHash)

    productsDF = pd.concat([pd.DataFrame(priorProducts), pd.DataFrame(trainProducts)], ignore_index=True)
    allUserProducts = productsDF[0].unique()

    priorPO = addLabels(orders, poHash, priorOrders, allUserProducts)
    trainPO = addLabels(orders, poHash, trainOrders, allUserProducts)

    '''
    expandedPO[LABEL] = 0
    expandedPO[LABEL] = expandedPO.apply(lambda x: poHash.get("{}:{}".format(x.order_id, x.product_id)) == 1 )
    '''
    print("done")

# We use one user's past history to predict the last order
def singleUserRegression():

    orders = loadDataFile('orders')

    prior = loadDataFile('order_products__prior')
    del prior['add_to_cart_order']
    del prior['reordered']
    train = loadDataFile('order_products__train')
    del train['add_to_cart_order']
    del train['reordered']

    # Every time pick a single user, merge the data just for this one user, and run regressor.
    users = orders.user_id.unique()
    for u in users:
        userOrders = orders.loc[lambda oi: oi.user_id == u]
        trainForUser(userOrders, prior, train)

    return

if __name__ == '__main__':
    singleUserRegression()
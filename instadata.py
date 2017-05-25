import pandas as pd 
import pdb
import numpy as np 

DATA_DIR = './data'

# Helper functions to make other code cleaner looking
def loadDataFile(dataname):
    d = pd.read_csv("{}/{}.csv".format(DATA_DIR, dataname), engine='c')
    print("Loaded {} {}".format(d.shape[0], dataname))
    return d

# unit tests
if __name__ == '__main__':
    a = loadDataFile('aisles')

    pdb.set_trace()

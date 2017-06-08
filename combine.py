import pandas as pd
import pdb
import numpy as np 
import argparse

def main():
    parser = argparse.ArgumentParser(description='Combine results from two submissions.csv')
    parser.add_argument(dest='files', metavar='files', type=str, nargs='+', help='Files to combine')
    parser.add_argument('-o', '--output', help='output file', required=True)
    parser.add_argument('-i', '--intersect', help='Intersect the files instead of combining them', action='store_true')

    args = parser.parse_args()                                                                                   
    pdb.set_trace()

    a0 = pd.read_csv(args.files[0]).values
    a1 = pd.read_csv(args.files[1]).values
    minLen = min(len(a0), len(a1))
    if len(a0) > len(a1):
        dest = a0
    else:
        dest = a1

    for i in range(0, minLen):
        e0 = a0[i]
        e1 = a1[i]
        # sanity check first
        if e0[0] != e1[0]:
            print("{} and {} don't match".format(e0, e1))

        assert dest[i][0] == e0[0]
        if args.intersect:
            unique = set(e0[1].split()).intersection(set(e1[1].split()))
        else:
            unique = np.unique((e0[1] + ' ' + e1[1]).split())
        dest[i][1] = ''.join(x + ' ' for x in unique)

    predictionDF = pd.DataFrame(dest, columns=['order_id', 'products'])
    predictionDF.to_csv(args.output, index=False)

if __name__ == '__main__':
    pdb.set_trace()
    main()
import numpy as np

def shuffleBatches(tensorTuple, batchSize=64):
    if type(tensorTuple) is list or type(tensorTuple) is tuple: 
        ids = list(range(tensorTuple[0].shape[0]))
        np.random.shuffle(ids)
        for i in range(0,len(ids),batchSize):
            lst = min(len(ids), i + batchSize)
            yield (np.array(x[ids[i:lst],]) for x in tensorTuple)
    else:
        ids = list(range(tensorTuple.shape[0]))
        np.random.shuffle(ids)
        for i in range(0,len(ids),batchSize):
            lst = min(len(ids), i + batchSize)
            yield np.array(tensorTuple[ids[i:lst],])
            
            
def splitSample(tensorTuple, pcts=[1]):
    cumvpct = np.array(pcts)
    cumvpct = cumvpct / np.sum(cumvpct)
    cumvpct = np.append(-0.1, np.cumsum(cumvpct))
    cumvpct[-1] = 1.1 #in order to exclude (1 < 1) situations
    ranges = [(cumvpct[i], cumvpct[i+1]) for i in range(len(pcts))]
    if type(tensorTuple) is list or type(tensorTuple) is tuple:
        z = np.random.uniform(size=tensorTuple[0].shape[0])
        return tuple(tuple(x[(z >= a) & (z < b),] for x in tensorTuple) for (a,b) in ranges)
    else:
        z = np.random.uniform(size=tensorTuple.shape[0])
        return tuple(tensorTuple[(z >= a) & (z < b),] for (a,b) in ranges)
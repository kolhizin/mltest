import numpy as np
import time
import sklearn, sklearn.metrics, sklearn.preprocessing, sklearn.linear_model

def shuffleBatches(tensorTuple, batchSize=64):
    if type(tensorTuple) is tuple: 
        ids = list(range(len(tensorTuple[0])))
        np.random.shuffle(ids)
        for i in range(0,len(ids),batchSize):
            lst = min(len(ids), i + batchSize)
            if type(tensorTuple[0]) is list:
                yield ([x[z] for z in ids[i:lst]] for x in tensorTuple)
            else:
                yield (np.array(x[ids[i:lst],]) for x in tensorTuple)
    else:
        ids = list(range(len(tensorTuple)))
        np.random.shuffle(ids)
        for i in range(0,len(ids),batchSize):
            lst = min(len(ids), i + batchSize)
            if type(tensorTuple) is list:
                yield [tensorTuple[z] for z in ids[i:lst]]
            else:
                yield np.array(tensorTuple[ids[i:lst],])
            
            
def splitSample(tensorTuple, pcts=[1]):
    cumvpct = np.array(pcts)
    cumvpct = cumvpct / np.sum(cumvpct)
    cumvpct = np.append(-0.1, np.cumsum(cumvpct))
    cumvpct[-1] = 1.1 #in order to exclude (1 < 1) situations
    ranges = [(cumvpct[i], cumvpct[i+1]) for i in range(len(pcts))]
    if type(tensorTuple) is tuple:
        z = np.random.uniform(size=len(tensorTuple[0]))
        if type(tensorTuple[0]) is list:
            w = np.array(range(len(tensorTuple[0])))
            return tuple(tuple([x[y] for y in w[(z >= a) & (z < b)]] for x in tensorTuple) for (a,b) in ranges)
        else:            
            return tuple(tuple(x[(z >= a) & (z < b)] for x in tensorTuple) for (a,b) in ranges)
    else:
        z = np.random.uniform(size=len(tensorTuple))
        if type(tensorTuple) is list:
            w = np.array(range(len(tensorTuple)))
            return tuple([tensorTuple[y] for y in w[(z >= a) & (z < b)]] for (a,b) in ranges)
        else:
            return tuple(tensorTuple[(z >= a) & (z < b)] for (a,b) in ranges)
    
def calcBinClassMetrics(X, y, model, cutoff=0.5):
    y_p = np.maximum(0.00001, np.minimum(0.99999, model.predict_proba(X)[:, 1]))
    y_l = np.log(y_p / (1 - y_p))
    y_f = (y_p > cutoff)*1
    gini = sklearn.metrics.roc_auc_score(y, y_l) * 2 - 1
    acc = sklearn.metrics.accuracy_score(y, y_f)
    logloss = sklearn.metrics.log_loss(y, y_p)
    logloss_i = sklearn.metrics.log_loss(y, y_f)
    return (gini, acc, logloss, logloss_i)

def calcBinClassMetrics_Continuous(X, y, model_class=sklearn.linear_model.LogisticRegression(), cutoff=0.5):
    model = model_class.fit(X, y)
    return calcBinClassMetrics(X, y, model, cutoff=cutoff)

def calcBinClassMetrics_Discrete(X, y, model_class=sklearn.linear_model.LogisticRegression(), cutoff=0.5):
    Xt = sklearn.preprocessing.OneHotEncoder().fit_transform(X)
    model = model_class.fit(Xt, y)
    return calcBinClassMetrics(Xt, y, model, cutoff=cutoff)
    
def makeMapping(col1, col2):
    v = np.vstack([np.array(col1), np.array(col2)]).transpose()
    m = [(x[0],x[1]) for x in v]
    m1 = dict(m)
    m2 = dict([(x[1],x[0]) for x in m])
    return m, m1, m2

def runEpoch(tfs, train_set, batch_size, set2feeddict, op_train, op_loss=None, batch_steps=1, verbatim=False):
    total = len(train_set[0]) if type(train_set) is tuple else len(train_set)
    step = 0
    for batch in shuffleBatches(train_set, batchSize=batch_size):
        batchobj = tuple(batch) if type(train_set) is tuple else batch 

        cur_size = len(batchobj[0]) if type(train_set) is tuple else len(batchobj)
        train_dict = set2feeddict(batchobj)
        
        tt0 = time.perf_counter()
        
        if op_loss is None:
            for i in range(batch_steps):
                tfs.run(op_train, feed_dict=train_dict)
        else:
            for i in range(batch_steps):
                (tl, _) = tfs.run([op_loss, op_train], feed_dict=train_dict)
                if i == 0:
                    tl0 = tl

        tl1 = tfs.run(op_loss, feed_dict=train_dict) if op_loss is not None else 0
        tt1 = time.perf_counter()
        step += cur_size
        if verbatim:
            if op_loss is not None:
                print('{0}/{1}:\t{2:.3f} -> {3:.3f}\t{4:.2f} sec'.format(step, total, tl0, tl1, tt1-tt0), end='\r')
            else:
                print('{0}/{1}:\t{2:.2f} sec'.format(step, total, tt1-tt0), end='\r')
    
def runDataset(tfs, calc_set, batch_size, set2feeddict, ops):
    if type(calc_set) is tuple:
        total = len(calc_set[0])
    else:
        total = len(calc_set)
    
    step = 0
    res = []
    while step < total:
        tS = calc_set[step:(step+batch_size)] if type(calc_set) is not tuple else tuple(z[step:(step+batch_size)] for z in calc_set)
        cdict = set2feeddict(tS)
        act_size = min(total, step+batch_size) - step
        
        tmp = tfs.run(ops, feed_dict=cdict)
        res.append((step, act_size, tmp))
        step += batch_size
    return res
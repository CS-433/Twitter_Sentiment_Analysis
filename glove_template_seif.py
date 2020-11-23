#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
    
    
print("loading cooccurrence matrix")
with open('cooc.pkl', 'rb') as f:
    cooc = pickle.load(f)
print("{} nonzero entries".format(cooc.nnz))

nmax = 100
print("using nmax =", nmax, ", cooc.max() =", cooc.max())

print("initializing embeddings")
embedding_dim = 20
np.random.seed(1)
xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

eta = 0.0002alpha = 3 / 4

epochs = 10
losses=[]
for epoch in range(epochs):
        
    loss=0
        
    voc=cooc.shape[0]
    scaleX=np.zeros((voc,embedding_dim))
    scaleY=np.zeros((voc,embedding_dim))
    print("epoch {}".format(epoch))
    for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            
        logn = np.log(n)
        fn = min(1.0, (n / nmax) ** alpha)
        x, y = xs[ix, :], ys[jy, :]
        dot=np.dot(x, y)
        loss +=fn*((np.dot(x, y)-logn)**2)
        scaleX[ix] += 2  * fn * ( dot-logn) *y *eta
        scaleY[jy] += 2  * fn * ( dot-logn) *x *eta
    xs -= scaleX 
    ys -= scaleY 
        
    print(loss)
    losses.append(loss)

			# fill in your SGD code here, 
			# for the update resulting from co-occurence (i,j)
		

    np.save('embeddings', xs)


if __name__ == '__main__':
    main()

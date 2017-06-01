import numpy as np
import random
import csv
import pandas as pd
from datetime import datetime
import _pickle as pickle
from saxpy import SAX

NUM = 835621

def cluster_points(X, strX, mu, sax):
    clusters  = {}
    mu_ind = []

    for i in range(len(X)):

        bestmukey = 0
        bestdist = sax.compare_strings(strX[i], mu[0])
        for j in range(1, len(mu)):
            dist = sax.compare_strings(strX[i], mu[j])
            if dist < bestdist:
                bestdist = dist
                bestmukey = j

        try:
            clusters[bestmukey].append(X[i])
        except KeyError:
            clusters[bestmukey] = [X[i]]

        mu_ind.append(bestmukey)

    return clusters, mu_ind
 
def reevaluate_centers(mu, clusters, sax):

    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(sax.to_letter_rep(np.mean(clusters[k], axis = 0))[0])
    return newmu
 
def has_converged(mu, oldmu):
    if set(mu) == set(oldmu):
        return True
    else:
        return False
    
def sax_kmeans(X, K, wordSize, alphabetSize): 
    '''Cluster by SAX k-means
    
    Args:
        X: 2D np array of dimension (n_households, time)
        K: Number of clusters
        See https://github.com/nphoff/saxpy

    Returns:
        List of K centroids
        List of SAX k-means cluster assignments for each load in X
    '''
    
    np.random.seed(NUM)

    # Initialize to K random centers
    sax = SAX(wordSize=wordSize, alphabetSize=alphabetSize)
    idx = np.random.randint(X.shape[0], size=K)
    xmu =  list(X[idx, :])
    mu = []
    
    for i in range(len(xmu)):
        mu.append(sax.to_letter_rep(xmu[i])[0])   
    oldmu = []

    strX = []
    for i in range(X.shape[0]):
        strX.append(sax.to_letter_rep(X[i])[0])

    #i = 1
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters, mu_ind = cluster_points(X, strX, mu, sax)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters, sax)

    return mu, mu_ind

 
def main():

    params = {
        'date': 12015,
        'numclust': 6,
        'wordSize': 24,
        'alphabetSize': 20
    }


    df = pd.read_csv('data\energy12015manorm_dfoutrem.csv')

    mu, cluster = sax_kmeans(df.as_matrix(), params['numclust'], params['wordSize'], params['alphabetSize'])

if __name__ == '__main__':
    main()

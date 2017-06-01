import numpy as np
import random
import csv
import pandas as pd
from datetime import datetime
import _pickle as pickle
from saxpy import SAX

NUM = 835621

#https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python
#improved seeding https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/
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
    
def sax_kmeans(X, K, wordSize, alphabetSize): #find centers
    
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
        #print i
        #i += 1

    return mu, mu_ind

 
def main():

    params = {
        'date': 12015,
        'numclust': 6,
        'wordSize': 24,
        'alphabetSize': 20
    }

    #df = pd.read_csv('data/df_irreg9356.csv')
    #df = pd.read_csv('data\energy12015manorm_df.csv')
    #df = pd.read_csv('data\high_entropy.csv')
    #df.set_index(['dataid', 'date'], inplace=True)
    df = pd.read_csv('data\energy12015manorm_dfoutrem.csv')
    df = pd.read_csv('data\energy12015manorm_dfoutrem.csv')

    #np.cumsum(df.as_matrix(), axis=1)
    mu, cluster = sax_kmeans(df.as_matrix(), params['numclust'], params['wordSize'], params['alphabetSize']) #mu = list of k arrays/centroids, cluster = cluster assignment, in order

    #print('# in each cluster: ', [[x, cluster.count(x)] for x in set(cluster)] ) #num people in each cluster

    # with open('data\cluster_sax_highentropy' + str(params['numclust']) + '_' + str(params['date']) + '.pkl', 'wb') as pkl_file:
    #         pickle.dump(cluster, pkl_file)
    # with open('data\mu_sax_highentropy' + str(params['numclust']) + '_' + str(params['date']) + '.pkl', 'wb') as pkl_file:
    #         pickle.dump(mu, pkl_file)
    with open('data\cluster_saxmanorm_outrem' + str(params['numclust']) + '_' + str(params['date']) + '.pkl', 'wb') as pkl_file:
            pickle.dump(cluster, pkl_file)
    # with open('data\mu_saxmanorm' + str(params['numclust']) + '_' + str(params['date']) + '.pkl', 'wb') as pkl_file:
    #         pickle.dump(mu, pkl_file)


if __name__ == '__main__':
    main()

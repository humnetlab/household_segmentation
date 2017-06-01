import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import _pickle as pickle
import pandas as pd
import peakutils
import math

def moving_average(x, N=8):
    '''Smoothes noisy time series

    Applies a simple moving average to time series x 

    Args:
        x: time series, 1D numpy array
        N: window size
    '''
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 


def findpeak(X, baseline_poly=1, thres = 0.35, min_dist = 12):
    '''Find the index of the peaks in a function

    Args:
        X: 1D np array containing a normalized load shape. Dimension (1, 24/interval_length)
        See peakutils documentation: http://pythonhosted.org/PeakUtils/

    Returns indices of peak maxima
    '''
    if baseline_poly > 0:
        base = peakutils.baseline(X, baseline_poly)
        peakidx = peakutils.indexes(X - base, thres, min_dist)
    else:
        peakidx = peakutils.indexes(X, thres, min_dist)
    
    if len(peakidx) > 3:
        #increase threshold if too many peaks are detected
        peakidx = peakutils.indexes(X, thres+0.15, min_dist)
        

    return peakidx


def peakinterval(prev_end, peakidx, next_peakidx, slopes, alpha=1.5, symmetry=0.7):
    '''Finds the start and end index of a peak

    Given the index of the peak maximum, finds (peak start, peak end) while controlling for noise

    Args:
        prev_end: index of the peak end of previous peak.
        peakidx: index of current peak maximum.
        next_peakidx: index of next peak maximum.
        slopes: approximate gradient of the load shape given by slopes between each consecutive pair of points.
        alpha: parameter that controls the size of peak intervals. A lower alpha gives a smaller peak range.
            A lower alpha is needed for noiser data to avoid including earlier fluctuations before peak start, etc.
        symmetry: parameter to control symmetry level for distance between (peak start, peak max) and (peak max, peak end).

    Returns (peak start, peak end) for the current peak.
    '''
    
    #lower alpha needed for noiser data (avoid including earlier fluctuations before peak start)
    
    #Find peak start point
    thres = slopes[prev_end:peakidx].max() - alpha*np.std(slopes[prev_end:peakidx])
    start =  prev_end + np.argmax(slopes[prev_end:peakidx] > thres)
    
    #Find peak endpoint    
    halfinterval = peakidx - start
    halfinterval = int(math.ceil(symmetry*halfinterval))
    
    if peakidx + halfinterval >= next_peakidx-1:
        return start, next_peakidx-1
    
    end = peakidx + halfinterval + np.argmax(slopes[peakidx + halfinterval:next_peakidx-1] > 0)

    return start, end


def store_peaks(mu):
    '''Store the peak intervals for an array of centroids

    Args:

        mu: 2D np array of centroids. Dimension (k, 24/interval_length)
    
    Returns a dictionary {centroid i: {peak j: [start, end] for all peaks j in centroid i}}
    '''

    centers = [] 
    for i in range(len(mu)):
        peakidx = findpeak(X = mu[i])
        slopes = np.ediff1d(mu[i])/0.25
        peakidx = np.insert(peakidx, peakidx.shape[0], mu[i].shape[0])

        end = 0
        center = {}

        for j in range(peakidx.shape[0]-1):
            start, end = peakinterval(prev_end = end, peakidx=peakidx[j], next_peakidx=peakidx[j+1], slopes=slopes)
            center[peakidx[j]] = [start, end]
        
        centers.append(center)
    return centers


def peak_accuracy(X, centers, cluster, mu, tol = 0, alpha = 1.5, symmetry=0.7):
    '''Calculate the peak overlap metric for an array of load shapes with respect to their assigned centroids

    Args:
        X: 2D np array of normalized load shapes. Dimension (n_loads, 24/interval_length)
        centers: dictionary given by store_peaks
        cluster: list containing cluster assignments of each load in X

    Returns list of peak overlap percentages of each load in X with respect to their assigned centroids
    '''

    perc_area_overlap = []

    for i in range(X.shape[0]):
        
        try:
            day = X[i]
            peakidx = findpeak(X = day)
        except ValueError:
            day = X[i][0]
            peakidx = findpeak(X = day)

        if len(peakidx)==0:
            perc_area_overlap.append(np.nan)
            continue

        slopes = np.ediff1d(day)/0.25
        peakidx = np.insert(peakidx, peakidx.shape[0], X.shape[1]-1)

        area = 0
        totarea = 0
        dayend = 0
        overlapped = []
    
        for j in range(peakidx.shape[0]-1):

            daystart, dayend = peakinterval(prev_end = dayend, peakidx=peakidx[j], next_peakidx=peakidx[j+1], slopes=slopes)

            mupeakidx = np.array(list(centers[cluster[i]].keys()))
            mupeak_valid = (mupeakidx >= daystart - tol) & (mupeakidx <= dayend + tol)
            
            if mupeak_valid.sum()==0:
                totarea += np.trapz(y = day[daystart:dayend], dx = 0.25)
                continue

            overlapidx = mupeakidx[np.argmax(mupeak_valid)]
            overlapped.append(overlapidx)
            mustart, muend = centers[cluster[i]][overlapidx]

            startov = np.maximum(daystart, mustart)
            endov = np.minimum(dayend, muend)

            starttot = np.minimum(daystart, mustart)
            endtot = np.maximum(dayend, muend)
            
            #Union of area of daily peak and cluster center
            totarea += np.trapz(y = np.maximum(day[starttot:endtot], mu[cluster[i]][starttot:endtot]), dx = 0.25)

            #Overlapping area with cluster center
            area += np.trapz(y = np.minimum(day[startov:endov], mu[cluster[i]][startov:endov]), dx = 0.25)
            
        extra_centroid_peaks = list(set(list(centers[cluster[i]].keys())) - set(overlapped))
        for mupeakidx in extra_centroid_peaks:
            mustart, muend = centers[cluster[i]][mupeakidx]
            totarea += np.trapz(y = mu[cluster[i]][mustart:muend], dx = 0.25)
            
        perc_area_overlap.append(float(area)/totarea)

    return perc_area_overlap
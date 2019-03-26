#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 11:29:32 2018
clustering wraper
@author: Tu Bui tb00083@surrey.ac.uk
"""
from sklearn import cluster, mixture
from scipy.spatial import distance
import numpy as np
from sklearn.metrics import silhouette_score
from AutoGMM import autoGMM
from xmeans import XMeans

class pycluster(object):
  def __init__(self, ctype = 'kmeans', auto_alg = None, **kwargs):
    assert ctype in ['kmeans', 'gmm','xmeans','meanshift','dbscan']
    self.type = ctype
    assert auto_alg in ['auto', 'bic','silhouette', 'aic', None]
    self.auto_alg = auto_alg
    self.params = kwargs
    self.model = self.n_clusters = self.cluster_centers_ = None
    
  def fit(self, X, k_min = 1, k_max = 10):
    """main method to do clustering
    X: input array of shape [N,D]
    k_min: min number of clusters, used when auto_alg is bic or silhouette, default None
    k_max: max number of clusters
    """
    if self.type == 'kmeans':
      if self.auto_alg == 'bic':
        self.model = self.kmeans_bic(X, k_min, k_max)
      elif self.auto_alg == 'silhouette':
        self.model = self.kmeans_silhouette(X, k_min, k_max)
      elif self.auto_alg is None:
        self.model = cluster.KMeans(n_clusters = k_max, init="k-means++").fit(X)
      else:
        raise ValueError('auto_alg {} unknown or not yet implemented'.format(self.auto_alg))
        
      self.n_clusters = self.model.n_clusters
      self.cluster_centers_ = self.model.cluster_centers_
      bincount = np.bincount(self.model.labels_)
      self.weights_  = bincount/float(bincount.sum())
      self.labels = self.model.labels_
    elif self.type == 'gmm':
      if self.auto_alg == 'auto':
        self.model = autoGMM(X)
      elif self.auto_alg == 'bic':
        self.model = self.gmm_bic(X, k_min, k_max)
      elif self.auto_alg == 'aic':
        self.model = self.gmm_aic(X, k_min, k_max)
      elif self.auto_alg is None:
        self.model = mixture.GaussianMixture(n_components=k_max, covariance_type='full').fit(X)
      else:
        raise ValueError('auto_alg {} unknown or not yet implemented for {}'.format(self.auto_alg, self.type))
        
      self.n_clusters = self.model.n_components
      self.cluster_centers_ = self.model.means_
      self.weights_ = self.model.weights_
      self.labels = self.model.predict(X)
    elif self.type == 'meanshift':
      bw = cluster.estimate_bandwidth(X, n_samples=200)
      self.model = cluster.MeanShift(bandwidth=bw, n_jobs=4).fit(X)
      self.cluster_centers_ = self.model.cluster_centers_
      self.labels = self.model.labels_
      self.n_clusters = len(np.unique(self.labels))
      bincount = np.bincount(self.labels)
      self.weights_  = bincount/float(bincount.sum())
    elif self.type == 'dbscan':
      self.model = cluster.DBSCAN(n_jobs=4).fit(X)
      self.labels = self.model.labels_
      self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
      self.cluster_centers_ = np.array([X[self.labels==i].mean(axis=0) for i in range(self.n_clusters)])
      bincount = np.array([np.sum(self.labels==i) for i in range(self.n_clusters)], dtype=np.float32)
      self.weights_ = bincount/bincount.sum()
      
    elif self.type == 'xmeans':
      self.model = XMeans(k_min,k_max).fit(X)
      self.n_clusters = self.model.n_clusters
      self.cluster_centers_ = self.model.cluster_centers_
      bincount = np.bincount(self.model.labels_)
      self.weights_  = bincount/float(bincount.sum())
      
    else:
      k = -10
      self.model = cluster.AffinityPropagation(preference=k).fit(X)
      n_clusters = len(self.model.cluster_centers_indices_)
      while (n_clusters == 0 and k < 0):
        k += 2
        self.model = cluster.AffinityPropagation(preference=k).fit(X)
        n_clusters = len(self.model.cluster_centers_indices_)
      self.n_clusters = n_clusters
      self.cluster_centers_ = X[self.model.cluster_centers_indices_]
  
  def predict(self, X):
    return self.model.predict(X)
  
  def predict_proba(self, X):
    if self.type == 'gmm':
      return self.model.predict_proba(X)
    elif self.type == 'kmeans':#kmeans, need to take opposite of distance for proba
      return np.exp(-self.model.transform(X))
    else: #uniform proba
      return np.ones((X.shape[0], len(self.weights_)), dtype=np.float32)
  
  def kmeans_bic(self, X, k_min, k_max):
    """clustering using kmeans
    automatically choose number of clusters
    """
    ks = range(k_min,k_max+1)
    kmeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
    BICs = [self.compute_bic(kmeansi,X) for kmeansi in kmeans]
    best_id = BICs.index(max(BICs))
    return kmeans[best_id]
  
  def kmeans_silhouette(self, X, k_min, k_max):
    ks = range(k_min,k_max+1)
    kmeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(X) for i in ks]
    pred_labels = [kmeansi.predict(X) for kmeansi in kmeans]
    n_clts = [len(np.unique(pred_label)) for pred_label in pred_labels]
    SILs = [silhouette_score(X, pred_labelsi) for pred_labelsi in pred_labels]
    best_id = SILs.index(max(SILs))
    return kmeans[best_id]
  
  def gmm_bic(self, X, k_min, k_max):
    """clustering using gmm
    auto select the best cluster number"""
    ks = range(k_min, k_max+1)
    gmms = [mixture.GaussianMixture(n_components=k, covariance_type='diag').fit(X) for k in ks]
    BICs = [gmm.bic(X) for gmm in gmms]
    best_id = BICs.index(min(BICs)) #lower is better
    return gmms[best_id]
  
  def gmm_aic(self, X, k_min, k_max):
    """clustering using gmm
    auto select the best cluster number"""
    ks = range(k_min, k_max+1)
    gmms = [mixture.GaussianMixture(n_components=k, covariance_type='diag').fit(X) for k in ks]
    AICs = [gmm.aic(X) for gmm in gmms]
    best_id = AICs.index(min(AICs)) #lower is better
    return gmms[best_id]
  
  def compute_bic(self, kmeans, X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])
    
    sf = 1.0 #by default 1.0
    const_term = 0.5 * m * np.log(N) * (d+1) * sf

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)
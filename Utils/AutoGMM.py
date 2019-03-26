#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:43:16 2018
GMM with auto selection of component number

@author: Tu Bui tb00083@surrey.ac.uk
"""
from sklearn import cluster, mixture
import numpy as np

def autoGMM(X, **kwargs):
  params = kwargs
  bestpen = float('inf')
  pen = []
  k = 1
  mix = doGMM(X, k, **params)
  pen.append(GMM_penalty(X, mix))
  while pen[-1] < bestpen:
    bestmix = mix
    bestpen = pen[-1]
    k += 1
    mix = doGMM(X, k, **params)
    pen.append(GMM_penalty(X, mix))
    
  #go uphill at least 2 steps
  stopk = k+2
  while k < stopk:
    k = k+1
    if X.shape[0] <= k:
      break
    mix = doGMM(X, k, **params)
    pen.append(GMM_penalty(X, mix))
    if pen[-1] < bestpen:
      bestmix = mix
      bestpen = pen[-1]
      stopk = k+2
  #print pen
  return bestmix

def doGMM(X, n_components, covariance_type = 'diag', tol=1e-4, **kwargs):
  params = kwargs
  kmean = cluster.KMeans(n_components,init='k-means++').fit(X)
  #return mixture.GaussianMixture(n_components, covariance_type='diag',tol=tol, **params).fit(X)
  return mixture.GaussianMixture(n_components, covariance_type=covariance_type,tol=tol,
                                 means_init = kmean.cluster_centers_, **params).fit(X)
  

def GMM_penalty(X, mix):
  """
  X: data DxN
  mix: mixture model
  return penalty (lower is better)

  Robert's et al Bayesian criterion
  % S.J. Roberts, D. Husmeier, I. Rezek, and W. Penny
  % 'Bayesian approaches to Gaussian mixture modelling'
  % IEEE TPAMI
  %
  % Adaptations are:
  % (1) Using likelihood * prior to compute L:
  %     a reading of Roberts et. al suggest using likelihood only,
  %     but experiments (REFERENCE HERE) show the adaptation is
  %     preferable.
  % (2) Using absolute value of log to compute pen5, below. This
  %     prevents very small priors from reducing the penalty term:
  %     clearly we want priors of low value to raise the term (see below).

  """
  alpha = 1
  beta = 1
  eps = np.finfo(np.float32).eps
  k = mix.n_components
 
  Nd, nd = X.shape
  sigpop = max((np.linalg.norm(X.max(axis=0) - X.min(axis=0)), eps))
  like = GMM_likelihood(X, mix)#mix.predict_proba(X)#
  pen1 = k*nd*np.log2(2*alpha*beta*sigpop**2)
  pen2 = np.log2(np.math.factorial(k-1))
  pen3 = GMM_Np(mix)/2.0*np.log2(2*np.pi)
  likeprior = like * mix.weights_[None,:]
  sumlikeprior = np.sum( likeprior, axis=1)
  L = np.sum( np.log2(np.where(sumlikeprior > eps, sumlikeprior, eps)))
  sumlikeprior = sumlikeprior + (sumlikeprior <= 0).astype(np.float32)
  post = likeprior / sumlikeprior[:,None]
  postp = post / mix.weights_[None,:]
  
  postpp = postp[:,1:] - postp[:,0][:,None]
  pen4 = 0.5*np.sum( np.log2( np.sum(postpp**2 + eps, axis = 0 ) ) )
  
  pen5 = nd*np.sum(np.abs(np.log2(np.sqrt(2)*Nd*mix.weights_)))
  pen6 = 0
  for j in range(k):
    if mix.covariance_type == 'diag':
      s = mix.covariances_[j]
    else:
      s,_ = np.linalg.eig(mix.covariances_[j])
# =============================================================================
#     if len(mix.covariances_[j].shape)==2:
#       s,_ = np.linalg.eig(mix.covariances_[j])
#     else:
#       s = mix.covariances_[j]
# =============================================================================
    pen6 = pen6 + sum( np.log2( eps+np.abs(s) ) )
    
  rob = L - pen1 + pen2 + pen3 - pen4 - pen5 + pen6;
  e = -rob
  #print('k={}: pen1={}, pen2={}, pen3={},pen4={}, pen5={}, pen6={}, L={}, total={}'.format(k,pen1,pen2,pen3,pen4,pen5,pen6,L,e))
  return e

def GMM_likelihood(X, mix):
  """
  input: X   data NxD
  mix: gmm model
  
  out: NxK prob
  """
  eps = np.finfo(np.float32).eps
  k = mix.n_components
  Nd, nd = X.shape
  like = np.zeros((Nd,k), dtype= np.float32)
  #normal = np.sqrt((2*np.pi)**nd)
  for i in range(k):
    diffs = X - mix.means_[i][None,:]
    covars = np.diag(mix.covariances_[i]) if mix.covariance_type == 'diag' else mix.covariances_[i]
    #covars = mix.covariances_[i] if len(mix.covariances_[i].shape)==2 else np.diag(mix.covariances_[i])
    U, S, V = np.linalg.svd(covars)
    S = np.where(S > eps, S, eps)
    sqs = np.sqrt(np.prod(S*2*np.pi))
    #sqs = np.sqrt(np.prod(S))
    if sqs < eps:
      sqs = eps
    a = np.exp( -0.5*np.sum(((np.dot(diffs,U))**2) / S[None,:],axis=1) )/ sqs
    like[:,i] = a
  return like

def GMM_Np(mix):
  k = mix.n_components
  nd = mix.means_.shape[1]
  if mix.covariance_type == 'diag':
    Np = (k-1) + 2*k*nd
  else:
    Np = (k-1) + k*nd + k*nd*(nd+1)/2
  return Np
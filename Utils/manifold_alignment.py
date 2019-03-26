#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 12:10:16 2018
manifold alignment between two domains
https://people.cs.umass.edu/~mahadeva/papers/bookchapter.pdf
@author: Tu Bui tb00083@surrey.ac.uk
"""
import numpy as np
from scipy.spatial import distance as dist
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors

class manifold(object):
  def __init__(self, Xua, Xla, Xub, Xlb, u=1.0, knn = None):
    """
    Xua (Nua x Da) and Xub (Nub x Db) are 2 datasets of domain A and B that we assume having same manifold
    Xla (nl x Da) and Xlb (nl x Db) are labelled sets (i.e. having correspondants) of A and B, 
      where Xla[i,.] and Xlb[i,.] are a correspondent pair
    u: weight of inter-domain correlation over intra-domain on the loss function, u=1.0 mean inter- and intra-domain
       contribute equally
    """
    self.nua, self.dima = Xua.shape
    self.nub, self.dimb = Xub.shape
    self.nl = Xla.shape[0]
    Xa = np.r_[Xua,Xla]
    Xb = np.r_[Xub,Xlb]
    #step1.0: build joint matrix X
    self.X = np.r_[
        np.pad(np.r_[Xua,Xla], ((0,0),(0, self.dimb)),'constant',constant_values=0),
        np.pad(np.r_[Xub,Xlb], ((0,0),(self.dima, 0)),'constant',constant_values=0)
        ]
    
    #step1.1: build adjaction matrix W
    if knn:
      nbra = NearestNeighbors(knn).fit(Xa)
      maska = nbra.kneighbors_graph(Xa).toarray()
      nbrb = NearestNeighbors(knn).fit(Xb)
      maskb = nbrb.kneighbors_graph(Xb).toarray()
    else:
      maska = np.ones((self.nua+self.nl, self.nua+self.nl), dtype=np.float32)
      maskb = np.ones((self.nub+self.nl, self.nub+self.nl), dtype=np.float32)
    Wa = dist.squareform(np.exp(-dist.pdist(Xa))) * maska
    Wb = dist.squareform(np.exp(-dist.pdist(Xb))) * maskb
    Wl = u*np.eye(self.nl, dtype=np.float32)
    tmp = np.pad(Wl, ((self.nua,0), (self.nub,0)),'constant',constant_values=0)
    W = np.r_[
        np.c_[Wa, tmp],
        np.c_[tmp.T, Wb]
        ]
    
    #step1.2: build D and joint laplacian matrix L
    self.D = np.diag(W.sum(axis=1))
    self.L = self.D - W
    
  def align_nonlinear(self, latten_dim = None, return_aux = False):
    val, vec = eigh(self.L, self.D)
    latten_dim_ = latten_dim if latten_dim else min(self.dima, self.dimb)
    val = val[:latten_dim_]
    vec = vec[:, :latten_dim_]
    split_points = np.cumsum([self.nua, self.nl, self.nub])
    if return_aux:
      return np.split(vec, split_points, axis=0), {'val': val}
    else:
      return np.split(vec, split_points, axis=0)
    
  def align_linear(self, latten_dim = None, return_aux = False):
    L_ = self.X.T.dot(self.L).dot(self.X)
    D_ = self.X.T.dot(self.D).dot(self.X)
    val, vec = eigh(L_,D_)
    latten_dim_ = latten_dim if latten_dim else min(self.dima, self.dimb)
    val = val[:latten_dim_]
    vec = vec[:, :latten_dim_]
    vec_ = self.X.dot(vec)
    split_points = np.cumsum([self.nua, self.nl, self.nub])
    if return_aux:
      veca, vecb = np.split(vec, (self.dima,))
      aux = {'val': val, 'veca': veca, 'vecb':vecb}
      return np.split(vec_, split_points,axis=0), aux
    else:
      return np.split(vec_, split_points,axis=0)
    
class matrixWarp(object):
  def __init__(self, lamda = 5e-1, niter = 100, lr=0.01,tol=1e-5, method='sgd'):
    self.lamda = lamda
    self.niter = niter
    self.lr = lr
    self.tol = tol
    self.method = method
  
  def learn(self, X, Y):
    """
    learn matrix warp from src to destination
    """
    N, dim = X.shape
    if self.method == 'sgd':
      I = np.eye(dim,dtype=np.float32)#np.r_[np.eye(newdim), np.zeros((1,newdim))].astype(np.float32)#
      self.W =np.zeros((dim+1, dim), dtype=np.float32)
      W = np.random.normal(0,0.01,(dim,dim)).astype(np.float32)
      b = np.zeros((1,dim),np.float32)
      #L = np.zeros((niter,2),np.float32)
      prevL = 0
      for i in range(self.niter):
        db = X.dot(W) + b - Y
        L_ = 0.5/N*np.sum(db.dot(db.T))+ 0.5*self.lamda*np.sum((W-I)**2)
        #L[i,...] = np.array((0.5/N*np.sum(db.dot(db.T)),0.5*lamda*np.sum((W-I)**2)))
        dLdW = X.T.dot(db)/N + self.lamda*(W-I)
        W = W - self.lr*dLdW
        b = b - self.lr/N*db.sum(axis=0)
        if i >0 and np.abs(L_ - prevL) < self.tol: break
        prevL = L_
      self.W = np.r_[W,b]
      #totalL.append(L)
    else:
      I = np.r_[np.eye(dim), np.zeros((1,dim))].astype(np.float32)#
      self.W =np.zeros((self.ncats, dim+1, dim), dtype=np.float32)
      X_ = np.c_[X, np.ones((N,1), np.float32)]#skt_feats[skt_id]#
      try:
        self.W = np.linalg.inv(X_.T.dot(X_)+N*self.lamda).dot(X_.T.dot(Y)+N*self.lamda*I)
# =============================================================================
#         X = X[:,:-1]
#         self.W=np.r_[np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y)), np.zeros((1, dim))]
# =============================================================================
      except:
        raise Exception('Matrix not invertable')
        
  def warp(self, X):
    X_ = np.c_[X, np.ones((X.shape[0],1), dtype=np.float32)]
    return X_.dot(self.W)
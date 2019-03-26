#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:37:36 2018

@author: Tu Bui tb00083@surrey.ac.uk
"""
import sys
import numpy as np

class retrieval(object):
  """class for image retrieval
  """
  def __init__(self, query_feats=None, query_labels=None, src_feats=None, src_labels=None):
    self.reg(query_feats, query_labels, src_feats, src_labels)
    #self.pool = ThreadPool()
  def reg(self, query_feats, query_labels, src_feats, src_labels):
    self.query_feats = query_feats + np.zeros((1,1), dtype=np.float32) if query_feats is not None else None
    self.query_labels = query_labels[:,None] if query_labels is not None else None
    self.src_feats = src_feats + np.zeros((1,1), dtype=np.float32) if src_feats is not None else None
    self.src_labels = np.array(src_labels).squeeze()
    
  def dist_L2(self, a_query):
    """
    Eucludean distance between a (single) query & all features in database
    used in pdist
    """
    return np.sum((a_query-self.src_feats)**2,axis=1)
    
  def pdist(self):
    """
    Compute distance (L2) between queries and data features
    query_feat: NxD numpy array with N features of D dimensions
    """
    res = [np.sum((self.query_feats[i]-self.src_feats)**2,axis=1) for i in range(self.query_feats.shape[0])]
#==============================================================================
#     threads = [self.pool.apply_async(self.dist_L2,(query_feat[i],)) for \
#                 i in range(query_feat.shape[0])]
#     res = [thread.get() for thread in threads]
#==============================================================================
    res = np.array(res)
    return res
  
  def retrieve(self):
    """retrieve: return the indices of the data points in relevant order
    """
    res = self.pdist()
    return res.argsort()
  
  def retrieve2file(self,queries, out_file, num=0):
    """perform retrieve but write output to a file
    num specifies number of returned images, if num==0, all images are returned
    """
    res = self.retrieve(queries)
    if num > 0 and num < self.data['feats'].shape[0]:
      res = res[:,0:num]
    if out_file.endswith('.npz'):
      np.savez(out_file, results = res)
    return 0
    
  def compute_mAP(self, top_k = None):
    """
    compute mAP given queries in .npz format
    """
    ids = self.retrieve()
    ret_labels = self.src_labels[ids]
    # relevant
    rel = ret_labels == self.query_labels
    if top_k:
      mAP = rel[:,:top_k].sum()/float(top_k)/len(rel)
    else:
      P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
      AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
      mAP = AP.mean()
    return mAP

import caffe
from caffe_func_utils import read_mean, read_db
from augmentation2 import ImageAugment
import time
from datetime import timedelta

class featExtractor(object):
  def __init__(self, batchsize=10, gpu = None, verbose = True):
    self.batchsize = batchsize
    self.model = None
    if gpu is not None:
      caffe.set_device(gpu)
      caffe.set_mode_gpu()
    self.verbose = verbose
  def set_model(self, deploy, weights, mean_file = 'mean_pixel', layer=0):
    if mean_file == 'mean_pixel':
      mean_file = [104, 117,123]
    self.mean_ = read_mean(mean_file)
    self.model = None
    self.model = caffe.Net(deploy, caffe.TEST, weights = weights)
    self.inblob = self.model.inputs[0]
    self.in_dim = self.model.blobs[self.inblob].data.shape[1:]
    self.prep = ImageAugment(mean=self.mean_,shape=self.in_dim[-2:],scale=1.0, verbose = self.verbose)
    
    self.feat_l = self.model.blobs.keys()[layer-1]
    self.out_dim = self.model.blobs[self.feat_l].data.shape[1]
    #bookeeping
    self.deploy = deploy
    self.weights = weights
    self.mean_file = mean_file
    
  def get_feat(self, DB, aux=None, enrich = False):
    db = read_db(DB, aux, self.verbose)
    label_lst = db.get_label_list()
    if type(label_lst) is tuple:
      NIMGS = len(label_lst[0])
    else:
      NIMGS = len(label_lst)
    if self.verbose:
      print 'Extracting cnn feats...'
      print '  Database: {}'.format(DB)
      print '  NIMGS: {}'.format(NIMGS)
      print '  Batchsize: {}'.format(self.batchsize)
      print '  Model def: {}\n  Weights: {}'.format(self.deploy, self.weights)
      print '  Mean image: {}'.format(self.mean_file)
      start_t = time.time()
    feats = np.zeros((NIMGS,self.out_dim),dtype=np.float32)

    for i in xrange(0,NIMGS,self.batchsize):
      batch = range(i,min(i+self.batchsize,NIMGS))
      #print 'batch #{}-{}'.format(batch[0],batch[-1])
      new_shape = (len(batch),) + self.in_dim
      self.model.blobs[self.inblob].reshape(*new_shape)
      
      chunk = db.get_data(batch)
      chunk_aug = self.prep.augment_deploy(chunk, enrich, crop = False) #1: crop has large impact on accuracy
      
      new_shape = (chunk_aug.shape[0],) + self.in_dim
      self.model.blobs[self.inblob].reshape(*new_shape)
      self.model.blobs[self.inblob].data[...] = chunk_aug
      _ = self.model.forward()
      out=self.model.blobs[self.feat_l].data.squeeze()
      feats[batch] = out.reshape((len(batch),-1,self.out_dim)).mean(axis=1)
    
    if self.verbose:
      end_t = time.time()
      print 'Time: {}\n'.format(timedelta(seconds = int(end_t-start_t)))
    return feats, label_lst
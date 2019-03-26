#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 13:23:02 2018
http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
@author: Tu Bui tb00083@surrey.ac.uk


# Copyright (c) 2008 Carnegie Mellon University
#
# You may copy and modify this freely under the same terms as
# Sphinx-III


Divergence and distance measures for multivariate Gaussians and
multinomial distributions.

This module provides some functions for calculating divergence or
distance measures between distributions, or between one distribution
and a codebook of distributions.
"""

__author__ = "David Huggins-Daines <dhuggins@cs.cmu.edu>"
__version__ = "$Revision$"

import numpy as np
eps = np.finfo(np.float32).eps
class divergence_metric(object):
  """compute divergence distance between two gaussian distribution
  assume diagonal covariance"""
  def __init__(self, metric = 'kl', inverse = False):
    """
    inverse: consider the opposite direction ie. compute d(q|p) instead of d(p|q)
    """
    assert metric in ['kl', 'bh', 'js']
    self.metric = metric
    self.metric_dict = {'kl': gau_kl, 'bh': gau_bh, 'js': gau_js}
    self.inverse = inverse
    
  def dist(self, PM, PV, QM, QV):
    """compute distance between gau P and Q
    P and Q can be sets of gaussian"""
    
    if len(PM.shape) == 2:
      out = []
      for g in range(PM.shape[0]):
        out.append(self.metric_dict[self.metric](PM[g], PV[g], QM, QV))
    else:
      out = self.metric_dict[self.metric](PM, PV, QM, QV)
    
    out = np.array(out).squeeze() + np.zeros((1,1))
    return out
  
  def neighbour(self, PM, PV, QM, QV):
    """find nearest neighbour of P -> Q"""
    if self.inverse:
      d = self.dist(QM, QV, PM, PV)
      ids = d.argmin(axis = 0)
    else:
      d = self.dist(PM, PV, QM, QV)
      ids = d.argmin(axis = 1)
    return ids
  
def gau_bh(pm, pv, qm, qv):
    """
    Classification-based Bhattacharyya distance between two Gaussians
    with diagonal covariance.  Also computes Bhattacharyya distance
    between a single Gaussian pm,pv and a set of Gaussians qm,qv.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Difference between means pm, qm
    diff = qm - pm
    # Interpolated variances
    pqv = (pv + qv) / 2.
    # Log-determinants of pv, qv
    ldpv = np.log(pv).sum()
    ldqv = np.log(qv).sum(axis)
    # Log-determinant of pqv
    ldpqv = np.log(pqv).sum(axis)
    # "Shape" component (based on covariances only)
    # 0.5 log(|\Sigma_{pq}| / sqrt(\Sigma_p * \Sigma_q)
    norm = 0.5 * (ldpqv - 0.5 * (ldpv + ldqv))
    # "Divergence" component (actually just scaled Mahalanobis distance)
    # 0.125 (\mu_q - \mu_p)^T \Sigma_{pq}^{-1} (\mu_q - \mu_p)
    dist = 0.125 * (diff * (1./pqv) * diff).sum(axis)
    return dist + norm

def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2): #one to many
        axis = 1
    else: #one to one
        axis = 0
    diag = True if len(pv.shape) == 1 else False
    if diag:
      # Determinants of diagonal covariances pv, qv
      dpv = np.where(pv>eps, pv, eps).prod()
      dqv = np.where(qv>eps, qv ,eps).prod(axis)
      # Inverse of diagonal covariance qv
      iqv = 1./qv
      # Difference between means pm, qm
      diff = qm - pm
      return (0.5 *
              (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
               + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
               + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
               - len(pm)))                     # - N
    else:#assume full matrix
      dpv = np.linalg.det(pv)
      dqv = np.linalg.det(qv)
      iqv = np.linalg.inv(qv)
      num_q = len(dqv)
      diff = qm-pm
      if num_q == 1:
        return 0.5*\
        (np.log(dqv/dpv) - len(pm) + np.sum(iqv*pv.T) + 
         diff.dot(iqv).dot(diff.T))
      else:
        out = [0.5*\
          (np.log(dqv[k]/dpv) - len(pm) + np.sum(iqv[k]*pv.T) + 
           diff[k].dot(iqv[k]).dot(diff[k].T)) for k in range(num_q)]
        return np.array(out)

def gau_js(pm, pv, qm, qv):
    """
    Jensen-Shannon divergence between two Gaussians.  Also computes JS
    divergence between a single Gaussian pm,pv and a set of Gaussians
    qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = np.where(pv, pv, 1).prod()
    dqv = np.where(qv, qv ,1).prod(axis)
    # Inverses of diagonal covariances pv, qv
    iqv = 1./qv
    ipv = 1./pv
    # Difference between means pm, qm
    diff = qm - pm
    # KL(p||q)
    kl1 = (0.5 *
           (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
            + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
            + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
            - len(pm)))                     # - N
    # KL(q||p)
    kl2 = (0.5 *
           (np.log(dpv / dqv)            # log |\Sigma_p| / |\Sigma_q|
            + (ipv * qv).sum(axis)          # + tr(\Sigma_p^{-1} * \Sigma_q)
            + (diff * ipv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
            - len(pm)))                     # - N
    # JS(p,q)
    return 0.5 * (kl1 + kl2)

def multi_kl(p, q):
    """Kullback-Liebler divergence from multinomial p to multinomial q,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Clip before taking logarithm to avoid NaNs (but still exclude
    # zero-probability mixtures from the calculation)
    return (p * (np.log(p.clip(1e-10,1))
                 - np.log(q.clip(1e-10,1)))).sum(axis)

def multi_js(p, q):
    """Jensen-Shannon divergence (symmetric) between two multinomials,
    expressed in nats."""
    if (len(q.shape) == 2):
        axis = 1
    else:
        axis = 0
    # D_{JS}(P\|Q) = (D_{KL}(P\|Q) + D_{KL}(Q\|P)) / 2
    return 0.5 * ((q * (np.log(q.clip(1e-10,1))
                        - np.log(p.clip(1e-10,1)))).sum(axis)
                      + (p * (np.log(p.clip(1e-10,1))
                              - np.log(q.clip(1e-10,1)))).sum(axis))
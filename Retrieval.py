#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:21:33 2018

@author: Tu Bui tb00083@surrey.ac.uk
"""
GPU = 0
# proposed model
WEIGHTS = 'model/triplet1_InceptionV1_InceptionV1_noshare_65cfine_cluster_triplet_magnet_gmm_batchrnd_pretrainS3_iter_26000.caffemodel'
# mid-grain benchmark
SKT_DB = 'data/sketches'
IMG_DB = 'data/images'
SKT_LST = 'data/sketch_list.txt'
IMG_LST = 'data/image_list.txt'
batchsize = 300

import sys,os
os.environ['GLOG_minloglevel'] = '2' 
sys.path.insert(1,'Utils')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
from sbir import featExtractor, retrieval
import numpy as np

mean_file = 'mean_pixel'
SIG = re.findall(r'^(.*)/triplet1(.*)_iter_(.*).caffemodel$',WEIGHTS)[0][1]
DEPLOY_SKT = 'model/deploy_sketch_net1'+ SIG + '.prototxt'
DEPLOY_IMG = 'model/deploy_images_net1'+ SIG + '.prototxt'
print('signature: ' + SIG)

###### print some info #######
print('skt list: {}'.format(os.path.basename(SKT_LST)))
print('img list: {}'.format(os.path.basename(IMG_LST)))
print('weights: {}'.format(os.path.basename(WEIGHTS)))
sys.stdout.flush()
extractor = featExtractor(batchsize, GPU, verbose = True)
ret_ = retrieval()

#extract skt
extractor.set_model(DEPLOY_SKT, WEIGHTS, mean_file)
skt_feats, skt_labels = extractor.get_feat(SKT_DB, SKT_LST)

#extract img
extractor.set_model(DEPLOY_IMG, WEIGHTS, mean_file)
img_feats, img_labels = extractor.get_feat(IMG_DB, IMG_LST)

#retrieval
mAP = []
for l in range(2):
  ret_.reg(skt_feats, skt_labels[l], img_feats, img_labels[l])
  mAP.append(ret_.compute_mAP())

print('mAP (class-level, midgrain-level)= {}'.format(mAP))
print('Done')
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:51:10 2016

@author: tb00083
"""

import numpy as np
import caffe
import h5py as h5
import subprocess
import os, sys, time
import scipy.io as sio
from caffe_class_utils import lmdbs, imgdb, jsondb, svgs
from augmentation2 import ImageAugment
from datetime import timedelta
import matplotlib.pyplot as plt
from helper import helper, progressbar
#from svg.SVGProcessor import SVGProcessor
import json

def caffe_set_device(type, id=0):
  if type.lower() == "gpu":
    caffe.set_device(id)
    caffe.set_mode_gpu()
  else:
    caffe.set_mode_cpu()

def mat2py_imdb(mat_file):
  """
  Convert matlab .mat file version 7.3 to numpy array
  You must know the structure of the mat file before hand.
  Here is for imdb mat file only
  """
  assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
  f = h5.File(mat_file)
  data = np.array(f['images']['data'])
  labels = np.array(f['images']['labels'])
  #img_mean = np.array(f['images']['data_mean'])
  #matlab store data column wise so we need to transpose it
  return data.transpose().astype(np.float32), labels.astype(np.float32)

def mat2py_mean(mat_file):
  """
  Convert matlab .mat file version 7.3 to numpy array
  You must know the structure of the mat file before hand.
  Here is for mat file containing matrix data_mean only
  """
  assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
  f = h5.File(mat_file)
  data_mean = np.array(f['data_mean'])
  return data_mean.transpose()

def biproto2py(binary_file):
  """
  read binaryproto (usually mean image) to array
  """
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open( binary_file , 'rb' ).read()
  blob.ParseFromString(data)
  arr = np.array( caffe.io.blobproto_to_array(blob) )
  #out = np.ascontiguousarray(out.transpose(1,2,0))
  out = np.ascontiguousarray(arr)
  out = out.astype(np.float32)
  return out
  
def compute_img_mean(lmdb_path, out_path):
  exe = '/vol/vssp/ddawrk/Tu/Toolkits/caffe/caffe_rollo/caffe/build/tools/compute_image_mean'
  if not os.path.isfile(exe):
    sys.exit('program {} does not exist'.format(exe))
  cmd = [exe, lmdb_path, out_path]
  process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
  process.wait()
  for line in process.stdout:
      print(line)

def svg2pickle(SVG_SRC, path_lst, out):
  """build svg database
  IN:  SVG_SRC  directory containing svg files
       path_lst file containing list of the svg images + labels
  OUT: out      output file (pickle)
  """
  helps = helper()
  lists = helps.read_list(path_lst,',',keep_original=False)
  labels = lists[0]
  paths = lists[1]
  if paths[0].endswith('.png'):
    paths = [path.replace('.png','.svg') for path in paths]
    
  data = [SVGProcessor(os.path.join(SVG_SRC,path)) for path in paths]
  helps.save(out,labels = labels, data = data)

def lmdb_get_info(DB):
  """
  get information about an lmdb
  IN:   DB    can be either a true lmdb or a python pickle
  includes: number of classes, number of sample per class, number of samples
  """
  if DB.endswith('.pkl'):
    helps = helper()
    data = helps.load(DB,1)  #load the first variable which is the labels
    labels = data['labels']
  elif DB.endswith('.json'): #google quickdraw json
    with open(DB,'r') as fin:
      data = json.load(fin)
      labels = [x['label'] for x in data]
  elif DB.endswith('.txt'): #list of images
    labels = helper().read_list(DB,',', False)[0]
  else:
    lmdb_ = lmdbs(DB)
    labels = lmdb_.get_label_list()
  out = {}
  out['num_classes']= len(set(labels))
  out['num_samples'] = len(labels)
  if out['num_samples']%out['num_classes']!=0:
    print 'We got an unbalance lmdb having {} samples of {} classes'.format(\
      out['num_samples'],out['num_classes'])
  out['samples_per_class'] = out['num_samples']/out['num_classes']
  return out

def py2mat(pydict,out_mat):
  """
  save python object (must be a dictionary) to .mat file
  """
  sio.savemat(out_mat,pydict)

def read_mean(mean_file):
  """
  return mean value whether it is pixel mean, scalar or image mean
  """
  if mean_file==0:
      img_mean = 0
  elif type(mean_file) is list:
    img_mean = np.array(mean_file, dtype = np.float32)
  elif mean_file[-4:] == '.mat':
    img_mean = mat2py_mean(mean_file)
  elif mean_file[-12:] == '.binaryproto':
    img_mean = biproto2py(mean_file).squeeze()
  else:
    assert 0, 'Invalid format for mean_file {}'.format(mean_file)
    
  return img_mean

def read_db(source, param = None, verbose = True):
  """read different db format"""
  if source.endswith('.json'):
    if verbose: print('json db detected: {}'.format(source))
    return jsondb(source, thin = False)
  elif os.path.exists(os.path.join(source, 'data.mdb')):
    if verbose: print('lmdb db detected: {}'.format(source))
    return lmdbs(source)
  elif os.path.isdir(source) and param is not None:
    if verbose: print('raw image db detected: {}'.format(source))
    return imgdb(source, param)
  elif source.endswith('.pkl'):
    if verbose: print('svgdb detected: {}'.format(source))
    return svgs(source)
  else:
    assert 0, 'Opps. Invalid db format {}'.format(source)

def extract_cnn_feat(net_params, DB, OUT, layer=0, enrich = False, verbose = True):
  """
  extract features from CNN

  DB: lmdb data you want to extract feature
  net_params: dictionary with keys "DEPLOY_PRO","data_mean",
    "WEIGHTS","scale", batch_size
  OUT: save output in mat file
  layer: 0 for last layer, -1: one before the last layer, -2: ...
  """
  assert layer <=0, 'layer should be a negative integer'
  DEPLOY_PRO = net_params['DEPLOY_PRO']
  WEIGHTS    = net_params['WEIGHTS']
  scale      = net_params['scale']
  data_mean  = net_params['data_mean']
  batch_size = net_params['batch_size']
  img_lst    = net_params['lst']
  
  net = caffe.Net(DEPLOY_PRO,caffe.TEST,weights=WEIGHTS)
  
  db = read_db(DB, img_lst)
  label_lst = db.get_label_list()
  if type(label_lst) is np.ndarray:
    labels = label_lst
    labels2 = 'not exist'
  elif type(label_lst) is tuple:
    print('Two label types detected.')
    labels = label_lst[0]
    labels2 = label_lst[1]
  
  NIMGS = labels.size
  
  img_mean = read_mean(data_mean)
  
  if verbose:
    print 'Extracting cnn feats...'
    print '  Database: {}'.format(DB)
    print '  NIMGS: {}'.format(NIMGS)
    print '  Batchsize: {}'.format(batch_size)
    print '  Model def: {}\n  Weights: {}'.format(DEPLOY_PRO,WEIGHTS)
    print '  Mean image: {}'.format(data_mean)
    start_t = time.time()
  
  inblob = net.inputs[0]
  in_dim = net.blobs[inblob].data.shape[1:]
  prep = ImageAugment(mean=img_mean,shape=in_dim[-2:],scale=scale)
  
  feat_l = net.blobs.keys()[layer-1]
  out_dim = net.blobs[feat_l].data.shape[1]
  feats = np.zeros((NIMGS,out_dim),dtype=np.float32)
  bar = progressbar(NIMGS)
  for i in xrange(0,NIMGS,batch_size):
    batch = range(i,min(i+batch_size,NIMGS))
    bar.show_progress(batch[-1])
    #print 'batch #{}-{}'.format(batch[0],batch[-1])
    new_shape = (len(batch),) + in_dim
    net.blobs[inblob].reshape(*new_shape)
    
    chunk = db.get_data(batch)
    chunk_aug = prep.augment_deploy(chunk, enrich)
    
    new_shape = (chunk_aug.shape[0],) + in_dim
    net.blobs[inblob].reshape(*new_shape)
    net.blobs[inblob].data[...] = chunk_aug
    _ = net.forward()
    out=net.blobs[feat_l].data.squeeze()
    feats[batch] = out.reshape((len(batch),-1,out_dim)).mean(axis=1)
  
  if OUT.endswith('.mat'):
    py2mat(dict(feats=feats,labels=labels, labels2 = labels2),OUT)
  elif OUT.endswith('.npz'):
    np.savez(OUT,feats = feats, labels=labels, labels2 = labels2)
  else: #assume it is pickle
    helps = helper()
    helps.save(OUT,feats = feats, labels = labels, labels2 = labels2)
  net = None
  if verbose:
    end_t = time.time()
    print '\nSave features to {}.'.format(OUT)
    print 'Time: {}\n'.format(timedelta(seconds = int(end_t-start_t)))

def caffe_run_solver(params, SOLVER_PRO, RESTORE = ''):
  """run caffe solver and keep track of the loss
  currently used with triplet hard selection only
  """
  OUT = params['out']
  postfix = params['postfix']
  nepoches = params['epoches']
  nimgs_train = params['nimgs_train']
  if 'phase' in params:
    phase = params['phase']
  else:
    phase = 0
  if 'snapshot_inter' in params:
    snapshot_inter = int(params['snapshot_inter'])
  else:
    snapshot_inter = 500
  
  solver = None
  solver = caffe.SGDSolver(SOLVER_PRO)
  
  sn_iter = 0
  if RESTORE.endswith('.solverstate'):
    solver.restore(RESTORE)
    sn_iter = solver.iter
    print('Resume training from {}'.format(RESTORE))
  if RESTORE.endswith('.caffemodel'):
    print('Weights loaded from {}'.format(RESTORE))
    solver.net.copy_from(RESTORE)
    
  start_t = time.time()
  batch_size = solver.net.blobs['data_p'].data.shape[0]
  niter_p_epoch = int(nimgs_train/batch_size)   #this should be an integer
  print 'number of epoches: {}, batchsize: {}, niter per epoch: {}'.format(
        nepoches, batch_size, niter_p_epoch)
  sys.stdout.flush()
  train_loss = np.zeros(nepoches)
  nreports = int(params['nreports']) #report loss every #epoches
  estart = int(solver.iter/niter_p_epoch)  #epoch number to resume from
  for it in range(estart,nepoches):
      for i in range(niter_p_epoch):
        solver.step(1)  # SGD by Caffe
        train_loss[it] += solver.net.blobs['loss'].data / niter_p_epoch
      
      if (it+1)%snapshot_inter == 0:
        solver.snapshot()
        sn_iter = (it+1)*niter_p_epoch
      
      if (it+1)%nreports ==0: #print results
        end_t = time.time()
        wall_t = int(end_t - start_t)
        print 'Epoch #{}, loss: {}, time passed: {}'.format(
              it,train_loss[it],timedelta(seconds = wall_t))
        sys.stdout.flush()
        plt.plot(np.arange(it+1),train_loss[0:it+1])
        plt.xlabel('# epoches')
        plt.ylabel('loss')
        plt.title('triplet training loss')
        plt.savefig(os.path.join(OUT,'train_loss{}.png'.format(postfix)), bbox_inches='tight')
        plt.show()
        plt.clf()
        np.savez(os.path.join(OUT,'train_loss{}.npz'.format(postfix)),loss=train_loss,
                 it=it,phase=phase,sn_iter = sn_iter)
  solver.snapshot()
  end_t = time.time()
  print 'Done. Training time: {}.'.format(timedelta(seconds=int(end_t-start_t)))


# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:01:37 2016
This module contains several classes that may be useful in caffe
@author: tb00083
"""

import lmdb
import time, os, sys
import numpy as np
import pandas as pd
import caffe
import PIL.Image
from StringIO import StringIO
import h5py as h5
from helper import helper,Timer
from bwmorph import bwmorph_thin
#from svg.SVGProcessor import SVGProcessor #for svgs class
import copy
from multiprocessing.pool import ThreadPool
from PIL import Image, ImageDraw
from bwmorph import bwmorph_thin
from scipy import misc
from helper import helper
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pickle, json
from pycluster import pycluster
from divergence import divergence_metric
from manifold_alignment import manifold, matrixWarp
class lmdbpy(object):
  """
  Manipulate LMDB data in caffe
  This class can read/write lmdb, convert matlab imdb to lmdb
  Used mainly when converting matlab imdb to the standard lmdb in caffe.
  To manipulate lmdb data in a caffe layer, use lmdbs instead
  """
  def create_dummy(lmdb_size, num_classes, out_path):
    """ create a dummy lmdb given 4-D size, number of classes and output path
    """
    # delete any previous db files
    try:
        os.remove(out_path + "/data.mdb")
        os.remove(out_path + "/lock.mdb")
        time.sleep(1)
    except OSError:
        pass
    
    start_t = time.clock()
    N = lmdb_size[0]    #number of instances
    # Let's pretend this is interesting data
    X = np.random.rand(N,lmdb_size[1],lmdb_size[2],lmdb_size[3])
    X = np.array(X*255,dtype=np.uint8)
    y = np.int64(num_classes*np.random.rand(N))

    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10
    
    env = lmdb.open(out_path,map_size=map_size)
    txn = env.begin(write=True)
    buffered_in_bytes = 0
    for i in range(N):
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = X.shape[1]
      datum.height = X.shape[2]
      datum.width = X.shape[3]
      datum.data = X[i].tobytes() # or .tostring() if numpy < 1.9
      datum.label = int(y[i])
      datum_str = datum.SerializeToString()
      str_id = '{:08}'.format(i)
      
      # The encode is only essential in Python 3
      txn.put(str_id.encode('ascii'), datum_str)
      
      buffered_in_bytes += sys.getsizeof(datum_str)
      
      # flush and generate new transactions if we have more 100mb pending
      if buffered_in_bytes > 100e6:
        buffered_in_bytes = 0
        txn.commit()
        env.sync()
        txn = env.begin(write=True)
    
    # ensure final output was written
    txn.commit()
    env.sync()
    
    # close databases
    env.close()
    end_t = time.clock()
    print "Complete after %d seconds" %(end_t - start_t)
    
  def setup(self,out_path):
    """setup env to write lmdb part by part
    """
    print 'Writing to ' + out_path
    try:
        os.remove(out_path + "/data.mdb")
        os.remove(out_path + "/lock.mdb")
        time.sleep(1)
    except OSError:
        pass
    
    self.timer = Timer()
    self.env = lmdb.open(out_path,map_size=1e12)
    self.counter = 0
    
  def write_part(self, data, label):
    """write npy image data and label to lmdb
    image data must be 4-D in format num x channels x height x width
    """
    assert data.ndim == 4, 'data must be 4-D format num x channels x height x width'
    if np.max(data[...]) > 1.0:
      data = data.astype(np.uint8)
    else:
      data = np.array(data*255, dtype=np.uint8)
    N = np.size(data,axis=0)
    txn = self.env.begin(write=True)
    buffered_in_bytes = 0
    for i in range(N):
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = data.shape[1]
      datum.height = data.shape[2]
      datum.width = data.shape[3]
      datum.data = data[i].tobytes() # or .tostring() if numpy < 1.9
      datum.label = int(label[i])
      datum_str = datum.SerializeToString()
      str_id = '{:08}'.format(i+self.counter)
      
      # The encode is only essential in Python 3
      txn.put(str_id.encode('ascii'), datum_str)
      
      buffered_in_bytes += sys.getsizeof(datum_str)
      
      # flush and generate new transactions if we have more 100mb pending
      if buffered_in_bytes > 100e6:
        buffered_in_bytes = 0
        txn.commit()
        self.env.sync()
        txn = self.env.begin(write=True)
    
    # ensure final output was written
    txn.commit()
    self.env.sync()
    print('Writing #{}-{}: {}'.format(self.counter,self.counter+N, self.timer.time()))
    self.counter += N
    
  def close(self):
    """house cleaning after write_part"""
    self.env.close()
    self.env = None
    self.counter = 0
    print('Completed after {}'.format(self.timer.time()))

  def write(self, data, label, out_path):
    """write npy image data and label to lmdb
    image data must be 4-D in format num x channels x height x width
    """
    print 'Writing to ' + out_path
    assert data.ndim == 4, 'data must be 4-D format num x channels x height x width'
    try:
        os.remove(out_path + "/data.mdb")
        os.remove(out_path + "/lock.mdb")
        time.sleep(1)
    except OSError:
        pass
    
    start_t = time.time()
    if np.max(data[...]) > 1.0:
      data = data.astype(np.uint8)
    else:
      data = np.array(data*255, dtype=np.uint8)
    
    N = np.size(data,axis=0)
    map_size = data.nbytes *2
    env = lmdb.open(out_path,map_size=map_size)
    txn = env.begin(write=True)
    buffered_in_bytes = 0
    for i in range(N):
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = data.shape[1]
      datum.height = data.shape[2]
      datum.width = data.shape[3]
      datum.data = data[i].tobytes() # or .tostring() if numpy < 1.9
      datum.label = int(label[i])
      datum_str = datum.SerializeToString()
      str_id = '{:08}'.format(i)
      
      # The encode is only essential in Python 3
      txn.put(str_id.encode('ascii'), datum_str)
      
      buffered_in_bytes += sys.getsizeof(datum_str)
      
      # flush and generate new transactions if we have more 100mb pending
      if buffered_in_bytes > 100e6:
        buffered_in_bytes = 0
        txn.commit()
        env.sync()
        txn = env.begin(write=True)
    
    # ensure final output was written
    txn.commit()
    env.sync()
    
    # close databases
    env.close()
    end_t = time.time()
    print "Complete after %d seconds" %int(end_t - start_t)
  
  
  def mat2lmdb(self, mat_file, lmdb_path):
    """
    convert imdb mat to lmdb and save it in caffe
    """
    mat_ = mat2py()
    data, label = mat_.read_imdb(mat_file)
    if len(data.shape) == 4:
      print 'loaded data has format HxWxCxN, assume they are photo RGB images'
      print 'Perform preprocess: BGR swap'
      data = data[:,:,::-1,:]   #convert RGB to BGR
      data = data.transpose(3,2,0,1) #change HxWxCxN to NxCxHxW
    else:
      print 'loaded data has format HxWxN, assume grayscale image'
      data = data.transpose(2,0,1)
      data = data[:,None, :, :]

    self.write(data, label, lmdb_path)
  
  
  def read(self, in_path):
    """
    read lmdb, return image data and label
    """
    print 'Reading ' + in_path
    env = lmdb.open(in_path, readonly=True)
    N = env.stat()['entries']
    txn = env.begin()
    for i in range(N):
      str_id = '{:08}'.format(i)
      raw_datum = txn.get(str_id)
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(raw_datum)
      feature = caffe.io.datum_to_array(datum)
      if i==0:
        data = np.zeros((N,feature.shape[0],feature.shape[1],
                         feature.shape[2]),dtype=np.uint8)
        label = np.zeros(N,dtype=np.int64)
      data[i] = feature
      label[i] = datum.label
    env.close()
    return data, label
    
  def read_encoded(self, in_path):
    """
    read lmdb of encoded data, e.g. PNG or JPG in nvidia digit
    """
    env = lmdb.open(in_path, readonly=True)
    N = env.stat()['entries']
    txn = env.begin()
    cursor = txn.cursor()
    count = 0
    for key, value in cursor:
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(value)
      s = StringIO()
      s.write(datum.data)
      s.seek(0)
      img = PIL.Image.open(s)
      if count == 0:
        data = np.zeros((N,datum.channels,datum.height, datum.width),dtype=np.uint8)
        label = np.zeros(N,dtype=np.int64)
      data[count] = np.array(img)
      label[count] = datum.label
      count += 1
      
    env.close()
    return data, label

###############################################################################
class lmdbs(object):
  """
  proper lmdb class to read lmdb data
  Recommend to be used in caffe data layer
  """
  def __init__(self,lmdb_path):
    if not os.path.isdir(lmdb_path):
      assert 0,'Invalid lmdb {}\n'.format(lmdb_path)
    self.env = lmdb.open(lmdb_path, readonly=True)
    self.NumDatum = self.env.stat()['entries']
    self.txn = self.env.begin()
  def __del__(self):
    self.env.close()
    
  def get_label_list(self):
    """get the list of labels in the lmdb"""
    labels = np.zeros(self.NumDatum,dtype=np.int64)
    for i in range(self.NumDatum):
      str_id = '{:08}'.format(i)
      raw_datum = self.txn.get(str_id)
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(raw_datum)
      labels[i] = datum.label
    return labels
  
  def get_datum(self,ind):
    """get datum in lmdb given its index"""
    str_id = '{:08}'.format(ind)
    raw_datum = self.txn.get(str_id)
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)
    img = caffe.io.datum_to_array(datum)
    return img
  
  def get_data(self,inds):
    """
    get array of data given its indices
    """
    inds = np.array(inds)
    for i in range(inds.size):
      img = self.get_datum(inds[i])
      if i==0:
        data = np.zeros(inds.shape+img.shape,dtype=img.dtype)
      data[i] = img
    return data
  
  def extract(self, lmdb_path,ids=0):
    """extract particular image(s) from lmdb (of non-encoded data)
    Deprecated. Use get_data instead
    """
    ids = np.array(ids)
    env = lmdb.open(lmdb_path, readonly=True)
    txn = env.begin()
    for i in range(ids.size):
      str_id = '{:08}'.format(ids[i])
      raw_datum = txn.get(str_id)
      datum = caffe.proto.caffe_pb2.Datum()
      datum.ParseFromString(raw_datum)
      feature = caffe.io.datum_to_array(datum)
      if i==0:
        X = np.zeros((len(ids),feature.shape[0],feature.shape[1],
                      feature.shape[2]),dtype=np.uint8)
        Y = np.zeros(len(ids),dtype=np.int64)
      X[i] = feature
      Y[i] = datum.label
    env.close()
    return (X,Y)
    
  def get_info(self):
    """for now, it just return number of channels in an image of the dataset"""
    img = self.get_datum(0)
    return dict(nchannels = img.shape[0])
  
  def get_image_deprocess(self,ind):
    """
    This only works with lmdb with color images
    get color image and deprocess
    """
    img = self.get_datum(ind).copy()
    if img.shape[0]==3: #color image
      img = img[::-1]    #BGR to RGB
      img = img.transpose(1,2,0) #CxHxW to HxWxC
      img = np.round(img)
      img = np.require(img,dtype=np.uint8)
    else:
      img = np.squeeze(img)
    return img
###############################################################################
class jsondb(object):
  """lmdb class for quickdraw sketches"""
  def __init__(self,lmdb_path, thin=False):
    with open(lmdb_path,'r') as fin:
      self.data = json.load(fin)
    self.thin = thin
  def __del__(self):
    self.data = None
    self.thin = None
  def get_info(self):
    """for now, it just return number of channels in an image of the dataset"""
    print('Json sketch: always converted to RGB')
    return dict(nchannels = 3)
  def get_image_deprocess(self,ind):
    """return a color image"""
    return self.sketch_preprocess(self.data[ind]['image'])
  def get_label_list(self):
    """get the list of labels in the lmdb"""
    labels = np.array([skt['label'] for skt in self.data], dtype= np.int64)
    return labels
  def get_datum(self,ind, color = True):
    """get datum in form CxHxW"""
    img = self.sketch_preprocess(self.data[ind]['image'], color = color)
    img = img.transpose(2,0,1) if color else img[None,...]
    return img
    
  def get_data(self,inds):
    """
    get array of data given its indices
    size BxCxHxW
    """
    inds = np.array(inds)
    for i in range(inds.size):
      img = self.get_datum(inds[i], color = True)
      if i==0:
        data = np.zeros(inds.shape+img.shape,dtype=img.dtype)
      data[i] = img
    return data
  
  def draw_image(self, strokes,background_pixel=255):
    im = Image.new('L', (256, 256), background_pixel)
    draw = ImageDraw.Draw(im)
    for stroke in strokes:
      draw.line(zip(stroke[0],stroke[1]), fill=255 - background_pixel,width=2)
    return np.array(im)
  
  def sketch_preprocess(self, img, color = True):
    """centering, skeletonise, resize
    img: json image"""
    max_dim = 200.0
    shape = [256,256]
    img = self.draw_image(img,background_pixel=0)
    nz = np.nonzero(img)
    if nz[0].size:
      ymin = max(0,nz[0].min() - 1)
      ymax = min(img.shape[0],nz[0].max()+2)
      xmin = max(0,nz[1].min()-1)
      xmax = min(img.shape[1],nz[1].max()+2)
      img = img[ymin:ymax,xmin:xmax].astype(np.float32)
    else:
      print('Opps! Blank query after pre-process. Make sure u use black colour to draw.')
    #resize to max_dim
    zf = float(max_dim)/max(img.shape)
    img = misc.imresize(img,zf)  #this automatically convert to [0,255] range
    if self.thin:
      #skeletonise sketch
      img = img > 50
      img = bwmorph_thin(img)
      img = np.uint8(255*(1-img))
    else:
      img = 255 - np.uint8(img)
    #Image.fromarray(np.uint8(img)).save('test.jpg')
    #padding
    p = (shape - np.array(img.shape))/2
    img = np.pad(img,((p[0],shape[0]-p[0]-img.shape[0]),(p[1],shape[1]-p[1]-img.shape[1])),
                 'constant',constant_values = 255)
    if color:
      img = img[...,None]
      img = np.repeat(img,3, axis=2)
    
    return img #output shape HxWx3 uint8 image
###############################################################################
class imgdb(object):
  """db class for raw images"""
  def __init__(self,src_dir, src_lst):
    self.src_dir = src_dir
    src_lst = pd.read_csv(src_lst)#helper().read_list(src_lst,',')
    self.label_lst = src_lst['class_label'].values#np.array(src_lst[0]).astype(np.int64)
    self.img_paths = src_lst['path'].tolist()#src_lst[-1]
    #self.label_lst2 = None if len(src_lst)==2 else np.array(src_lst[1]).astype(np.int64)
    self.label_lst2 = None if 'midgrain_label' not in src_lst.keys() else src_lst['midgrain_label'].values
    
  def __del__(self):
    self.src_dir = self.label_lst = self.img_paths = self.label_lst2 = None
  
  def get_info(self):
    """for now, it just return number of channels in an image of the dataset"""
    return dict(nchannels = 3)
  def get_image_deprocess(self,ind):
    """return a color image"""
    img = caffe.io.load_image(os.path.join(self.src_dir, self.img_paths[ind]))*255
    return np.uint8(img)
  
  def get_label_list(self):
    """get the list of labels in the lmdb"""
    if self.label_lst2 is not None:
      return self.label_lst, self.label_lst2
    else:
      return self.label_lst
  def get_datum(self,ind):
    """get datum in form CxHxW"""
    img = caffe.io.load_image(os.path.join(self.src_dir, self.img_paths[ind]))*255 #RGB in [0,255] float32
    #img = np.array(Image.fromarray(np.uint8(img)).resize((224,224), Image.BILINEAR), dtype=np.float32)
    return img.transpose(2,0,1)[::-1]
    
  def get_data(self,inds):
    """
    get array of data given its indices
    size BxCxHxW
    """
    inds = np.array(inds)
    for i in range(inds.size):
      img = self.get_datum(inds[i])
      if i==0:
        data = np.zeros(inds.shape+img.shape,dtype=img.dtype)
      data[i] = img
    return data

###############################################################################
class svgs(object):
  """
  a class to read svg data
  Recommend to be used in caffe data layer
  """
  def __init__(self,lmdb_path):
    if not os.path.isfile(lmdb_path):
      assert 0,'Invalid .pkl lmdb {}\n'.format(lmdb_path)
    
    #note: we need to define svg class prior to loading it
    helps = helper()
    self.svgdb = helps.load(lmdb_path)
    
  def get_label_list(self):
    """get the list of labels in the lmdb"""
    return self.svgdb['labels'].astype(np.int64)
  
  def get_datum(self,ind):
    """get datum in lmdb given its index
    Preprocess:  randomly remove strokes and skeletonise
    """
    #svg = copy.deepcopy(self.svgdb['data'][ind])
    svg = self.svgdb['data'][ind]
#==============================================================================
#     svg.randomly_occlude_strokes(n=4)
#     img = svg.get_numpy_array(size=256)
#==============================================================================
    
    img = svg.remove_strokes_and_convert(n=4,size=256)
    #convert to grayscale binary
    img = (0.299*img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]) < 254
    img = bwmorph_thin(img)
    img = np.float32(255*(1-img))
    #svg = None
    return img
  
  def get_data(self,inds):
    """
    get array of data given its indices
    """
    inds = np.array(inds)
    for i in range(inds.size):
      img = self.get_datum(inds[i])
      if i==0:
        data = np.zeros(inds.shape+img.shape,dtype=img.dtype)
      data[i] = img
    return data
  
  def get_num_strokes(self):
    """
    special function to get list of stroke number of the svg database
    """
    out = [svg.get_num_strokes() for svg in self.svgdb['data']]
    return out

###############################################################################
class mat2py(object):
  """Manipulate matlab v7.3 (.mat) file
  Most of the case you need to know in advance the structure of saved variables
  in the .mat file.
  there is an odd problem that prevents caffe from working if data is read 
   from mat file. So this class is deprecated until the problem is solved
  """
  
  def read_imdb(self,mat_file):
    """
    mat file variables:
    images
      data          #
      labels        #
      data_mean
    meta
      classes
    """
    assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
    f = h5.File(mat_file)
    data = np.array(f['images']['data'])
    labels = np.array(f['images']['labels'])
    #img_mean = np.array(f['images']['data_mean'])
    #matlab store data column wise so we need to transpose it
    return data.transpose(), labels#, img_mean.transpose()
    
  def read_mean(self,mat_file):
    """
    mat file variables:
    data_mean       #
    """
    assert os.path.isfile(mat_file), 'Mat file {} not exist.'.format(mat_file)
    f = h5.File(mat_file)
    data_mean = np.array(f['data_mean'])
    return data_mean.transpose()

class matlab(object):
  """designed to replace mat2py
  manipulate .mat file given the list of the variables
  """
  def __init__(self, mat_file):
    self.f = h5.File(mat_file)
  def load(self, var):
    pass

###############################################################################
class sbir(object):
  """class for image retrieval
  """
  def __init__(self,params):
    if 'data' in params:
      self.data = np.load(params['data'])
    if 'queries' in params:
      self.queries = np.load(params['queries'])
    #self.pool = ThreadPool()
  
  def load_database(self, data, labels=None):
    if isinstance(data,basestring): #path to database
      assert data.endswith('.npz'), 'Database must be a .npz file'
      self.data = np.load(data)
    elif isinstance(data,np.ndarray): #numpy array NxD
      self.data = {}
      self.data['feats'] = data
      if labels is not None:
        assert data.shape[0] == labels.size, 'Error: dimension mismatch bw data and label'
        self.data['labels'] = labels
    else:
      print('Error: data is neither .npz database or numpy array. Exit now.')
      sys.exit()
  
  def dist_L2(self, a_query):
    """
    Eucludean distance between a (single) query & all features in database
    used in pdist
    """
    return np.sum((a_query-self.data['feats'])**2,axis=1)
    
  def pdist(self, query_feat):
    """
    Compute distance (L2) between queries and data features
    query_feat: NxD numpy array with N features of D dimensions
    """
    if len(query_feat.shape) <2:   #single query
      query_feat = query_feat[None,...]
    res = [np.sum((query_feat[i]-self.data['feats'])**2,axis=1) for i in range(query_feat.shape[0])]
#==============================================================================
#     threads = [self.pool.apply_async(self.dist_L2,(query_feat[i],)) for \
#                 i in range(query_feat.shape[0])]
#     res = [thread.get() for thread in threads]
#==============================================================================
    res = np.array(res)
    return res
  
  def retrieve(self,queries):
    """retrieve: return the indices of the data points in relevant order
    """
    if isinstance(queries,basestring):
      query_feat = np.load(queries)['feats']
    elif isinstance(queries, np.ndarray):
      query_feat = queries
    else:
      print('Error: queries must be a .npz file or ndarray')
      sys.exit()
    
    res = self.pdist(query_feat)
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
  
  def retrieve2file_hardsel(self,queries, out_file, num=0, qsf = 2.0):
    """same as retrieve2file but highly customised for hard triplet selection
    Hard negative: exclude images of the same cat in the retrieving process
    Hard positive: include only images of the same cat and choose the farthest
    queries: must be a .npz file containing feats & labels, e.g. output of extract_cnn_feat()
    num: number of returned results
    """
    queries_ = np.load(queries)
    query_feat = queries_['feats'] * qsf
#==============================================================================
#     """old code"""
#     query_label = queries_['labels']
#     # compute L2 distance
#     dist = [np.sum((query_feat[i]-self.data['feats'])**2,axis=1) for i in range(query_feat.shape[0])]
#     # returned image id
#     ids = [dist[i].argsort() for i in range(len(dist))]
#     #labels of the returned images
#     ret_labels = [self.data['labels'][ids[i]] for i in range(len(ids))]
#     # relevant
#     rel = [ret_labels[i] == query_label[i] for i in range(len(ret_labels))]
#     #include/exclude the relevant
#     pos = [ids[i][rel[i]] for i in range(len(ids))]
#     pos = np.fliplr(np.array(pos))   #hard positive
#     neg = [ids[i][~rel[i]] for i in range(len(ids))]
#     neg = np.array(neg)              #hard negative
#==============================================================================
    """new code"""
    query_label = queries_['labels'][...,None]
    ids = self.retrieve(query_feat)
    ret_labels = self.data['labels'][ids]
    # relevant
    rel = ret_labels == query_label
    #include/exclude the relevant in hard pos/neg selection
    pos = ids[rel].reshape([rel.shape[0],-1])
    pos = np.fliplr(pos)                       #hard positive
    neg = ids[~rel].reshape([rel.shape[0],-1]) #hard negative
    
    if num > 0 and num < self.data['feats'].shape[0]:
      pos = pos[:,0:num]
      neg = neg[:,0:num]
    if out_file.endswith('.npz'):
      np.savez(out_file, pos = pos, neg = neg)
    
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP
    
  def compute_mAP(self,queries,qsf = 1.0):
    """
    compute mAP given queries in .npz format
    """
    assert isinstance(queries,basestring) and queries.endswith('.npz'),'Opps! Input must be a .npz file'
    tmp = np.load(queries)
    query_feat = tmp['feats'] * qsf
    query_label = tmp['labels'][...,None]
    ids = self.retrieve(query_feat)
    ret_labels = self.data['labels'][ids]
    # relevant
    rel = ret_labels == query_label
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP

class sbir_hardsel(object):
  """sbir for hard selection
  similar to sbir, __init is a bit different
  """
  def __init__(self,params, qsf = 1.0):
    self.data_src = self.data_q = self.label_src = self.label_q = None
    self.qsf = qsf
    if 'data_src' in params:
      self.data_src = params['data_src']
    if 'data_q' in params:
      self.data_q = params['data_q']
      if len(self.data_q.shape) < 2: #single query
        self.data_q = self.data_q[None,...]
    if 'label_src' in params:
      self.label_src = np.array(params['label_src'])
    if 'label_q' in params:
      self.label_q = np.array(params['label_q'])
      if len(self.label_q.shape) < 2:
        self.label_q = self.label_q[...,None]
  
  def update(self,params):
    if 'data_src' in params:
      self.data_src = params['data_src']
    if 'data_q' in params:
      self.data_q = params['data_q']
      if len(self.data_q.shape) < 2: #single query
        self.data_q = self.data_q[None,...]
    if 'label_src' in params:
      self.label_src = np.array(params['label_src'])
    if 'label_q' in params:
      self.label_q = np.array(params['label_q'])
      if len(self.label_q.shape) < 2:
        self.label_q = self.label_q[...,None]
        
  def pdist(self):
    """
    Compute distance (L2) between queries and data features
    query_feat: NxD numpy array with N features of D dimensions
    """
    
    res = [np.sum((self.data_q[i]*self.qsf-self.data_src)**2,axis=1) for i in range(self.data_q.shape[0])]
    res = np.array(res)
    return res
  
  def retrieve(self):
    """retrieve: return the indices of the data points in relevant order
    """
    res = self.pdist()
    return res.argsort()
  def retrieve2file(self, out_file, numn=0, nump=0):
    """highly customised for hardsel"""
    ids = self.retrieve()
    ret_labels = self.label_src[ids]
    rel = ret_labels == self.label_q
    #include/exclude the relevant in hard pos/neg selection
    pos = ids[rel].reshape([rel.shape[0],-1])
    pos = np.fliplr(pos)                       #hard positive
    neg = ids[~rel].reshape([rel.shape[0],-1]) #hard negative
    
    if nump > 0 and nump < pos.shape[1]:
      pos = pos[:,0:nump]
    if numn > 0 and numn < neg.shape[1]:
      neg = neg[:,0:numn]
    if out_file.endswith('.npz'):
      np.savez(out_file, pos = pos, neg = neg)
    
    P = np.cumsum(rel,axis=1) / np.arange(1,rel.shape[1]+1,dtype=np.float32)[None,...]
    AP = np.sum(P*rel,axis=1) / (rel.sum(axis=1) + np.finfo(np.float32).eps)
    mAP = AP.mean()
    return mAP
    
class decay(object):
  def __init__(self,params):
    if 'k' in params:
      self.k = params['k']
    if 'max_iter' in params:
      self.max_iter = params['max_iter']
    if 'bias' in params:
      self.bias = params['bias']
    if 'power' in params:
      self.power = params['power']
  def poly_decay(self,x):
    return int(self.k * (1 - float(x)/self.max_iter)**self.power + self.bias)
    
class MagnetFinegrain(object):
  def __init__(self,dim, ncats, nskts, nimgs):
    self.skt_feats = np.zeros((nskts, dim), dtype= np.float32)
    self.img_feats = np.zeros((nimgs, dim), dtype = np.float32)
    self.skt_labels = np.zeros(nskts, dtype = np.int64)
    self.img_labels = np.zeros(nimgs, dtype = np.int64)
    self.dim = dim
    self.ncats = ncats
    self.W = None
  def reg(self, skt_id, skt_feat, skt_label, img_id, img_feat, img_label):
    """register skt and img features"""
    if skt_id is not None:
      self.skt_feats[skt_id,...]= skt_feat
      self.skt_labels[skt_id] = copy.copy(skt_label)
    if img_id is not None:
      self.img_feats[img_id,...] = img_feat
      self.img_labels[img_id] = copy.copy(img_label)
    
  def learn_warp_matrix(self, pca = None, lamda = 5e-1, niter = 100, lr=0.01,tol=1e-5, method='sgd'):
    """
    if pca model is provided, do PCA projection before learning warp matrix
    """
    nspi = self.skt_feats.shape[0]/self.img_feats.shape[0]
    if pca is not None:
      newdim = pca.n_components_
      skt_feats = pca.transform(self.skt_feats)
      img_feats = pca.transform(self.img_feats)
    else:
      newdim = self.dim
      skt_feats = self.skt_feats
      img_feats = self.img_feats
    if method == 'sgd':
      I = np.eye(newdim,dtype=np.float32)#np.r_[np.eye(newdim), np.zeros((1,newdim))].astype(np.float32)#
      self.W =np.zeros((self.ncats, newdim+1, newdim), dtype=np.float32)
      #totalL = []
      for cat in range(self.ncats):
        img_id = self.img_labels == cat
        skt_id = self.skt_labels == cat
        N = skt_id.sum()
        img_feats_ = img_feats[img_id]
        X = skt_feats[skt_id]#np.c_[skt_feats[skt_id], np.ones((N,1), np.float32)]#
        Y = np.repeat(img_feats_[:,None,:], nspi, axis=1).reshape((-1,newdim))
        
        W = np.random.normal(0,0.01,(newdim,newdim)).astype(np.float32)
        b = np.zeros((1,newdim),np.float32)
        #L = np.zeros((niter,2),np.float32)
        prevL = 0
        for i in range(niter):
          db = X.dot(W) + b - Y
          L_ = 0.5/N*np.sum(db.dot(db.T))+ 0.5*lamda*np.sum((W-I)**2)
          #L[i,...] = np.array((0.5/N*np.sum(db.dot(db.T)),0.5*lamda*np.sum((W-I)**2)))
          dLdW = X.T.dot(db)/N + lamda*(W-I)
          W = W - lr*dLdW
          b = b - lr/N*db.sum(axis=0)
          if i >0 and np.abs(L_ - prevL) < tol: break
          prevL = L_
        self.W[cat,...] = np.r_[W,b]
      #totalL.append(L)
    else:
      I = np.r_[np.eye(newdim), np.zeros((1,newdim))].astype(np.float32)#
      self.W =np.zeros((self.ncats, newdim+1, newdim), dtype=np.float32)
      for cat in range(self.ncats):
        img_id = self.img_labels == cat
        skt_id = self.skt_labels == cat
        N = skt_id.sum()
        img_feats_ = img_feats[img_id]
        X = np.c_[skt_feats[skt_id], np.ones((N,1), np.float32)]#skt_feats[skt_id]#
        Y = np.repeat(img_feats_[:,None,:], nspi, axis=1).reshape((-1,newdim))
        try:
          #self.W[cat,...] = np.linalg.inv(X.T.dot(X)+N*lamda).dot(X.T.dot(Y)+N*lamda*I)
          X = X[:,:-1]
          self.W[cat,...]=np.r_[np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y)), np.zeros((1, newdim))]
        except:
          raise Exception('Matrix not invertable for class #{}'.format(cat))
    return copy.copy(self.W)
      

class MagnetCluster(object):
  def __init__(self,dim, ncats, nskts, nimgs, nskt_c, nimg_c, skt_thres = 10, img_thres=10):
    self.skt_feats = np.zeros((nskts, dim), dtype= np.float32)
    self.img_feats = np.zeros((nimgs, dim), dtype = np.float32)
    self.skt_labels = np.zeros(nskts, dtype = np.int64)
    self.img_labels = np.zeros(nimgs, dtype = np.int64)
    self.dim = dim
    self.ncats = ncats
    self.nskt_c = nskt_c
    self.nimg_c = nimg_c
    self.skt_thres = skt_thres
    self.img_thres = img_thres
    self.pca = None
    self.counter =0
  def set_impostor(self, nskt_impostor, nimg_impostor):
    """configure number of skt and img impostor clusters"""
    self.nskt_impostor = nskt_impostor
    self.nimg_impostor = nimg_impostor
# =============================================================================
#   def learn_warp_matrix(self):
#     nspi = self.skt_feats.shape[0]/self.img_feats.shape[0]
#     img_rep = np.repeat(self.img_feats[:,None,:],nspi, axis=1).reshape((-1, self.dim))
#     try:
#       W = np.linalg.inv(self.skt_feats.T.dot(self.skt_feats)).dot(self.skt_feats.T).dot(img_rep)
#     except:
#       raise Exception('Matrix not invertable')
#     return W
# =============================================================================
    
  def doPCA(self):
    self.pca = PCA(0.95).fit(np.concatenate((self.skt_feats, self.img_feats)))
    return copy.deepcopy(self.pca)
# =============================================================================
#     allfeat = self.pcamodel.transform(np.concatenate((self.skt_feats, self.img_feats)))
#     self.skt_feats = allfeat[:self.skt_feats.shape[0]]
#     self.img_feats = allfeat[-self.img_feats.shape[0]:]
#     self.pcadim = allfeat.shape[1]
# =============================================================================
      
      
  def reg(self, skt_id, skt_feat, skt_label, img_id, img_feat, img_label):
    """register skt and img features"""
    if skt_id is not None:
      self.skt_feats[skt_id,...]= skt_feat
      self.skt_labels[skt_id] = copy.copy(skt_label)
    if img_id is not None:
      self.img_feats[img_id,...] = img_feat
      self.img_labels[img_id] = copy.copy(img_label)
# =============================================================================
#   def clustering(self):
#     """clustering each domain by category"""
#     self.skt_clusters = dict(cat_centers = np.zeros((self.ncats,self.dim),dtype=np.float32),
#                              clt_centers=[], clt_assigns = [], clt_cat = [],cprob = [],
#                              nn = [], nsi = [], nii = [],
#                              clabel_bs = -1*np.ones(self.skt_labels.size,dtype=np.int64))
#     self.img_clusters = dict(cat_centers=np.zeros((self.ncats, self.dim),dtype=np.float32),
#                              clt_centers = [], clt_assigns = [], clt_cat=[],cprob = [],
#                              clabel_bs = -1*np.ones(self.img_labels.size,dtype=np.int64))
#     skt_ccounter = img_ccounter = 0
#     mycluster = pycluster(ctype='gmm', auto_alg = 'bic')
#     for cat in range(self.ncats):
#       #print cat
#       #sketch
#       tid = np.where(self.skt_labels==cat)[0]
#       self.skt_clusters['cat_centers'][cat,...] = self.skt_feats[tid].mean(axis=0)
#       
#       mycluster.fit(self.skt_feats[tid], k_min=2, k_max=self.nskt_c)
#       tcluster = mycluster.predict(self.skt_feats[tid])
#       csize = []
#       for i in range(self.nskt_c):
#         tassignments = tid[tcluster==i]
#         if len(tassignments) > self.skt_thres:
#           self.skt_clusters['clt_centers'].append(mycluster.cluster_centers_[i].copy())
#           self.skt_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
#           self.skt_clusters['clt_cat'].append(cat)
#           self.skt_clusters['clabel_bs'][tassignments] = skt_ccounter
#           csize.append(len(tassignments))
#           skt_ccounter += 1
#       self.skt_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
#       
#       #image
#       tid = np.where(self.img_labels==cat)[0]
#       self.img_clusters['cat_centers'][cat,...] = self.img_feats[tid].mean(axis=0)
#       #mycluster = KMeans(self.nimg_c, init = 'k-means++').fit(self.img_feats[tid])
#       mycluster = mycluster.fit(self.img_feats[tid], k_min=2, k_max=self.nimg_c)
#       tcluster = mycluster.predict(self.img_feats[tid])
#       for i in range(self.nimg_c):
#         tassignments = tid[tcluster==i]
#         if len(tassignments) > self.img_thres:
#           self.img_clusters['clt_centers'].append(mycluster.cluster_centers_[i].copy())
#           self.img_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
#           self.img_clusters['clt_cat'].append(cat)
#           self.img_clusters['clabel_bs'][tassignments] = img_ccounter #cluster label by sample
#           img_ccounter += 1
#     
#     self.skt_clusters['clt_centers'] = np.array(self.skt_clusters['clt_centers'], dtype=np.float32)
#     self.skt_clusters['clt_cat'] = np.array(self.skt_clusters['clt_cat'], dtype = np.int64)
#     self.img_clusters['clt_centers'] = np.array(self.img_clusters['clt_centers'], dtype=np.float32)
#     self.img_clusters['clt_cat'] = np.array(self.img_clusters['clt_cat'], dtype = np.int64)
#       
#   def kNC_bak(self, W = None):
#     self.clustering()
#     self.skt_clusters['W'] = W
#     """nearest cluster"""
#     nclusters = self.skt_clusters['clt_cat'].size
#     #sketch anchor: find nearest sketch and image impostor clusters
#     self.skt_clusters['nsi'] = np.zeros((nclusters, self.nskt_impostor),dtype=np.int64)
#     self.skt_clusters['nii'] = np.zeros((nclusters, self.nimg_impostor),dtype=np.int64)
#     
#     nbrs_skt = NearestNeighbors(min((self.nskt_c+self.nskt_impostor,nclusters))).fit(self.skt_clusters['clt_centers'])
#     _, inds_skt = nbrs_skt.kneighbors(self.skt_clusters['clt_centers'])
#     
#     nbrs_img = NearestNeighbors(min((self.nimg_c*(1+self.nskt_impostor)+self.nimg_impostor,len(self.img_clusters['clt_centers'])))).fit(self.img_clusters['clt_centers'])
#     _, inds_img = nbrs_img.kneighbors(self.skt_clusters['clt_centers'])
#     for ci in range(nclusters):
#       cat_id = self.skt_clusters['clt_cat'][ci]
#       sel_cid_skt = [c for c in inds_skt[ci] if self.skt_clusters['clt_cat'][c] != cat_id]
#       sel_cid_skt = sel_cid_skt[:self.nskt_impostor]
#       self.skt_clusters['nsi'][ci,...] = np.array(sel_cid_skt, dtype=np.int64)
#       
#       exclude_cat_lst = set([self.skt_clusters['clt_cat'][c] for c in sel_cid_skt] + [cat_id,])
#       sel_cid_img = [c for c in inds_img[ci] if self.img_clusters['clt_cat'][c] not in exclude_cat_lst]
#       self.skt_clusters['nii'][ci,...] = np.array(sel_cid_img[:self.nimg_impostor], dtype=np.int64)
#     
#     #sketch anchor: find nearest img clusters
#     
#     if W is not None:
#       skt_clt_center_trans = self.skt_clusters['clt_centers'].dot(W)
#     self.skt_clusters['nn'] = np.zeros(nclusters, dtype=np.int64)
#     for cat in range(self.ncats):
#       if W is None:#standard PCA
#         skt_pca = PCA().fit(self.skt_feats[self.skt_labels==cat])
#         img_pca = PCA().fit(self.img_feats[self.img_labels==cat])
#         skt_std_inv = np.diag(1.0/np.sqrt(np.where(skt_pca.explained_variance_>0, skt_pca.explained_variance_,1.0)))
#         img_std = np.diag(np.sqrt(np.where(img_pca.explained_variance_>0, img_pca.explained_variance_,1.0)))
#         skt_clt_center_trans = skt_pca.transform(self.skt_clusters['clt_centers']).dot(skt_std_inv).dot(img_std).dot(img_pca.components_) + img_pca.mean_[None,:]#
#         
#       skt_cid = np.argwhere(self.skt_clusters['clt_cat'] == cat).squeeze()
#       img_cid = np.argwhere(self.img_clusters['clt_cat'] == cat)[:,0] #special notation
#       tskt_cluster = skt_clt_center_trans[skt_cid]+np.zeros((1,1))#self.skt_clusters['clt_centers'][skt_cid] - self.skt_clusters['cat_centers'][cat][None,...]#
#       timg_cluster = self.img_clusters['clt_centers'][img_cid]+np.zeros((1,1)) #- self.img_clusters['cat_centers'][cat][None,...]#
#       nbrs = NearestNeighbors(1, algorithm = 'brute').fit(timg_cluster)
#       _, inds = nbrs.kneighbors(tskt_cluster)
#       self.skt_clusters['nn'][skt_cid] = img_cid[inds.squeeze()]
#       
#   def kNC_bak2(self, W = None):
#     """assume W=None"""
#     self.skt_clusters = dict(cat_centers = np.zeros((self.ncats,self.dim),dtype=np.float32),
#                              clt_centers=[], clt_assigns = [], clt_cat = [],cprob = [],
#                              nn = [], nii = [], nsi = [],
#                              clabel_bs = -1*np.ones(self.skt_labels.size,dtype=np.int64))
#     self.img_clusters = dict(cat_centers=np.zeros((self.ncats, self.dim),dtype=np.float32),
#                              clt_centers = [], clt_assigns = [], clt_cat=[],cprob = [],
#                              nn = [], nii = [], nsi = [],
#                              clabel_bs = -1*np.ones(self.img_labels.size,dtype=np.int64))
#     skt_ccounter = img_ccounter = 0
#     mycluster = pycluster(ctype='kmeans', auto_alg = 'bic')
#     for cat in range(self.ncats):
#       skt_id = np.where(self.skt_labels == cat)[0]
#       img_id = np.where(self.img_labels == cat)[0]
#       skt_feats = self.skt_feats[skt_id]
#       img_feats = self.img_feats[img_id]
#       #PCA
#       skt_pca = PCA().fit(skt_feats)
#       img_pca = PCA().fit(img_feats)
#       skt_std_inv = np.diag(1.0/np.sqrt(np.where(skt_pca.explained_variance_>0, skt_pca.explained_variance_,1.0)))
#       img_std = np.diag(np.sqrt(np.where(img_pca.explained_variance_>0, img_pca.explained_variance_,1.0)))
#       skt_feats_trans = skt_pca.transform(skt_feats).dot(skt_std_inv).dot(img_std).dot(img_pca.components_) + img_pca.mean_[None,:]#
#       
#       #skt clustering
#       self.skt_clusters['cat_centers'][cat,...] = skt_feats_trans.mean(axis=0)
#       mycluster.fit(skt_feats_trans, k_min=2, k_max=self.nskt_c)
#       tcluster = mycluster.predict(skt_feats_trans)
#       csize = []
#       skt_clt_c = []
#       for i in range(mycluster.n_clusters):
#         tassignments = skt_id[tcluster==i]
#         if len(tassignments) > self.skt_thres:
#           skt_clt_c.append(mycluster.cluster_centers_[i].copy())
#           self.skt_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
#           self.skt_clusters['clt_cat'].append(cat)
#           self.skt_clusters['clabel_bs'][tassignments] = skt_ccounter
#           csize.append(len(tassignments))
#           skt_ccounter += 1
#       self.skt_clusters['clt_centers'].extend(skt_clt_c)
#       self.skt_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
#       
#       #img clustering
#       self.img_clusters['cat_centers'][cat,...] = img_feats.mean(axis=0)
#       mycluster = mycluster.fit(img_feats, k_min=2, k_max=self.nimg_c)
#       tcluster = mycluster.predict(img_feats)
#       csize = []
#       img_clt_c = []
#       for i in range(mycluster.n_clusters):
#         tassignments = img_id[tcluster==i]
#         if len(tassignments) > self.img_thres:
#           img_clt_c.append(mycluster.cluster_centers_[i].copy())
#           self.img_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
#           self.img_clusters['clt_cat'].append(cat)
#           self.img_clusters['clabel_bs'][tassignments] = img_ccounter #cluster label by sample
#           csize.append(len(tassignments))
#           img_ccounter += 1
#       self.img_clusters['clt_centers'].extend(img_clt_c)
#       self.img_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
#       
#       #skt: kNC
#       skt_clt_c_array = np.array(skt_clt_c)+np.zeros((1,1))
#       img_clt_c_array = np.array(img_clt_c)+np.zeros((1,1))
#       nbrs = NearestNeighbors(1, algorithm = 'brute').fit(img_clt_c_array)
#       _, inds = nbrs.kneighbors(skt_clt_c_array)
#       self.skt_clusters['nn'].extend(list(inds.squeeze()+img_ccounter-len(img_clt_c)+np.zeros(1,np.int64)))
#       
#       #img: kNC
#       nbrs = NearestNeighbors(1, algorithm = 'brute').fit(skt_clt_c_array)
#       _, inds = nbrs.kneighbors(img_clt_c_array)
#       self.img_clusters['nn'].extend(list(inds.squeeze()+skt_ccounter-len(skt_clt_c)+np.zeros(1,np.int64)))
#       
#     self.skt_clusters['clt_centers'] = np.array(self.skt_clusters['clt_centers'], dtype=np.float32)
#     self.skt_clusters['clt_cat'] = np.array(self.skt_clusters['clt_cat'], dtype = np.int64)
#     self.skt_clusters['nn'] = np.array(self.skt_clusters['nn'], dtype=np.int64)
#     
#     self.img_clusters['clt_centers'] = np.array(self.img_clusters['clt_centers'], dtype=np.float32)
#     self.img_clusters['clt_cat'] = np.array(self.img_clusters['clt_cat'], dtype = np.int64)
#     self.img_clusters['nn'] = np.array(self.img_clusters['nn'], dtype=np.int64)
#     
#     #SKT: nearest imposters
#     nskt_clt = self.skt_clusters['clt_cat'].size
#     nimg_clt = self.img_clusters['clt_cat'].size
#     print('Clustering done. nskt c = {}, nimg_c = {}'.format(nskt_clt, nimg_clt))
#     self.skt_clusters['nsi'] = np.zeros((nskt_clt, self.nskt_impostor),dtype=np.int64)
#     self.skt_clusters['nii'] = np.zeros((nskt_clt, self.nimg_impostor),dtype=np.int64)
#     nbrs_skt = NearestNeighbors(min((self.nskt_c*(1+self.nimg_impostor)+self.nskt_impostor,nskt_clt))).fit(self.skt_clusters['clt_centers'])
#     nbrs_img = NearestNeighbors(min((self.nimg_c*(1+self.nskt_impostor)+self.nimg_impostor,nimg_clt))).fit(self.img_clusters['clt_centers'])
#     _, inds_skt_skt = nbrs_skt.kneighbors(self.skt_clusters['clt_centers'])
#     _, inds_skt_img = nbrs_img.kneighbors(self.skt_clusters['clt_centers'])
#     for ci in range(nskt_clt):
#       cat_id = self.skt_clusters['clt_cat'][ci]
#       sel_cid_skt = [c for c in inds_skt_skt[ci] if self.skt_clusters['clt_cat'][c] != cat_id]
#       sel_cid_skt = sel_cid_skt[:self.nskt_impostor]
#       self.skt_clusters['nsi'][ci,...] = np.array(sel_cid_skt, dtype=np.int64)
#       
#       exclude_cat_lst = set([self.skt_clusters['clt_cat'][c] for c in sel_cid_skt] + [cat_id,])
#       sel_cid_img = [c for c in inds_skt_img[ci] if self.img_clusters['clt_cat'][c] not in exclude_cat_lst]
#       self.skt_clusters['nii'][ci,...] = np.array(sel_cid_img[:self.nimg_impostor], dtype=np.int64)
#       
#     #IMG: nearest imposters
#     self.img_clusters['nsi'] = np.zeros((nimg_clt, self.nskt_impostor),dtype=np.int64)
#     self.img_clusters['nii'] = np.zeros((nimg_clt, self.nimg_impostor),dtype=np.int64)
#     _, inds_img_skt = nbrs_skt.kneighbors(self.img_clusters['clt_centers'])
#     _, inds_img_img = nbrs_img.kneighbors(self.img_clusters['clt_centers'])
#     for ci in range(nimg_clt):
#       cat_id = self.img_clusters['clt_cat'][ci]
#       sel_cid_img = [c for c in inds_img_img[ci] if self.img_clusters['clt_cat'][c] != cat_id]
#       sel_cid_img = sel_cid_img[:self.nimg_impostor]
#       self.img_clusters['nii'][ci,...] = np.array(sel_cid_img, dtype=np.int64)
#       
#       exclude_cat_lst = set([self.img_clusters['clt_cat'][c] for c in sel_cid_img] + [cat_id,])
#       sel_cid_skt = [c for c in inds_img_skt[ci] if self.skt_clusters['clt_cat'][c] not in exclude_cat_lst]
#       self.img_clusters['nsi'][ci,...] = np.array(sel_cid_skt[:self.nskt_impostor], dtype=np.int64)
# =============================================================================
  
  def kNC(self, W = None):
    """both skt and img perspective"""
    self.skt_clusters = dict(cat_centers = np.zeros((self.ncats,self.dim),dtype=np.float32),
                             clt_centers=[], clt_assigns = [], clt_cat = [],cprob = [],
                             nn = [], nii = [], nsi = [],clt_prob_bs=[],
                             clabel_bs = -1*np.ones(self.skt_labels.size,dtype=np.int64))
    self.img_clusters = dict(cat_centers=np.zeros((self.ncats, self.dim),dtype=np.float32),
                             clt_centers = [], clt_assigns = [], clt_cat=[],cprob = [],
                             nn = [], nii = [], nsi = [],clt_prob_bs=[],
                             clabel_bs = -1*np.ones(self.img_labels.size,dtype=np.int64))
    skt_ccounter = img_ccounter = 0
    mycluster = pycluster(ctype='kmeans', auto_alg = 'bic')
    KL = divergence_metric('kl', inverse = False)
    for cat in range(self.ncats):
      skt_id = np.where(self.skt_labels == cat)[0]
      img_id = np.where(self.img_labels == cat)[0]
      skt_feats = np.c_[self.pca.transform(self.skt_feats[skt_id]), np.ones((skt_id.size,1),np.float32)].dot(W[cat])
      img_feats = self.pca.transform(self.img_feats[img_id])
      nskts = skt_feats.shape[0]
      nimgs = img_feats.shape[0]
      
      #warp
# =============================================================================
#       skt_pca = PCA().fit(skt_feats)
#       img_pca = PCA().fit(img_feats)
#       skt_std_inv = np.diag(1.0/np.sqrt(np.where(skt_pca.explained_variance_>0, skt_pca.explained_variance_,1.0)))
#       img_std = np.diag(np.sqrt(np.where(img_pca.explained_variance_>0, img_pca.explained_variance_,1.0)))
#       skt_feats_trans = skt_pca.transform(skt_feats).dot(skt_std_inv).dot(img_std).dot(img_pca.components_) + img_pca.mean_[None,:]#
# =============================================================================
      skt_feats_trans = skt_feats
      
      #dimension reduction
# =============================================================================
#       allpca = PCA(0.95).fit(np.concatenate((skt_feats_trans, img_feats)))
#       allfeat = allpca.transform(np.concatenate((skt_feats_trans, img_feats)))
#       skt_feats_trans = allfeat[:nskts]
#       img_feats = allfeat[-nimgs:]
# =============================================================================
      allpca = self.pca
      
      #skt clustering
      self.skt_clusters['cat_centers'][cat,...] = self.skt_feats[skt_id].mean(axis=0)
      mycluster.fit(skt_feats_trans, k_min=1, k_max=self.nskt_c)
      #tcluster1 = mycluster.predict(skt_feats_trans)
      tproba = mycluster.predict_proba(skt_feats_trans)
      tcluster = tproba.argmax(axis=1)
      csize = []
      skt_clt_c = []
      skt_clt_v = []
      for i in range(mycluster.n_clusters):
        tassignments = skt_id[tcluster==i]
        if len(tassignments) > self.skt_thres:
          skt_clt_c.append(mycluster.cluster_centers_[i].copy())
          #skt_clt_v.append(mycluster.model.covariances_[i].copy())
          prob_ = tproba[tcluster==i, i]
          self.skt_clusters['clt_prob_bs'].append(prob_/prob_.sum())
          self.skt_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
          self.skt_clusters['clt_centers'].append(self.skt_feats[tassignments].mean(axis=0))
          self.skt_clusters['clt_cat'].append(cat)
          self.skt_clusters['clabel_bs'][tassignments] = skt_ccounter
          csize.append(len(tassignments))
          skt_ccounter += 1
      #self.skt_clusters['clt_centers'].extend([allpca.inverse_transform(kk) for kk in skt_clt_c])
      self.skt_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
      
      #img clustering
      self.img_clusters['cat_centers'][cat,...] = self.img_feats[img_id].mean(axis=0)
      mycluster.fit(img_feats, k_min=1, k_max=self.nimg_c)
      #tcluster = mycluster.predict(img_feats)
      tproba = mycluster.predict_proba(img_feats)
      tcluster = tproba.argmax(axis=1)
      csize = []
      img_clt_c = []
      img_clt_v = []
      for i in range(mycluster.n_clusters):
        tassignments = img_id[tcluster==i]
        if len(tassignments) > self.img_thres:
          img_clt_c.append(mycluster.cluster_centers_[i].copy())
          #img_clt_v.append(mycluster.model.covariances_[i].copy())
          prob_ = tproba[tcluster==i, i]
          self.img_clusters['clt_prob_bs'].append(prob_/prob_.sum())
          self.img_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
          self.img_clusters['clt_centers'].append(self.img_feats[tassignments].mean(axis=0))
          self.img_clusters['clt_cat'].append(cat)
          self.img_clusters['clabel_bs'][tassignments] = img_ccounter #cluster label by sample
          csize.append(len(tassignments))
          img_ccounter += 1
      #self.img_clusters['clt_centers'].extend([allpca.inverse_transform(kk) for kk in img_clt_c])
      self.img_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
      
      #skt: kNC
      skt_clt_c_array = np.array(skt_clt_c)+np.zeros((1,1))
      img_clt_c_array = np.array(img_clt_c)+np.zeros((1,1))
# =============================================================================
#       skt_clt_v_array = np.array(skt_clt_v)+np.zeros((1,1))
#       img_clt_v_array = np.array(img_clt_v)+np.zeros((1,1))
# =============================================================================
      
      #inds = KL.neighbour(skt_clt_c_array, skt_clt_v_array, img_clt_c_array, img_clt_v_array)
      nbrs = NearestNeighbors(1, algorithm = 'brute').fit(img_clt_c_array)
      _, inds = nbrs.kneighbors(skt_clt_c_array)

      self.skt_clusters['nn'].extend(list(inds.squeeze()+img_ccounter-len(img_clt_c)+np.zeros(1,np.int64)))
      
      #img: kNC
      #inds = KL.neighbour(img_clt_c_array, img_clt_v_array, skt_clt_c_array, skt_clt_v_array)
      nbrs = NearestNeighbors(1, algorithm = 'brute').fit(skt_clt_c_array)
      _, inds = nbrs.kneighbors(img_clt_c_array)
      
      self.img_clusters['nn'].extend(list(inds.squeeze()+skt_ccounter-len(skt_clt_c)+np.zeros(1,np.int64)))
      
    self.skt_clusters['clt_centers'] = np.array(self.skt_clusters['clt_centers'], dtype=np.float32)
    self.skt_clusters['clt_cat'] = np.array(self.skt_clusters['clt_cat'], dtype = np.int64)
    self.skt_clusters['nn'] = np.array(self.skt_clusters['nn'], dtype=np.int64)
    
    self.img_clusters['clt_centers'] = np.array(self.img_clusters['clt_centers'], dtype=np.float32)
    self.img_clusters['clt_cat'] = np.array(self.img_clusters['clt_cat'], dtype = np.int64)
    self.img_clusters['nn'] = np.array(self.img_clusters['nn'], dtype=np.int64)
    
    #SKT: nearest imposters
    nskt_clt = self.skt_clusters['clt_cat'].size
    nimg_clt = self.img_clusters['clt_cat'].size
    print('Clustering done. nskt c = {}, nimg_c = {}'.format(nskt_clt, nimg_clt))
    self.skt_clusters['nsi'] = np.zeros((nskt_clt, self.nskt_impostor),dtype=np.int64)
    self.skt_clusters['nii'] = np.zeros((nskt_clt, self.nimg_impostor),dtype=np.int64)
    nbrs_skt = NearestNeighbors(min((self.nskt_c*(1+self.nimg_impostor)+self.nskt_impostor,nskt_clt))).fit(self.skt_clusters['clt_centers'])
    nbrs_img = NearestNeighbors(min((self.nimg_c*(1+self.nskt_impostor)+self.nimg_impostor,nimg_clt))).fit(self.img_clusters['clt_centers'])
    _, inds_skt_skt = nbrs_skt.kneighbors(self.skt_clusters['clt_centers'])
    _, inds_skt_img = nbrs_img.kneighbors(self.skt_clusters['clt_centers'])
    for ci in range(nskt_clt):
      cat_id = self.skt_clusters['clt_cat'][ci]
      try:
        sel_cid_skt = [c for c in inds_skt_skt[ci] if self.skt_clusters['clt_cat'][c] != cat_id]
      except:
        print 'Error happen'
      sel_cid_skt = sel_cid_skt[:self.nskt_impostor]
      self.skt_clusters['nsi'][ci,...] = np.array(sel_cid_skt, dtype=np.int64)
      
      exclude_cat_lst = set([self.skt_clusters['clt_cat'][c] for c in sel_cid_skt] + [cat_id,])
      sel_cid_img = [c for c in inds_skt_img[ci] if self.img_clusters['clt_cat'][c] not in exclude_cat_lst]
      self.skt_clusters['nii'][ci,...] = np.array(sel_cid_img[:self.nimg_impostor], dtype=np.int64)
      
    #IMG: nearest imposters
    self.img_clusters['nsi'] = np.zeros((nimg_clt, self.nskt_impostor),dtype=np.int64)
    self.img_clusters['nii'] = np.zeros((nimg_clt, self.nimg_impostor),dtype=np.int64)
    _, inds_img_skt = nbrs_skt.kneighbors(self.img_clusters['clt_centers'])
    _, inds_img_img = nbrs_img.kneighbors(self.img_clusters['clt_centers'])
    for ci in range(nimg_clt):
      cat_id = self.img_clusters['clt_cat'][ci]
      sel_cid_img = [c for c in inds_img_img[ci] if self.img_clusters['clt_cat'][c] != cat_id]
      sel_cid_img = sel_cid_img[:self.nimg_impostor]
      self.img_clusters['nii'][ci,...] = np.array(sel_cid_img, dtype=np.int64)
      
      exclude_cat_lst = set([self.img_clusters['clt_cat'][c] for c in sel_cid_img] + [cat_id,])
      sel_cid_skt = [c for c in inds_img_skt[ci] if self.skt_clusters['clt_cat'][c] not in exclude_cat_lst]
      self.img_clusters['nsi'][ci,...] = np.array(sel_cid_skt[:self.nskt_impostor], dtype=np.int64)
      
  def save2file(self, file_path):
    if os.path.exists(file_path):
      os.rename(file_path, file_path+'.' + str(self.counter))
      self.counter += 1
    if file_path.endswith('.pkl'):
      with open(file_path, 'wb') as pkl:
        pickle.dump(self.skt_clusters,pkl)
        pickle.dump(self.img_clusters, pkl)
    elif file_path.endswith('.json'):
      with open(file_path,'w') as jf:
        jf.write('[')
        json.dump(self.skt_clusters, jf)
        jf.write(',')
        json.dump(self.img_clusters, jf)
        jf.write(']')
    
###############################################################################
class MagnetCluster2(object):
  """one class to process both unlabelled and lablled dataset"""
  def __init__(self,dim, ncats, nskts, nimgs, nskt_c, nimg_c, skt_thres = 25, img_thres=25):
    self.skt_feats = np.zeros((nskts, dim), dtype= np.float32)
    self.img_feats = np.zeros((nimgs, dim), dtype = np.float32)
    self.skt_labels = np.zeros(nskts, dtype = np.int64)
    self.img_labels = np.zeros(nimgs, dtype = np.int64)
    self.dim = dim
    self.ncats = ncats
    self.nskt_c = nskt_c
    self.nimg_c = nimg_c
    self.skt_thres = skt_thres
    self.img_thres = img_thres
    self.pca = None
    self.counter =0
  def set_finegrain_data(self, nskts2, nimgs2):
    self.skt_feats2 = np.zeros((nskts2, self.dim), dtype= np.float32)
    self.img_feats2 = np.zeros((nimgs2, self.dim), dtype = np.float32)
    self.skt_labels2 = np.zeros(nskts2, dtype = np.int64)
    self.img_labels2 = np.zeros(nimgs2, dtype = np.int64)
  
  def set_impostor(self, nskt_impostor=1, nimg_impostor=1):
    """configure number of skt and img impostor clusters"""
    self.nskt_impostor = nskt_impostor
    self.nimg_impostor = nimg_impostor
    
  def doPCA(self):
    self.pca = PCA(0.95).fit(np.concatenate((self.skt_feats, self.img_feats)))
    return copy.deepcopy(self.pca)
      
      
  def reg(self, skt_id, skt_feat, skt_label, img_id, img_feat, img_label, net_id):
    """register skt and img features"""
    if net_id==0:
      if skt_id is not None:
        self.skt_feats[skt_id,...]= skt_feat
        self.skt_labels[skt_id] = copy.copy(skt_label)
      if img_id is not None:
        self.img_feats[img_id,...] = img_feat
        self.img_labels[img_id] = copy.copy(img_label)
    else:
      if skt_id is not None:
        self.skt_feats2[skt_id,...]= skt_feat
        self.skt_labels2[skt_id] = copy.copy(skt_label)
      if img_id is not None:
        self.img_feats2[img_id,...] = img_feat
        self.img_labels2[img_id] = copy.copy(img_label)
  def kNC(self):
    """cluster both skt and img together after warp"""
    self.skt_clusters = dict(cat_centers = np.zeros((self.ncats,self.dim),dtype=np.float32),
                             clt_centers=[], clt_assigns = [], clt_cat = [],cprob = [],
                             nn = [], nii = [], nsi = [],clt_prob_bs=[],
                             clabel_bs = -1*np.ones(self.skt_labels.size,dtype=np.int64))
    self.img_clusters = dict(cat_centers=np.zeros((self.ncats, self.dim),dtype=np.float32),
                             clt_centers = [], clt_assigns = [], clt_cat=[],cprob = [],
                             nn = [], nii = [], nsi = [],clt_prob_bs=[],
                             clabel_bs = -1*np.ones(self.img_labels.size,dtype=np.int64))
    ccounter = 0
    mycluster = pycluster(ctype='gmm', auto_alg = 'auto')
    allpca = self.doPCA()
    newdim = allpca.n_components_
    nspi = len(self.skt_labels2)/len(self.img_labels2)
    warp = matrixWarp(niter=200)
    for cat in range(self.ncats):
      skt_id = np.where(self.skt_labels == cat)[0]
      img_id = np.where(self.img_labels == cat)[0]
      skt_id2 = np.where(self.skt_labels2 == cat)[0]
      img_id2 = np.where(self.img_labels2 == cat)[0]
      skt_feats = allpca.transform(self.skt_feats[skt_id])
      img_feats = allpca.transform(self.img_feats[img_id])
      skt_feats2 = allpca.transform(self.skt_feats2[skt_id2])
      img_feats2 = allpca.transform(self.img_feats2[img_id2])
      img_feats2 = np.repeat(img_feats2[:,None,:], nspi, axis=1).reshape((-1,newdim))
      nskts = skt_feats.shape[0]
      nimgs = img_feats.shape[0]
      
      #warp
# =============================================================================
#       #using unsupervised method(not ultilise labelled features)
#       skt_pca = PCA().fit(skt_feats)
#       img_pca = PCA().fit(img_feats)
#       skt_std_inv = np.diag(1.0/np.sqrt(np.where(skt_pca.explained_variance_>0, skt_pca.explained_variance_,1.0)))
#       img_std = np.diag(np.sqrt(np.where(img_pca.explained_variance_>0, img_pca.explained_variance_,1.0)))
#       skt_feats_trans = skt_pca.transform(skt_feats).dot(skt_std_inv).dot(img_std).dot(img_pca.components_) + img_pca.mean_[None,:]#
#       img_feats_trans = img_feats
# =============================================================================

      #  using manifold alignment
      mn = manifold(skt_feats, skt_feats2, img_feats, img_feats2, 10.0, 16)
      skt_feats_trans, _, img_feats_trans,_ = mn.align_linear()
      
      #using matrix transformation
# =============================================================================
#       warp.learn(skt_feats2, img_feats2)
#       skt_feats_trans, img_feats_trans = warp.warp(skt_feats), warp.warp(img_feats)
# =============================================================================
      
      
      self.skt_clusters['cat_centers'][cat,...] = self.skt_feats[skt_id].mean(axis=0)
      self.img_clusters['cat_centers'][cat,...] = self.img_feats[img_id].mean(axis=0)
      
      mycluster.fit(np.r_[skt_feats_trans, img_feats_trans], k_min=2, k_max=self.nskt_c)
      tproba = mycluster.predict_proba(skt_feats_trans)
      sktproba = tproba[:nskts]
      imgproba = tproba[-nimgs:]
      sktcluster = mycluster.labels[:nskts]#predict(skt_feats_trans)#sktproba.argmax(axis=1)
      imgcluster = mycluster.labels[-nimgs:]#predict(img_feats_trans)#imgproba.argmax(axis=1)
      sktsize = []
      imgsize = []
      #skt_clt_c = []
      subcounter = 0
      for i in range(mycluster.n_clusters):
        skt_assign = skt_id[sktcluster==i]
        img_assign = img_id[imgcluster==i]
        if len(skt_assign) > self.skt_thres and len(img_assign) > self.img_thres:
          #skt_clt_c.append(mycluster.cluster_centers_[i].copy())
          #skt_clt_v.append(mycluster.model.covariances_[i].copy())
          prob_skt = sktproba[sktcluster==i, i]
          self.skt_clusters['clt_prob_bs'].append(prob_skt/prob_skt.sum())
          self.skt_clusters['clt_assigns'].append(copy.deepcopy(skt_assign))
          self.skt_clusters['clt_centers'].append(self.skt_feats[skt_assign].mean(axis=0))
          self.skt_clusters['clt_cat'].append(cat)
          self.skt_clusters['clabel_bs'][skt_assign] = ccounter
          sktsize.append(len(skt_assign))
          
          
          prob_img = imgproba[imgcluster==i, i]
          self.img_clusters['clt_prob_bs'].append(prob_img/prob_img.sum())
          self.img_clusters['clt_assigns'].append(copy.deepcopy(img_assign))
          self.img_clusters['clt_centers'].append(self.img_feats[img_assign].mean(axis=0))
          self.img_clusters['clt_cat'].append(cat)
          self.img_clusters['clabel_bs'][img_assign] = ccounter
          imgsize.append(len(img_assign))
          subcounter += 1
          ccounter += 1
      
      if subcounter==0:#no cluster eligible, put all into 1 cluster
        self.skt_clusters['clt_prob_bs'].append(np.ones(nskts, dtype=np.float32)/nskts)
        self.skt_clusters['clt_assigns'].append(copy.deepcopy(skt_id))
        self.skt_clusters['clt_centers'].append(self.skt_feats[skt_id].mean(axis=0))
        self.skt_clusters['clt_cat'].append(cat)
        self.skt_clusters['clabel_bs'][skt_id] = ccounter
        sktsize.append(nskts)
        self.img_clusters['clt_prob_bs'].append(np.ones(nimgs, dtype=np.float32)/nimgs)
        self.img_clusters['clt_assigns'].append(copy.deepcopy(img_id))
        self.img_clusters['clt_centers'].append(self.img_feats[img_id].mean(axis=0))
        self.img_clusters['clt_cat'].append(cat)
        self.img_clusters['clabel_bs'][img_id] = ccounter
        imgsize.append(nimgs)
        ccounter+=1

      self.skt_clusters['cprob'].append(np.array(sktsize,dtype=np.float32)/np.sum(sktsize))
      self.img_clusters['cprob'].append(np.array(imgsize,dtype=np.float32)/np.sum(imgsize))
    
    self.skt_clusters['clt_centers'] = np.array(self.skt_clusters['clt_centers'], dtype=np.float32)
    self.skt_clusters['clt_cat'] = np.array(self.skt_clusters['clt_cat'], dtype = np.int64)
    #self.skt_clusters['nn'] = np.array(self.skt_clusters['nn'], dtype=np.int64)
    
    self.img_clusters['clt_centers'] = np.array(self.img_clusters['clt_centers'], dtype=np.float32)
    self.img_clusters['clt_cat'] = np.array(self.img_clusters['clt_cat'], dtype = np.int64)
    #self.img_clusters['nn'] = np.array(self.img_clusters['nn'], dtype=np.int64)
    
    #SKT: nearest imposters
    nskt_clt = self.skt_clusters['clt_cat'].size
    nimg_clt = self.img_clusters['clt_cat'].size
    print('Clustering done. nskt c = {}, nimg_c = {}'.format(nskt_clt, nimg_clt))
    self.skt_clusters['nsi'] = np.zeros((nskt_clt, self.nskt_impostor),dtype=np.int64)
    self.skt_clusters['nii'] = np.zeros((nskt_clt, self.nimg_impostor),dtype=np.int64)
    nbrs_skt = NearestNeighbors(min((self.nskt_c*(1+self.nimg_impostor)+self.nskt_impostor,nskt_clt))).fit(self.skt_clusters['clt_centers'])
    nbrs_img = NearestNeighbors(min((self.nimg_c*(1+self.nskt_impostor)+self.nimg_impostor,nimg_clt))).fit(self.img_clusters['clt_centers'])
    dist_skt_skt, inds_skt_skt = nbrs_skt.kneighbors(self.skt_clusters['clt_centers'])
    dist_skt_img, inds_skt_img = nbrs_img.kneighbors(self.skt_clusters['clt_centers'])

    for ci in range(nskt_clt):
      cat_id = self.skt_clusters['clt_cat'][ci]
      sel_cid_skt = [c for c in inds_skt_skt[ci] if self.skt_clusters['clt_cat'][c] != cat_id]
      sel_cid_skt = sel_cid_skt[:self.nskt_impostor]
      self.skt_clusters['nsi'][ci,...] = np.array(sel_cid_skt, dtype=np.int64)
      
      #exclude_cat_lst = set([self.skt_clusters['clt_cat'][c] for c in sel_cid_skt] + [cat_id,])
      sel_cid_img = [c for c in inds_skt_img[ci] if self.img_clusters['clt_cat'][c] != cat_id]#not in exclude_cat_lst]
      self.skt_clusters['nii'][ci,...] = np.array(sel_cid_img[:self.nimg_impostor], dtype=np.int64)
      
    #IMG: nearest imposters
    self.img_clusters['nsi'] = np.zeros((nimg_clt, self.nskt_impostor),dtype=np.int64)
    self.img_clusters['nii'] = np.zeros((nimg_clt, self.nimg_impostor),dtype=np.int64)
    dist_img_skt, inds_img_skt = nbrs_skt.kneighbors(self.img_clusters['clt_centers'])
    dist_img_img, inds_img_img = nbrs_img.kneighbors(self.img_clusters['clt_centers'])
    for ci in range(nimg_clt):
      cat_id = self.img_clusters['clt_cat'][ci]
      sel_cid_img = [c for c in inds_img_img[ci] if self.img_clusters['clt_cat'][c] != cat_id]
      sel_cid_img = sel_cid_img[:self.nimg_impostor]
      self.img_clusters['nii'][ci,...] = np.array(sel_cid_img, dtype=np.int64)
      
      #exclude_cat_lst = set([self.img_clusters['clt_cat'][c] for c in sel_cid_img] + [cat_id,])
      sel_cid_skt = [c for c in inds_img_skt[ci] if self.skt_clusters['clt_cat'][c] != cat_id]#not in exclude_cat_lst]
      self.img_clusters['nsi'][ci,...] = np.array(sel_cid_skt[:self.nskt_impostor], dtype=np.int64)
      
    
    
  def kNC2(self):
    """both skt and img perspective"""
    self.skt_clusters = dict(cat_centers = np.zeros((self.ncats,self.dim),dtype=np.float32),
                             clt_centers=[], clt_assigns = [], clt_cat = [],cprob = [],
                             nn = [], nii = [], nsi = [],clt_prob_bs=[],
                             clabel_bs = -1*np.ones(self.skt_labels.size,dtype=np.int64))
    self.img_clusters = dict(cat_centers=np.zeros((self.ncats, self.dim),dtype=np.float32),
                             clt_centers = [], clt_assigns = [], clt_cat=[],cprob = [],
                             nn = [], nii = [], nsi = [],clt_prob_bs=[],
                             clabel_bs = -1*np.ones(self.img_labels.size,dtype=np.int64))
    skt_ccounter = img_ccounter = 0
    mycluster = pycluster(ctype='kmeans', auto_alg = 'bic')
    KL = divergence_metric('kl', inverse = False)
    allpca = self.doPCA()
    newdim = allpca.n_components_
    nspi = len(self.skt_labels2)/len(self.img_labels2)
    for cat in range(self.ncats):
      skt_id = np.where(self.skt_labels == cat)[0]
      img_id = np.where(self.img_labels == cat)[0]
      skt_id2 = np.where(self.skt_labels2 == cat)[0]
      img_id2 = np.where(self.img_labels2 == cat)[0]
      skt_feats = allpca.transform(self.skt_feats[skt_id])
      img_feats = allpca.transform(self.img_feats[img_id])
      skt_feats2 = allpca.transform(self.skt_feats2[skt_id2])
      img_feats2 = allpca.transform(self.img_feats2[img_id2])
      img_feats2 = np.repeat(img_feats2[:,None,:], nspi, axis=1).reshape((-1,newdim))
      nskts = skt_feats.shape[0]
      nimgs = img_feats.shape[0]
      
      mn = manifold(skt_feats, skt_feats2, img_feats, img_feats2, 2.0, 16)
      skt_feats_trans, _, img_feats_trans,_ = mn.align_linear()
      
      #skt clustering
      self.skt_clusters['cat_centers'][cat,...] = self.skt_feats[skt_id].mean(axis=0)
      mycluster.fit(skt_feats_trans, k_min=1, k_max=self.nskt_c)
      #tcluster1 = mycluster.predict(skt_feats_trans)
      tproba = mycluster.predict_proba(skt_feats_trans)
      tcluster = tproba.argmax(axis=1)
      csize = []
      skt_clt_c = []
      skt_clt_v = []
      for i in range(mycluster.n_clusters):
        tassignments = skt_id[tcluster==i]
        if len(tassignments) > self.skt_thres:
          skt_clt_c.append(mycluster.cluster_centers_[i].copy())
          #skt_clt_v.append(mycluster.model.covariances_[i].copy())
          prob_ = tproba[tcluster==i, i]
          self.skt_clusters['clt_prob_bs'].append(prob_/prob_.sum())
          self.skt_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
          self.skt_clusters['clt_centers'].append(self.skt_feats[tassignments].mean(axis=0))
          self.skt_clusters['clt_cat'].append(cat)
          self.skt_clusters['clabel_bs'][tassignments] = skt_ccounter
          csize.append(len(tassignments))
          skt_ccounter += 1
      #self.skt_clusters['clt_centers'].extend([allpca.inverse_transform(kk) for kk in skt_clt_c])
      self.skt_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
      
      #img clustering
      self.img_clusters['cat_centers'][cat,...] = self.img_feats[img_id].mean(axis=0)
      mycluster.fit(img_feats_trans, k_min=1, k_max=self.nimg_c)
      #tcluster = mycluster.predict(img_feats)
      tproba = mycluster.predict_proba(img_feats_trans)
      tcluster = tproba.argmax(axis=1)
      csize = []
      img_clt_c = []
      img_clt_v = []
      for i in range(mycluster.n_clusters):
        tassignments = img_id[tcluster==i]
        if len(tassignments) > self.img_thres:
          img_clt_c.append(mycluster.cluster_centers_[i].copy())
          #img_clt_v.append(mycluster.model.covariances_[i].copy())
          prob_ = tproba[tcluster==i, i]
          self.img_clusters['clt_prob_bs'].append(prob_/prob_.sum())
          self.img_clusters['clt_assigns'].append(copy.deepcopy(tassignments))
          self.img_clusters['clt_centers'].append(self.img_feats[tassignments].mean(axis=0))
          self.img_clusters['clt_cat'].append(cat)
          self.img_clusters['clabel_bs'][tassignments] = img_ccounter #cluster label by sample
          csize.append(len(tassignments))
          img_ccounter += 1
      #self.img_clusters['clt_centers'].extend([allpca.inverse_transform(kk) for kk in img_clt_c])
      self.img_clusters['cprob'].append(np.array(csize,dtype=np.float32)/np.sum(csize))
      
      #skt: kNC
      skt_clt_c_array = np.array(skt_clt_c)+np.zeros((1,1))
      img_clt_c_array = np.array(img_clt_c)+np.zeros((1,1))
# =============================================================================
#       skt_clt_v_array = np.array(skt_clt_v)+np.zeros((1,1))
#       img_clt_v_array = np.array(img_clt_v)+np.zeros((1,1))
# =============================================================================
      
      #inds = KL.neighbour(skt_clt_c_array, skt_clt_v_array, img_clt_c_array, img_clt_v_array)
      nbrs = NearestNeighbors(1, algorithm = 'brute').fit(img_clt_c_array)
      _, inds = nbrs.kneighbors(skt_clt_c_array)

      self.skt_clusters['nn'].extend(list(inds.squeeze()+img_ccounter-len(img_clt_c)+np.zeros(1,np.int64)))
      
      #img: kNC
      #inds = KL.neighbour(img_clt_c_array, img_clt_v_array, skt_clt_c_array, skt_clt_v_array)
      nbrs = NearestNeighbors(1, algorithm = 'brute').fit(skt_clt_c_array)
      _, inds = nbrs.kneighbors(img_clt_c_array)
      
      self.img_clusters['nn'].extend(list(inds.squeeze()+skt_ccounter-len(skt_clt_c)+np.zeros(1,np.int64)))
      
    self.skt_clusters['clt_centers'] = np.array(self.skt_clusters['clt_centers'], dtype=np.float32)
    self.skt_clusters['clt_cat'] = np.array(self.skt_clusters['clt_cat'], dtype = np.int64)
    self.skt_clusters['nn'] = np.array(self.skt_clusters['nn'], dtype=np.int64)
    
    self.img_clusters['clt_centers'] = np.array(self.img_clusters['clt_centers'], dtype=np.float32)
    self.img_clusters['clt_cat'] = np.array(self.img_clusters['clt_cat'], dtype = np.int64)
    self.img_clusters['nn'] = np.array(self.img_clusters['nn'], dtype=np.int64)
    
    #SKT: nearest imposters
    nskt_clt = self.skt_clusters['clt_cat'].size
    nimg_clt = self.img_clusters['clt_cat'].size
    print('Clustering done. nskt c = {}, nimg_c = {}'.format(nskt_clt, nimg_clt))
    self.skt_clusters['nsi'] = np.zeros((nskt_clt, self.nskt_impostor),dtype=np.int64)
    self.skt_clusters['nii'] = np.zeros((nskt_clt, self.nimg_impostor),dtype=np.int64)
    nbrs_skt = NearestNeighbors(min((self.nskt_c*(1+self.nimg_impostor)+self.nskt_impostor,nskt_clt))).fit(self.skt_clusters['clt_centers'])
    nbrs_img = NearestNeighbors(min((self.nimg_c*(1+self.nskt_impostor)+self.nimg_impostor,nimg_clt))).fit(self.img_clusters['clt_centers'])
    dist_skt_skt, inds_skt_skt = nbrs_skt.kneighbors(self.skt_clusters['clt_centers'])
    dist_skt_img, inds_skt_img = nbrs_img.kneighbors(self.skt_clusters['clt_centers'])
# =============================================================================
#     self.skt_clusters['cprob'] = []
#     self.img_clusters['cprob'] = []
# =============================================================================
    for ci in range(nskt_clt):
      cat_id = self.skt_clusters['clt_cat'][ci]
      sel_cid_skt = [c for c in inds_skt_skt[ci] if self.skt_clusters['clt_cat'][c] != cat_id]
      sel_cid_skt = sel_cid_skt[:self.nskt_impostor]
      self.skt_clusters['nsi'][ci,...] = np.array(sel_cid_skt, dtype=np.int64)
      
      exclude_cat_lst = set([self.skt_clusters['clt_cat'][c] for c in sel_cid_skt] + [cat_id,])
      sel_cid_img = [c for c in inds_skt_img[ci] if self.img_clusters['clt_cat'][c] not in exclude_cat_lst]
      self.skt_clusters['nii'][ci,...] = np.array(sel_cid_img[:self.nimg_impostor], dtype=np.int64)
      
# =============================================================================
#       sel_dist_skt = dist_skt_skt[np.in1d(inds_skt_skt[ci], sel_cid_skt).nonzero()[0]][:self.nskt_impostor]
#       sel_dist_img = dist_skt_img[np.in1d(inds_skt_img[ci], sel_cid_img).nonzero()[0]][:self.nimg_impostor]
#       self.skt_clusters['cprob'].append(np.exp(-sel_dist_skt).sum()+np.exp(-sel_dist_img).sum())
# =============================================================================
      
    #IMG: nearest imposters
    self.img_clusters['nsi'] = np.zeros((nimg_clt, self.nskt_impostor),dtype=np.int64)
    self.img_clusters['nii'] = np.zeros((nimg_clt, self.nimg_impostor),dtype=np.int64)
    dist_img_skt, inds_img_skt = nbrs_skt.kneighbors(self.img_clusters['clt_centers'])
    dist_img_img, inds_img_img = nbrs_img.kneighbors(self.img_clusters['clt_centers'])
    for ci in range(nimg_clt):
      cat_id = self.img_clusters['clt_cat'][ci]
      sel_cid_img = [c for c in inds_img_img[ci] if self.img_clusters['clt_cat'][c] != cat_id]
      sel_cid_img = sel_cid_img[:self.nimg_impostor]
      self.img_clusters['nii'][ci,...] = np.array(sel_cid_img, dtype=np.int64)
      
      exclude_cat_lst = set([self.img_clusters['clt_cat'][c] for c in sel_cid_img] + [cat_id,])
      sel_cid_skt = [c for c in inds_img_skt[ci] if self.skt_clusters['clt_cat'][c] not in exclude_cat_lst]
      self.img_clusters['nsi'][ci,...] = np.array(sel_cid_skt[:self.nskt_impostor], dtype=np.int64)
      
# =============================================================================
#       sel_dist_img = dist_img_img[np.in1d(inds_img_img[ci], sel_cid_img).nonzero()[0]][:self.nimg_impostor]
#       sel_dist_skt = dist_img_skt[np.in1d(inds_img_skt[ci], sel_cid_skt).nonzero()[0]][:self.nskt_impostor]
#       self.img_clusters['cprob'].append(np.exp(-sel_dist_skt).sum()+np.exp(-sel_dist_img).sum())
#     
#     self.skt_clusters['cprob'] = np.array(self.skt_clusters['cprob'])/np.sum(self.skt_clusters['cprob'])
#     self.img_clusters['cprob'] = np.array(self.img_clusters['cprob'])/np.sum(self.img_clusters['cprob'])
# =============================================================================
    
  def save2file(self, file_path):
    if os.path.exists(file_path):
      os.rename(file_path, file_path+'.' + str(self.counter))
      self.counter += 1
    if file_path.endswith('.pkl'):
      with open(file_path, 'wb') as pkl:
        pickle.dump(self.skt_clusters,pkl)
        pickle.dump(self.img_clusters, pkl)
    elif file_path.endswith('.json'):
      with open(file_path,'w') as jf:
        jf.write('[')
        json.dump(self.skt_clusters, jf)
        jf.write(',')
        json.dump(self.img_clusters, jf)
        jf.write(']')
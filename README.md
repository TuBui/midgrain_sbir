# Triplet Loss SBIR
This repo contains code for the ACCV 2018 paper "[Deep Manifold Alignment for Mid-grain Sketch based Image Retrieval](https://cvssp.org/data/Flickr25K/ACCV18_files/ACCV_Sketch_2018.pdf)" 

## Pre-trained model and the MidGrain65c dataset
The Caffe pretrained model and dataset can be downloaded from our [project page](https://cvssp.org/data/Flickr25K/ACCV18.html).

## Feature extraction
We provide a Python script for extracting features from the image and sketch sets, then query the sketches against images. Simply unzip the model and datasets that you have downloaded, edit the [Retrieval.py](Retrieval.py) with paths to the model and data, then run:

```
python Retrieval.py
``` 

to get the class-level and midgrain-level retrieval mAP.

## Reference
```
@inproceedings{bui2018deep,
title = {Compact descriptors for sketch-based image retrieval using a triplet loss convolutional neural network},
author = {Tu Bui and Leonardo Ribeiro and Moacir Ponti and John Collomosse},
booktitle = {Proceedings of the 14th Asian Conference on Computer Vision},
publisher={Springer}
}

```

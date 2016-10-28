#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from fcolor import FColor 


import matplotlib
from matplotlib.pyplot import plot,savefig
import cv2.cv as cv

def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show()

CLASSES = ('__background__', 'car', 'person', 'bike', 'truck', 'van','tram', 'misc')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'vgg_m': ('VGG_CNN_M_1024',
                   'VGG_CNN_M_1024_faster_rcnn_final.caffemodel')}


def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""
    # Load the demo image

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    index=1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        #im = im[:, :, (2, 1, 0)]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0 and index==len(CLASSES[1:]):
            #cv2.imwrite(path,im)
            #video.write(im)
            return
        elif len(inds) == 0 and index<len(CLASSES[1:]):
            index+=1
            continue
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            x = bbox[0]
            y = bbox[1]
            rect_start = (x,y)
            x1 = bbox[2]
            y1 = bbox[3]
            rect_end = (x1,y1)
            color0=(100,100,100)
            color1=(255,0,0)


            xx1 = bbox[0]
            yy1= int(bbox[1]-10)
            point_start = (xx1,yy1)
            xx2 = bbox[0]+(bbox[2]-bbox[0])*score
            yy2= int(bbox[1]-2)
            point_end = (xx2,yy2)
            color2=(0,0,225)
            color3=(0,255,0)
            if cls_ind in [1, 4, 5, 6, 7, 8]:
                cv2.rectangle(im, rect_start, rect_end, color1, 2)
            elif cls_ind==2:
                cv2.rectangle(im, rect_start, rect_end, color3, 2)
            elif cls_ind==3:
                cv2.rectangle(im, rect_start, rect_end, color0, 2)
            cv2.rectangle(im, point_start, point_end, color2, -1)
    cv2.namedWindow("Image")
    res=cv2.resize(im,(1080,608),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Image", res)
    cv2.waitKey (0)
    #cv2.imwrite(path,im)
    #video.write(im)



def visualization(net, layer_name, save_dir):
    save_path = os.path.join(cfg.ROOT_DIR, 'visualization', save_dir, layer_name) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    feat = net.blobs[layer_name].data[0]
    print feat.shape
    feat -= feat.min()
    feat /= feat.max()
    feat *=255
    i = 0
    for im in feat:
        #iFColor = FColor(im)
        cv2.imwrite(os.path.join(save_path, '{:d}.png'.format(i)), im)
        i = i + 1
    #vis_square(feat, padval=1)

def visualization_plus(net, layer_name, save_dir):
    save_path = os.path.join(cfg.ROOT_DIR, 'visualization_plus', save_dir) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    feat = net.blobs[layer_name].data[0]
    fm = feat[0]
    print type(fm)

    print fm.shape
    for f in feat:
        fm += f
    fm = fm - feat[0]

    print 'fm  max = {}, min = {}'.format(fm.max(), fm.min())
    #fm -= fm.min()
    #fm /= fm.max()

    fm *=255
    i = 0
    cv2.imwrite(os.path.join(save_path, '{:s}.png'.format(layer_name)), fm)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    file_name = '0000008'
    #img_path = os.path.join('visualization', file_name+'.jpg')
    img_path = os.path.join('data/training/image_2', file_name+'.png')
    im = cv2.imread(img_path)
    demo(net, im)
    #visualization_plus(net,'conv4_3', file_name)
    #visualization_plus(net,'conv5_3', file_name)
    visualization(net,'conv3_3', file_name)
    #visualization(net,'conv2_2', file_name)
    #visualization(net,'conv1_1', file_name)
    #visualization(net,'rpn/output', file_name)

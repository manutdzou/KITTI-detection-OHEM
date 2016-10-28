close all;clear all;clc;
path='/home/bsl/KITTI-detection/data';
comp_id='comp4-7629';
test_set='KakouTest';
output_dir='/home/bsl/KITTI-detection/output/faster_rcnn_alt_opt/KakouTest/VGG16_faster_rcnn_final';
img_list='KITTI_val_list.txt';
img_gt='KITTI_gt_val.txt';
res = detection_eval(path, comp_id, test_set,output_dir,img_list,img_gt);

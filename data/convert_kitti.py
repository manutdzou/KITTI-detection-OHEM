#!/usr/bin/env python
# --------------------------------------------------------
# kitti tool
# Copyright (c) 2016 Chao Chen
# Licensed under The MIT License [see LICENSE for details]
# Written by Chao Chen
# --------------------------------------------------------

import os

def getLabelFilename(img_filename):
     vector_string = img_filename.split('.'); 
     tmp = vector_string[0] + '.txt'
     vector_tmp = tmp.split('/')
     return vector_tmp[-1]





def parse_line(img_line, label_file):
    count_p = 0
    count_c = 0  
    count_cyc = 0
    count_van = 0
    count_truck = 0
    count_tram = 0
    count_misc = 0

    p_flag = 0
    c_flag = 0
    cyc_flag = 0
    van_flag = 0
    truck_flag = 0
    misc_flag = 0

    bbox_car = []
    bbox_per = []
    bbox_cyc = []
    bbox_truck = []
    bbox_van = []
    bbox_tram = []
    bbox_misc = []

    
    file = open(label_file, 'r')
    for line in file.xreadlines():
        line = line.strip('\n')
        vector_str = line.split(' ')
        #print vector_str
        if ("Car" == vector_str[0] ):
            count_c += 1
            for k in range(4, 8):
                bbox_car.append(vector_str[k])
            continue
        if ('Pedestrian' == vector_str[0]):
            count_p += 1
            for k in range(4, 8):
                bbox_per.append(vector_str[k])
            continue

        if('Cyclist' == vector_str[0]):
            count_cyc += 1
            for k in range(4, 8):
                bbox_cyc.append(vector_str[k])
            continue

        if('Van' == vector_str[0]):
            count_van += 1
            for k in range(4, 8):
                bbox_van.append(vector_str[k])
            continue

        if('Truck' == vector_str[0]):
            count_truck += 1
            for k in range(4, 8):
                bbox_truck.append(vector_str[k])
            continue
        if('Misc' == vector_str[0]):
            count_misc += 1
            for k in range(4, 8):
                bbox_misc.append(vector_str[k])
            continue

        if('Tram' == vector_str[0]):
            count_tram += 1
            for k in range(4, 8):
                bbox_tram.append(vector_str[k])
            continue

    num = count_c + count_p + count_cyc + count_van + count_truck + count_misc + count_tram

    final_line = img_line + ' '+ str(num)
    #car 
    final_line += ' '+ str(count_c)
    for i in bbox_car:
        final_line += ' ' + i 
    #pre
    final_line += ' '+ str(count_p)
    for i in bbox_per:
        final_line += ' ' + i 
    #cyc
    final_line += ' '+ str(count_cyc)
    for i in bbox_cyc:
        final_line += ' ' + i 
    #truck
    final_line += ' '+ str(count_truck)
    for i in bbox_truck:
        final_line += ' ' + i 
    #van
    final_line += ' '+ str(count_van)
    for i in bbox_van:
        final_line += ' ' + i 
    #tram
    final_line += ' '+ str(count_tram)
    for i in bbox_tram:
        final_line += ' ' + i 
    #misc
    final_line += ' '+ str(count_misc)
    for i in bbox_misc:
        final_line += ' ' + i 
    return final_line + '\n' 

def convertKitti(label_file_list, savedFilename):
    if os.path.exists(label_file_list):
        file = open(label_file_list, 'r')
        final_lines_list = []
        for line in file.xreadlines():
            line = line.strip('\n')
            print line
            labelFile = getLabelFilename(line)
            print labelFile
            finalLine = parse_line(line, './training/label_2/'+labelFile)
            final_lines_list.append(finalLine)
    result_file = open('./' + savedFilename, 'w')
    result_file.writelines(final_lines_list)
    result_file.close()

if '__main__' == __name__:
    convertKitti('Train_image_list.txt', 'TrainIndex.txt')





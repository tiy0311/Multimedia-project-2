#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob

from PIL import Image
from numpy import array
import sift
from pylab import *

from scipy.spatial import distance
from scipy.fftpack import dct
from operator import itemgetter
from sklearn.cluster import KMeans
import numpy as np

from Tkinter import *
import ttk
from PIL import Image, ImageTk



dst_list_Q1_unsorted = []
dst_list_Q2_unsorted = []
dst_list_Q3_unsorted = []
dst_list_Q4_unsorted = []


def print_similarity(dst_list):
    for x in range(9):
        print "No.0%d --- %s (%s)" % (x+1, dst_list[x][0], dst_list[x][1])
    print "No.10 --- %s (%s)" % (dst_list[9][0], dst_list[9][1])
    print "----------------------------------------------"



#--------#
#   Q1   #
#--------#
def Q1(im_target):
    im_target_histo = im_target.histogram()

    dst_list_Q1 = []
    for fname in glob.glob('dataset/*.jpg'):
        #print fname
        im = Image.open(fname)
        im_histo = im.histogram()

        # Euclidean distance
        #dst = distance.euclidean(im_target_histo, im_histo)

        # Manhattan distance
        dst = distance.cityblock(im_target_histo, im_histo)

        dst_list_Q1.append((fname,dst))

    global dst_list_Q1_unsorted
    dst_list_Q1_unsorted = dst_list_Q1

    # sort by distance
    dst_list_Q1.sort(key=itemgetter(1))
    print "Q1. Using Color Histogram---------------------"
    print_similarity(dst_list_Q1)





#--------#
#   Q2   #
#--------#
def Q2(im_target):
    def average(pixel, x, y, width, height):
        R_list, G_list, B_list = [], [], []
        for X in range(x, x+(width/8)):
            for Y in range(y, y+(height/8)):
                R, G, B = pixel[X,Y]
                R_list.append(R)
                G_list.append(G)
                B_list.append(B)
        return  np.mean(R_list), np.mean(G_list), np.mean(B_list)
                    

    def rgb_ycbcr(avg_list):
        Y_list, CB_list, CR_list = [], [], []
        for x in range(64):
            y = .299*avg_list[x][0] + .587*avg_list[x][1] + .114*avg_list[x][2]
            cb = 128 -.168736*avg_list[x][0] -.331364*avg_list[x][1] + .5*avg_list[x][2]
            cr = 128 +.5*avg_list[x][0] - .418688*avg_list[x][1] - .081312*avg_list[x][2]
            Y_list.append(y)
            CB_list.append(cb)
            CR_list.append(cr)
    
        return Y_list, CB_list, CR_list


    def zigzag(dct_y, dct_cb, dct_cr):
        DY, DCb, DCr = [], [], []
    
        temp = 0
        DY.append(dct_y[temp])
        DCb.append(dct_cb[temp])
        DCr.append(dct_cr[temp])
        for x in range(1,8):
            if (x % 2 != 0):
                temp += 1
                DY.append(dct_y[temp])
                DCb.append(dct_cb[temp])
                DCr.append(dct_cr[temp])
            else:
                temp += 8
                DY.append(dct_y[temp])
                DCb.append(dct_cb[temp])
                DCr.append(dct_cr[temp])
            for y in range(x):
                if (x % 2 == 0):
                    temp -= 7
                    DY.append(dct_y[temp])
                    DCb.append(dct_cb[temp])
                    DCr.append(dct_cr[temp])
                else:
                    temp += 7
                    DY.append(dct_y[temp])
                    DCb.append(dct_cb[temp])
                    DCr.append(dct_cr[temp])
   
        for x in range(8,1,-1):
            if (x % 2 == 0):
                temp += 1
                DY.append(dct_y[temp])
                DCb.append(dct_cb[temp])
                DCr.append(dct_cr[temp])
            else:
                temp += 8
                DY.append(dct_y[temp])
                DCb.append(dct_cb[temp])
                DCr.append(dct_cr[temp])
            for y in range(x-2):
                if (x % 2 == 0):
                    temp -= 7
                    DY.append(dct_y[temp])
                    DCb.append(dct_cb[temp])
                    DCr.append(dct_cr[temp])
                else:
                    temp += 7
                    DY.append(dct_y[temp])
                    DCb.append(dct_cb[temp])
                    DCr.append(dct_cr[temp])

        return DY, DCb, DCr


    def CLD(pixel):
        width, height = im_target.size
        avg_list = []
        for x in range(width):
            for y in range(height):
                if ( (y % (height/8) == 0) and (x % (width/8) == 0)):
                    avg_list.append(average(pixel, x, y, width, height))

        Y_list, Cb_list, Cr_list = [], [], []
        Y_list, Cb_list, Cr_list = rgb_ycbcr(avg_list)

        DCT_Y, DCT_Cb, DCT_Cr = [], [], []
        DCT_Y = dct(Y_list)
        DCT_Cb = dct(Cb_list)
        DCT_Cr = dct(Cr_list)

        DY, DCb, DCr = zigzag(DCT_Y, DCT_Cb, DCT_Cr)

        return DY, DCb, DCr


    # in Q2 main
    pixel = im_target.load()
    input_DY, input_DCb, input_DCr = CLD(pixel)


    dst_list_Q2 = []
    for fname in glob.glob('dataset/*.jpg'):
        #print fname
        im = Image.open(fname)
        pixel = im.load()
        output_DY, output_DCb, output_DCr = CLD(pixel)

        dst_y, dst_cb, dst_cr = .0, .0, .0
        for x in range(64):
            dst_y += .2 * ( (input_DY[x] - output_DY[x]) ** 2 )
            dst_cb += .4 * ( (input_DCb[x] - output_DCb[x]) ** 2 )
            dst_cr += .4 * ( (input_DCr[x] - output_DCr[x]) ** 2 )

        dst = sqrt(dst_y) + sqrt(dst_cb) + sqrt(dst_cr)
        dst_list_Q2.append((fname,dst))

    global dst_list_Q2_unsorted
    dst_list_Q2_unsorted = dst_list_Q2

    #print dst_list_Q2

    # sort by distance
    dst_list_Q2.sort(key=itemgetter(1))
    print "Q2. Using Color Layout------------------------"
    print_similarity(dst_list_Q2)




#--------#
#   Q3   #
#--------#
def Q3(qname):

    QFnumber = qname[18:21]
    QFnumber = int(QFnumber)

    d1_list = []
    d1_length_list = []
    statistics_list = []
    dst_list_Q3 = []
    fnum = 0
    for fname in glob.glob('dataset/*.jpg'):
        #print fname
        fnum += 1
        
        if os.path.isfile(fname + '.sift') == False :
            im = array(Image.open(fname).convert('L'))
            sift.process_image(fname, fname+'.sift')

        l1, d1 = sift.read_features_from_file(fname+'.sift')

#print "d1's len: %s" % len(d1)
        d1_length_list.append(len(d1))

        for x in d1:
            d1_list.append(x)
    

    #print "d1_length_list"
    #print d1_length_list

    #print "---clusters---"
    a = KMeans(n_clusters=100).fit(d1_list)
    #print "a"
    #print len(a.labels_)
    #print a.labels_
    #print "---"


    # Statistic the number of 0, 1, 2 in each file
    counter_0, counter_1, counter_2, start, end = 0, 0, 0, 0, 0
    for f in range(fnum):
        #print "f = %d" % f
        end += d1_length_list[f]
        #print ">>>> %d ~ %d" % (start, end)
        counter_0, counter_1, counter_2 = 0, 0, 0
        for i in range(start, end):
            if a.labels_[i] == 0:
                counter_0 += 1
            elif a.labels_[i] == 1:
                counter_1 += 1
            elif a.labels_[i] == 2:
                counter_2 += 1
        start += d1_length_list[f]
        statistics_list.append((counter_0, counter_1, counter_2))
        #print "0: %d, 1:%d, 2:%d ---- total:%d" % (counter_0, counter_1, counter_2, counter_0+counter_1+counter_2)

    #print statistics_list


    # Querying
    for f in range(fnum):
        #print "f = %d" % f
        dst = np.sqrt((statistics_list[QFnumber][0]-statistics_list[f][0])**2 + (statistics_list[QFnumber][1]-statistics_list[f][1])**2 + (statistics_list[QFnumber][2]-statistics_list[f][2])**2)
        
        if (0 <= f and f < 10):
            fname = "dataset1/ukbench0000"+ str(f)  +".jpg"
        elif (10 <= f and f < 100):
            fname = "dataset1/ukbench000"+ str(f)  +".jpg"
        else:
            fname = "dataset1/ukbench00"+ str(f)  +".jpg"


        #print fname
        dst_list_Q3.append((fname,dst))

    global dst_list_Q3_unsorted
    dst_list_Q3_unsorted = dst_list_Q3

    # sort by distance
    dst_list_Q3.sort(key=itemgetter(1))

    print "Q3. Using SIFT Visual Words-------------------"
    print_similarity(dst_list_Q3)




#--------#
#   Q4   #
#--------#
def Q4(qname):

    QFnumber = qname[18:21]
    QFnumber = int(QFnumber)

    d1_list = []
    d1_length_list = []
    statistics_list = []
    dst_list_Q4 = []
    fnum = 0
    for fname in glob.glob('dataset1/*.jpg'):
        #print fname
        fnum += 1
        
        if os.path.isfile(fname + '.sift') == False :
            im = array(Image.open(fname).convert('L'))
            sift.process_image(fname, fname+'.sift')

        l1, d1 = sift.read_features_from_file(fname+'.sift')

        d1_length_list.append(len(d1))

        for x in d1:
            d1_list.append(x)
    

    #print "d1_length_list"
    #print d1_length_list

    #print "---clusters---"
    a = KMeans(n_clusters=100).fit(d1_list)
    #print "a"
    #print len(a.labels_)
    #print a.labels_
    #print "---"


    # find max
    total_0, total_1, total_2 = 0, 0, 0
    for i in range(len(a.labels_)):
        if a.labels_[i] == 0:
            total_0 += 1
        elif a.labels_[i] == 1:
            total_1 += 1
        elif a.labels_[i] == 2:
            total_2 += 1
    #print "total_0: %d, total_1: %d, total_2: %d" % (total_0, total_1, total_2)

    #print "max *= 0.9"
    ignore = 0
    if( total_0 > total_1 and total_0 > total_2):
        total_0 *= 0.9
        ignore = 0
    elif( total_1 > total_0 and total_1 > total_2):
        total_1 *= 0.9
        ignore = 1
    elif( total_2 > total_0 and total_2 > total_1):
        total_2 *= 0.9
        ignore = 2
    #print "total_0: %d, total_1: %d, total_2: %d" % (total_0, total_1, total_2)



    # Statistic the number of 0, 1, 2 in each file
    counter_0, counter_1, counter_2, start, end = 0, 0, 0, 0, 0
    for f in range(fnum):
        #print "f = %d" % f
        end += d1_length_list[f]
        #print ">>>> %d ~ %d" % (start, end)
        counter_0, counter_1, counter_2 = 0, 0, 0
        for i in range(start, end):
            if a.labels_[i] == 0:
                counter_0 += 1
            elif a.labels_[i] == 1:
                counter_1 += 1
            elif a.labels_[i] == 2:
                counter_2 += 1
        start += d1_length_list[f]
        if ignore == 0:
            counter_0 *= 0.9
        elif ignore == 1:
            counter_1 *= 0.9
        elif ignore == 2:
            counter_2 *= 0.9
        statistics_list.append((counter_0, counter_1, counter_2))
        #print "0: %d, 1:%d, 2:%d ---- total:%d" % (counter_0, counter_1, counter_2, counter_0+counter_1+counter_2)

    #print statistics_list


    # Querying
    for f in range(fnum):
        #print "f = %d" % f
        dst = np.sqrt((statistics_list[QFnumber][0]-statistics_list[f][0])**2 + (statistics_list[QFnumber][1]-statistics_list[f][1])**2 + (statistics_list[QFnumber][2]-statistics_list[f][2])**2)
        
        if (0 <= f and f < 10):
            fname = "dataset1/ukbench0000"+ str(f)  +".jpg"
        elif (10 <= f and f < 100):
            fname = "dataset1/ukbench000"+ str(f)  +".jpg"
        else:
            fname = "dataset1/ukbench00"+ str(f)  +".jpg"


        #print fname
        dst_list_Q4.append((fname,dst))


    global dst_list_Q4_unsorted
    dst_list_Q4_unsorted = dst_list_Q4

    # sort by distance
    dst_list_Q4.sort(key=itemgetter(1))


    print "Q4. Using SIFT Visual Words with Stop Words---"
    print_similarity(dst_list_Q4)




#--------#
#   Q5   #
#--------#
def Q5(qname):
    print "qname =  %s" % qname

    dst_list_Q1, dst_list_Q2, dst_list_Q4 = [], [], []

    dst_list_Q1_unsorted.sort(key=itemgetter(0))
    dst_list_Q2_unsorted.sort(key=itemgetter(0))
    dst_list_Q4_unsorted.sort(key=itemgetter(0))

    for x in range(2, len(dst_list_Q1_unsorted)):
        dst_list_Q1.append((dst_list_Q1_unsorted[x][0],float(dst_list_Q1_unsorted[x][1] / dst_list_Q1_unsorted[1][1])))
        dst_list_Q2.append((dst_list_Q2_unsorted[x][0],float(dst_list_Q2_unsorted[x][1] / dst_list_Q2_unsorted[1][1])))
        dst_list_Q4.append((dst_list_Q4_unsorted[x][0],float(dst_list_Q4_unsorted[x][1] / dst_list_Q4_unsorted[1][1])))

    print "dst_list_Q1"
    print dst_list_Q1
    print "dst_list_Q2"
    print dst_list_Q2
    print "dst_list_Q4"
    print dst_list_Q4

    dst = []
    dst_list_Q5 = []
    for x in range(len(dst_list_Q1)):
        if qname == dst_list_Q1[x][0]:
            dst_list_Q5.append((qname,0.0))
        else:
            dst = (dst_list_Q1[x][1]) + (dst_list_Q2[x][1]) + (dst_list_Q4[x][1])
            dst_list_Q5.append((dst_list_Q1[x][0],dst))
    
    print "dst_list_Q5"
    print dst_list_Q5

    # sort by distance
    dst_list_Q5.sort(key=itemgetter(1))

    print "Q5. Merge All Results-------------------------"
    print_similarity(dst_list_Q5)


# main
if len(sys.argv) < 2:
    print 'please enter query file name.'
    sys.exit()
else:
    qname = sys.argv[1]
    print "----------------------------------------------"
    print "Query: %s" % qname
    print "----------------------------------------------"
    im_target = Image.open(qname)
    Q1(im_target)
    Q2(im_target) 
    Q3(qname)
    Q4(qname)
    Q5(qname)

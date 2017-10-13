#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from pylab import *

def plot_results(res):
    """ Show images in result list 'res'. """

    figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        subplot(1,nbr_results,i+1)
        imshow(array(Image.open(res[i])))
        axis('off')
    show()

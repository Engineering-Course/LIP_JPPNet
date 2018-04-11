from __future__ import division
import math
import random
import scipy.misc
import numpy as np
from scipy.stats import multivariate_normal
import scipy.io as sio 
import csv

csv_file = 'lip_train_set.csv'

with open(csv_file, "r") as input_file:

    for row in csv.reader(input_file):

        img_id = row.pop(0)[:-4]
        print img_id
        
        image_path = './images/{}.jpg'.format(img_id)
        img = scipy.misc.imread(image_path).astype(np.float)
        rows = img.shape[0]
        cols = img.shape[1]
        heatmap_ = np.zeros((rows, cols, 16), dtype=np.float64)
        
        for idx, point in enumerate(row):
            if 'nan' in point:
                point = 0
            if idx % 3 == 0:
                c_ = int(point)
                c_ = min(c_, cols-1)
                c_ = max(c_, 0)
            elif idx % 3 == 1 :
                r_ = int(point)
                r_ = min(r_, rows-1)
                r_ = max(r_, 0)
                if c_ + r_ > 0:
                    var = multivariate_normal(mean=[r_, c_], cov=64)
                    l1 = max(r_-25, 0)
                    r1 = min(r_+25, rows-1)
                    l2 = max(c_-25, 0)
                    r2 = min(c_+25, cols-1)
                    for i in xrange(l1, r1):
                        for j in xrange(l2, r2):
                            heatmap_[i, j, int(idx / 3)] = var.pdf([i, j]) * 400
                save_path = './heatmap/{}_{}.png'.format(img_id, int(idx/3))
                scipy.misc.imsave(save_path, heatmap_[:,:,int(idx/3)])
        heatsum_ = np.sum(heatmap_, axis=2)

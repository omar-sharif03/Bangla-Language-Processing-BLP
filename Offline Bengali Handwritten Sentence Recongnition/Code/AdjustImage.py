import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import cv2
import os
from skimage import io
# load the image from disk

def adjustImage(invert, b):

    #io.imshow(invert);
    xmin, xmax, ymin, ymax = 100000, -1, 100000, -1;
    for j in range(invert.shape[1]):
        for i in range(invert.shape[0]):
            if(invert[i][j] == b):
                #print(i, j, invert[i][j]);
                xmin = min(xmin, i);
                xmax = max(xmax, i);
                ymin = min(ymin, j);
                ymax = max(ymax, j);

    #print(xmin, xmax, ymin, ymax);

    crop = invert[xmin:xmax+1 , ymin:ymax+1];

    #io.imshow(crop);

    return crop;
    #io.imshow(rotated);

#image = cv2.imread('/home/s4k1b/Downloads/skw.jpg');
#procImage = deskew(image);

#io.imshow(procImage);
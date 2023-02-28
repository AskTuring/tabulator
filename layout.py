from utils import *
import numpy as np
import cv2 as cv
import time

def getTableOutlines(bw, debug=False):
    horizontal, vertical, text = np.copy(bw),np.copy(bw),np.copy(bw)
    thicken = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    vertical = cv.dilate(text, thicken, iterations=1)
    if debug:
        show('thickened vertical binary', vertical)

    cols = horizontal.shape[1]
    horizontal_size = cols // 3
    horizontalStruct = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = opening(horizontal, horizontalStruct)
    if debug:
        show('horizontal', horizontal)

    verticalsize = 10
    verticalStruct = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    vertical = opening(vertical, verticalStruct, iters=5)
    if debug:
        show(f'vert at {verticalsize}', vertical)

    verticalStruct = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    vertical = cv.dilate(vertical, verticalStruct, iterations=1)
    if debug:
        show(f'dilation at {verticalsize}', vertical)

    #vertical,horizontal = smoothen(vertical, debug=debug), smoothen(horizontal, debug=debug)
    overlay = vertical+horizontal
    if debug:
        show('overlay', overlay)
    cont, _ = cv.findContours(overlay, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    table_boxes = []
    for c in cont:
        x,y,w,h = cv.boundingRect(c)
        # remove noise
        if w*h > bw.shape[0]*bw.shape[1]/64:
            table_boxes.append([x,y,w,h])
    return table_boxes
'''
Extract edges and smooth image according to the logic
1. extract edges
2. dilate(edges)
3. src.copyTo(smooth)
4. blur smooth img
5. smooth.copyTo(src, edges)
'''
def smoothen(src, debug=False):
    edges = cv.adaptiveThreshold(src, 255, cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY, 3, -2)
    
    kernel = np.ones((2,2), np.uint8)
    edges = cv.dilate(edges, kernel)
    if debug:
        show("dilate", edges)

    smooth = np.copy(src)
    # step 4
    smooth = cv.blur(smooth, (2,2))
    # step 5
    (rows, cols) = np.where(edges != 0)
    src[rows, cols] = smooth[rows, cols] 
    return src



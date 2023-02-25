from c import getTlines, Comp
from l import getTables
from typing import *
import cv2 as cv
import sys
import os
import numpy as np
import time

# TODO make more efficient later
# column by line
def columnByLine(seg, tlines: List[Comp], p=10, debug=False):
    src = np.zeros((seg.shape[0], seg.shape[1]))
    #TODO padding can be inconsistent... bad
    xs = set()
    for tline in tlines:
        for c in tline:
            x,y,w,h = c.x, c.y, c.w, c.h
            if x+p < src.shape[1]: pad = p
            else: pad = src.shape[1]-x-1
            src[y:y+h,x+pad:x+w] = 1
            xs.add(x+pad)
    if debug:
        show('blacked',src)
    
    cols = []
    for i in range(0, src.shape[1]-p, p):
        kernel = src[:,i+p]
        if sum(kernel) == 0:
            cols.append(i)
    
    s = np.copy(seg)
    for col in cols:
        cv.rectangle(s, (col,0), (col+p,src.shape[0]), (0,0,255), 2)
    if debug:
        show('cols', s)
    return cols

# column by histogram
def columnByBucket(seg, tlines: List[List[Comp]], debug=False):
    step = 50
    rlist = [i*step for i in range(seg.shape[1]//step)]
    nlist = [0 for _ in range(len(rlist))]
    llist = [[]for _ in range(len(rlist))]
    matched = set()
    for line in tlines:
        for c in line:
            if c in matched: continue
            for i in range(len(rlist)):
                if i!=0 and c.L < rlist[i] and c.L >= rlist[i-1]: 
                    nlist[i]+=1
                    llist[i].append(c)
                    matched.add(c)
    cols = []
    total = sum(nlist)
    for i in range(len(nlist)):
        x = nlist[i]
        if x / total > 0.10:
            cols.append(llist[i])
    return cols

# good ol line detection
def columnByMorph(bw, tlines, debug=False):
    thicken = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    vertical = cv.dilate(bw, thicken, iterations=1)
    if debug:
        show(f'dilate', vertical)

    vertsize = 10
    verticalstruct = cv.getStructuringElement(cv.MORPH_RECT, (1,vertsize))
    vertical = opening(vertical, verticalstruct, iters=5)
    if debug:
        show(f'vert opening', vertical)
    for _ in range(5):
        vertsize+=5
        verticalstruct = cv.getStructuringElement(cv.MORPH_RECT, (1,vertsize))
        vertical = cv.dilate(vertical, verticalstruct, iterations=1)
    if debug:
        show(f'vert dilate', vertical)
    cnts,_ = cv.findContours(vertical, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cols = []
    for cnt in cnts:
        col = cv.boundingRect(cnt)
        if isColumn(bw, tlines, col):
            cols.append(col)
    return cols

def rowByMorph(bw, tlines, debug=False):
    thicken = cv.getStructuringElement(cv.MORPH_RECT, (3, 2))
    horizontal = cv.dilate(bw, thicken, iterations=1)
    if debug:
        show(f'dilate', horizontal)

    horsize = bw.shape[1]//3
    horstruct = cv.getStructuringElement(cv.MORPH_RECT, (horsize,1))
    horizontal = opening(horizontal, horstruct, iters=1)
    if debug:
        show(f'hor opening', horizontal)
    cnts, _ = cv.findContours(horizontal, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    rows = []
    for cnt in cnts:
        row = cv.boundingRect(cnt)
        if isRow(bw, tlines, row):
            rows.append(row)
    return rows
        
def show(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

# TODO: rule --> column cannot STRIKE through comp
def isColumn(src, tlines: List[List[Comp]], col, thresh=0.33):
    if not isinstance(col, tuple):
        col = tuple(col)
    x,y,w,h = col
    if h < len(src)*0.33: return False
    isLeftMost = True
    isRightMost = True
    for line in tlines:
        f,l = line[0],line[len(line)-1]
        if f.L < x:
            isLeftMost = False 
        if l.L > x:
            isRightMost = False
    midL,midR = src.shape[1]//4, src.shape[1]//4*3
    isInMiddle = x > midL and x < midR
    return (not (isLeftMost or isRightMost)) or (isInMiddle)

def isRow(src, tlines: List[List[Comp]], row, thresh=0.75):
    if not isinstance(row, tuple):
        row = tuple(row)
    x,y,w,h = row
    if w < src.shape[1]*0.75: return False
    isTop = True
    isBottom = True
    smallest = max(tlines[0],key=lambda x: x.T)
    greatest = max(tlines[len(tlines)-1], key=lambda x:x.B)
    if smallest.T < y:
        isTop = False
    if greatest.B > y:
        isBottom = False
    midT, midB = src.shape[0]//4, src.shape[0]//4*3
    isInMiddle = y > midT and y < midB
    return (not (isTop or isBottom)) or (isInMiddle)


def opening(src, struct, iters):
    src = cv.erode(src, struct, iterations=iters)
    src = cv.dilate(src, struct, iterations=iters)
    return src

def drawCols(src, cols):
    src = np.copy(src)
    for col in cols:
        x,y,w,h = tuple(col)
        cv.rectangle(src,(x,0),(x+1,src.shape[0]),(0,0,255),2)
    return src

def drawRows(src, rows):
    src = np.copy(src)
    for row in rows:
        x,y,w,h = tuple(row)
        cv.rectangle(src,(0,y),(src.shape[1],y+1),(0,0,255),2)
    return src

def main(f):
    src = cv.imread(f, cv.IMREAD_COLOR)
    tables = getTables(f)
    for table in tables:
        x,y,w,h = tuple(table)
        seg = src[y:y+h,x:x+w,:]
        print('-'*50)
        print(seg.shape)
        gray = cv.cvtColor(seg, cv.COLOR_BGR2GRAY)
        gray = cv.bitwise_not(gray)
        bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, 17, -2)
        tlines = getTlines(seg)
        cols = columnByMorph(bw, tlines, debug=False)
        rows = rowByMorph(bw, tlines, debug=True)
        seg = drawCols(seg, cols)
        seg = drawRows(seg, rows)
        show('cells', seg)

# TODO:
'''
    - merge textlines to form contiguous "text blocks"
    - count rows again
    - OCR on cells 

'''

TESTDIR1 = '/Users/minjunes/tabulator/data/simatic-st70-complete-english-2022.pdf'
TESTDIR2 ='/Users/minjunes/tabulator/data/simatic-st70-complete-english-2022.pdf[555:558]'
TESTDIR3 ='/Users/minjunes/tabulator/data/s71200_system_manual_en-US_en-US.pdf[1032:1036]'

if __name__ == '__main__':
    argv = sys.argv[1:]
    f = argv[0]
    if f == 'test':
        for f in os.listdir(TESTDIR1):
            f = os.path.join(TESTDIR1, f)
            main(f)
    else:
        main(f)

    
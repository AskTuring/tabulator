import cv2 as cv
import numpy as np
import math
from typing import *
import multiprocessing
import logging
import torch

class Box:
    def __init__(self, x,y,w,h, text=None):
        self.x,self.y,self.w,self.h = x,y,w,h
        self.L = x
        self.R = x + w
        self.T = y 
        self.B = y + h
        self.List = [self.x, self.y, self.w, self.h]
        self.A = self.w*self.h
        self.cx = x+(w/2)
        self.cy = y+(h/2)
        self.text = text
    def __repr__(self):
        return f'(x:{self.x},y:{self.y},w:{self.w},h:{self.h})'
    def __getitem__(self,key):
        if not (key >= 0 and key < 4): raise IndexError 
        return self.List[key]
    def __iter__(self):
        yield from self.List

# cv ops
def getPaddedSeg(src,x,y,w,h,p=5):
    p = p//2
    x = x-p if x-p >= 0 else 0
    y = y-p if y-p >= 0 else 0
    r = x+w+p if x+w+p < src.shape[1] else src.shape[1]
    t = y+h+p if y+h+p < src.shape[0] else src.shape[0]
    return src[y:t,x:r,:]

def pad(src,x,y,w,h,p=20):
    seg = src[y:y+h,x:x+w,:]
    new = (np.ones((seg.shape[0]+p*2,seg.shape[1]+p*2,seg.shape[2]))*255).astype(src.dtype)
    #print(seg.shape, new.shape)
    new[p:p+h,p:p+w,:] = seg
    return new

def opening(src, struct, iters=1):
    src = cv.erode(src, struct, iterations=iters)
    src = cv.dilate(src, struct, iterations=iters)
    return src

def closing(src, struct, iters=1):
    src = cv.dilate(src, struct, iterations=iters)
    src = cv.erode(src, struct, iterations=iters)
    return src

def getBwOtsu(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _,bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    bw = cv.bitwise_not(bw)
    return bw

def getBwGauss(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, 17, -2)
    return bw

def seg(src,x,y,w,h):
    if len(src.shape)>2:
        return src[y:y+h,x:x+w,:]        
    return src[y:y+h,x:x+w]

# algs
def percentile(L, percent):
    N = len(L)
    p = 1.0/percent
    if N//p % 2 == 0:
        l,h = math.floor(N/p), math.ceil(N/p)
        return (L[l] + L[h])/2
    return L[math.floor(N/p)]

def F(c1: Box,c2: Box, scale=0.5):
    c1B, c2B = c1.y+c1.h*scale, c2.y+c2.h*scale
    return c1.T <= c2B and c1B >= c2.T

def D(c1: Box, c2: Box):
    return c2.L - c1.R

def rasterScan(L: List[Box]) -> List[List[Box]]:
    ysorted = sorted(L, key=lambda x:x.cy)
    if not ysorted: return [[]]
    if len(ysorted)<2: return [L]
    lines,line=[],[]
    i,j=0,1
    while j<len(ysorted):
        c1,c2=ysorted[i],ysorted[j]
        if not line:
            line.append(c1)        
        if F(c1,c2):
            line.append(c2)
            j+=1
        else:
            lines.append(sorted(line, key=lambda x:x.cx))
            line=[]
            i=j
            j+=1
    # case 1 
    # F(ysorted[i], ysorted[j-1]) == True
    # line
    # sort, done
    if line:
        lines.append(sorted(line, key=lambda x:x.cx))
    # F(ysorted[i], ysorted[j-1]) == False 
    # not line
    # add j-1 to line, done
    else:
        lines.append([ysorted[j-1]])
    return lines

# Draws
def drawStats(src, stats):
    for i in range(len(stats)):
        x,y,w,h,_  = stats[i][0],stats[i][1],stats[i][2],stats[i][3],stats[i][4]
        cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

def drawBoxes(src, boxes):
    for i in range(len(boxes)):
        x,y,w,h  = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
        cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

def drawCols(src, cols):
    for col in cols:
        x,_,_,_ = col
        cv.rectangle(src,(x,0),(x+1,src.shape[0]),(0,255,0),2)

def drawRows(src, rows):
    for row in rows:
        _,y,_,_ = row
        cv.rectangle(src,(0,y),(src.shape[1],y+1),(0,255,0),2)

def drawOcrTable(src, boxes: List[Box]):
    for box in boxes:
        x,y,w,h = tuple(box)
        cv.rectangle(src, (x,y), (x+w,y+h), (0,0,255),3)

def show(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

# conversions
def ocr2Boxes(table, hcut=1):
    boxes = []
    for res in table:
        box,text = res[0],res[1]
        x,y = tuple(box[0])
        w,h = (abs(box[0][0]-box[1][0]), abs(box[1][1]-box[2][1]))
        boxes.append(Box(int(x),int(y),int(w),int(h*hcut), text=text)) 
    return boxes

def stats2Boxes(stats, hcut=1):
    boxes = []
    for stat in stats:
        x,y,w,h,_ = tuple(stat)
        boxes.append(Box(int(x),int(y),int(w),int(h*hcut)))
    return boxes

def stretch(xss):
    return [x for xs in xss for x in xs]

def concatText(L: List[List[Box]]):
    res = ''
    for line in L:
        for box in line:
            res += box.text + ' '
    if res: return res[:-1]
    return ''

logger = logging.getLogger(__name__)
def detect_gpus():
    cpu_count = multiprocessing.cpu_count()
    logger.info("{} CPUs detected".format(cpu_count))

    try:
        gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    except:
        gpus = []

    if len(gpus) == 0:
        logger.info("No GPU detected.")
    else:
        logger.info("{} GPUs detected".format(len(gpus)))

    return gpus
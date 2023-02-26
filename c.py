# connected component analysis + https://www.cse.iitb.ac.in/~sharat/icvgip.org/icvgip2004/proceedings/ip2.1_156.pdf
# what is guassian filtering: https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import math
import time
from typing import *
import numpy as np

class Comp:
    def __init__(self, s):
        self.x = s[0] 
        self.y = s[1]
        self.w = s[2]
        self.h = s[3]
        self.L = s[0]
        self.R = s[0] + s[2]
        self.T = s[1] 
        self.B = s[1] + s[3] 
        self.List = [self.x, self.y, self.w, self.h]
        self.A = self.w*self.h
    def __repr__(self):
        return f'(x:{self.x},y:{self.y},w:{self.w},h:{self.h})'
    def __getitem__(self,key):
        if not (key >= 0 and key < 4): raise IndexError 
        return self.List[key]
    def __iter__(self):
        yield from self.List

def stretch(xss):
    return [x for xs in xss for x in xs]

def show(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def drawStats(src, stats, title='drawStats'):
    for i in range(len(stats)):
        x,y,w,h,_  = stats[i][0],stats[i][1],stats[i][2],stats[i][3],stats[i][4]
        cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)
    show(title,src)
    return src

def drawComps(src, comps, title='drawComps'):
    src = np.copy(src)
    for i in range(len(comps)):
        x,y,w,h  = comps[i][0],comps[i][1],comps[i][2],comps[i][3]
        cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)
    show(title,src)
    return src

def opening(src, struct, iters=1):
    src = cv.erode(src, struct, iterations=iters)
    src = cv.dilate(src, struct, iterations=iters)
    return src

def closing(src, struct, iters=1):
    src = cv.dilate(src, struct, iterations=iters)
    src = cv.erode(src, struct, iterations=iters)
    return src

# TODO something wrong with getPeaks, returning None
def getPeaks(freqs):
    peaks = []
    for i in range(len(freqs)):
        if len(peaks) == 2:
            return peaks
        if i != 0 and i < len(freqs)-1:
            if freqs[i]>freqs[i-1] and freqs[i]>freqs[i+1]:
                peaks.append(freqs[i])
    if not peaks:
        return None
    if len(peaks) != 2:
        peaks.append(peaks[0])
        return peaks
        
# get upper limit of hump given its peak
def getUpperVal(freqs, peakIdx):
    for i in range(peakIdx, len(freqs)):
        if freqs[i] <= freqs[peakIdx]:
            peakIdx = i
        else:
            return peakIdx
    return len(freqs)-1

def getV(ds, debug=False):
    # x-freq
    # y-mag
    ds = sorted(ds)
    mags = list(set(ds))
    freqs = []
    c = 1
    for i in range(len(ds)):
        if i < len(ds)-1 and ds[i]==ds[i+1]:
            c+=1
        else:
            freqs.append(c)
            c = 1
    assert(len(freqs) == len(mags))
    peaks = getPeaks(freqs)
    if not peaks:
        return None
    nextHighestPeak = peaks[1]
    upperVal = getUpperVal(freqs, freqs.index(nextHighestPeak))
    if debug:
        plt.bar(list(mags), freqs, align='center', alpha=0.5)
        plt.axvline(x=upperVal, color='b')
        plt.show()
    return upperVal

def percentile(L, percent):
    N = len(L)
    p = 1.0/percent
    if N//p % 2 == 0:
        l,h = math.floor(N/p), math.ceil(N/p)
        return (L[l] + L[h])/2
    return L[math.floor(N/p)]
# same line
def F(c1: Comp,c2: Comp):
    return c1.T <= c2.B and c1.B >= c2.T

# distance function
# REQUIRES: F(c1,c2) = True AND c2.L > c1.R
def D(c1: Comp, c2: Comp):
    return c2.L - c1.R

# minimum vert distance between two text lines
def VD(v1: List[Comp], v2: List[Comp]):
    c2 = max(v2, key=lambda x: x.T)
    c1 = min(v1, key=lambda x: x.B)
    return c2.T - c1.B

# useless...
def VDS(tlines: List[List[Comp]]):
    longest = len(sorted(tlines, key=lambda x:len(x))[-1])
    grid: List[List[Comp]] = []
    for tline in tlines:
        grid.append(tline + [None]*(longest-len(tline)))
    
    ds = []
    for i in range(len(grid)-1):
        for j in range(longest):
            if grid[i][j]:
                if grid[i+1][j]:
                    ds.append(abs(grid[i][j].B-grid[i+1][j].T))
            else:
                continue
    return ds

def VDS2(tlines):
    ds = []
    for i in range(len(tlines)-1):
        for j in range(1):
            if tlines[i]:
                ds.append(abs(tlines[i][j].B-tlines[i+1][j].T))
    return ds

# all consecutive distances between ci, cj where j > i AND F(ci, cj)
# LEGACY
def allDistances(L):
    area = sorted(L, key=lambda x: x[4])
    ysorted = sorted(area[:-1], key=lambda x: x[1])
    ds = []
    for i in range(len(ysorted)):
        c1 = Comp(ysorted[i])
        lowest = float('inf')
        for j in range(i, len(ysorted)):
            c2 = Comp(ysorted[j])
            if F(c1,c2):
                d = D(c1,c2)
                if d >= 0 and d < lowest: lowest = d
            else:
                break
        if lowest != float('inf'): ds.append(lowest)
    return ds

# input -> textlines
def getDistances(L):
    ds = []
    for line in L:
        for i in range(len(line)):
            if i < len(line)-1:
                ds.append(D(line[i], line[i+1]))
    return ds

# raster scan to get text lines
def rasterScan(L) -> List[List[Comp]]:
    # remove entire bounding box
    area = sorted(L, key=lambda x: x[4])
    ysorted = sorted(area[:-1], key=lambda x: x[1])
    if not ysorted:
        return []
    elif len(ysorted)==1:
        return [[Comp(ysorted[0])]]
    lines,line = [],[]
    i = 0
    j = 1
    while j < len(ysorted):
        c1,c2 = Comp(ysorted[i]), Comp(ysorted[j])
        if F(c1,c2): 
            line.append(c2)
            j+=1
        else:
            line=[c1]+line
            lines.append(sorted(line, key=lambda x:x.L))
            line=[]
            i,j=j,j+1
    if not line:
        line = [c2]
    line = [c1] + line
    lines.append(sorted(line, key=lambda x:x.L))
    return lines

def getBwOtsu(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _,bw = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    bw = cv.bitwise_not(bw)
    return bw

def getBwGauss(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, 17, -2)
    gray = cv.bitwise_not(gray)
    return bw
                
def getTlines(src, debug=False):
    bw = getBwOtsu(src)
    if debug:
        show('bw', bw)    

    thicken = cv.getStructuringElement(cv.MORPH_RECT, (2,3))
    bw = cv.dilate(bw, thicken, iterations=1)
    if debug:
        show('thickened', bw)

    _, _, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(
        bw,
        8,
        cv.CV_32S,
        cv.CCL_DEFAULT
    )

    ws = [s[2] for s in stats]
    hs = [s[3] for s in stats]
    w,h = int(percentile(sorted(ws), 0.03)), int(percentile(sorted(hs), 0.04))
    vertstruct = cv.getStructuringElement(cv.MORPH_RECT, (1,h))
    horstruct = cv.getStructuringElement(cv.MORPH_RECT, (w,1))
    bw = opening(bw, vertstruct, iters=1)
    bw = opening(bw, horstruct, iters=1)

    # first pass detecting all characters
    _, _, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(
        bw,
        8,
        cv.CV_32S,
        cv.CCL_DEFAULT
    )
    tlines = rasterScan(stats)
    if not tlines: return []
    ds = getDistances(tlines)
    v = getV(ds) 
    if not v: return []
    v+=5
    h=8

    # TODO: 
    # --> we can see how well tesseract
    # groups words together

    vclose = cv.getStructuringElement(cv.MORPH_RECT, (v,h))
    bw = closing(bw, vclose, iters=1)
    if debug:
        show('clustering text', bw)

    # extractling coalesced words
    _, _, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(
        bw,
        8,
        cv.CV_32SC1,
        cv.CCL_DEFAULT
    )

    tlines = rasterScan(stats)
    #drawComps(src, stretch(tlines), title='allTlines')
    return tlines

if __name__ == '__main__':
    argv = sys.argv[1:]
    f = argv[0]
    global debug
    debug = argv[1] if len(argv) > 1 and int(argv[1])>0 else False
    getTlines(cv.imread(f, cv.IMREAD_COLOR))
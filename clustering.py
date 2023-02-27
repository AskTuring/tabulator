# connected component analysis + https://www.cse.iitb.ac.in/~sharat/icvgip.org/icvgip2004/proceedings/ip2.1_156.pdf
# what is guassian filtering: https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm
import cv2 as cv
import matplotlib.pyplot as plt
from typing import *
from utils import *
from dotenv import load_dotenv
import sys
import os
load_dotenv()

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

def getV(ds: List, debug=False):
    if not ds: return None
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
    if not peaks: return None
    nextHighestPeak = peaks[1]
    upperVal = getUpperVal(freqs, freqs.index(nextHighestPeak))
    if debug:
        plt.bar(list(mags), freqs, align='center', alpha=0.5)
        plt.axvline(x=upperVal, color='b')
        plt.show()
    return upperVal

# input -> textlines
def getDistances(L: List[List[Box]]):
    ds = []
    for line in L:
        for i in range(len(line)):
            if i < len(line)-1:
                ds.append(D(line[i], line[i+1]))
    return ds

def getTextLines(src, debug=False) -> List[List[Box]]:
    if len(src.shape) != 2:
        src = getBwOtsu(src)
    if debug:
        show('bw', src)    
    thicken = cv.getStructuringElement(cv.MORPH_RECT, (2,3))
    bw = cv.dilate(src, thicken, iterations=1)
    if debug:
        show('thickened', bw)
    _, _, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(
        bw,
        8,
        cv.CV_32S,
        cv.CCL_DEFAULT
    )
    ws,hs = [s[2] for s in stats],[s[3] for s in stats]
    w,h = int(percentile(sorted(ws), 0.03)), int(percentile(sorted(hs), 0.04))
    vertstruct = cv.getStructuringElement(cv.MORPH_RECT, (1,h))
    horstruct = cv.getStructuringElement(cv.MORPH_RECT, (w,1))
    bw = opening(bw, vertstruct)
    bw = opening(bw, horstruct)
    _, _, stats, _ = cv.connectedComponentsWithStatsWithAlgorithm(
        bw,
        8,
        cv.CV_32S,
        cv.CCL_DEFAULT
    )
    tlines = rasterScan(stats2Boxes(stats))
    ds = getDistances(tlines)
    v,h = getV(ds), int(os.getenv('clusterh'))
    if not v: return []
    v += int(os.getenv('clusterv'))
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

    tlines = rasterScan(stats2Boxes(stats))
    return tlines

if __name__ == '__main__':
    argv = sys.argv[1:]
    f = argv[0]
    global debug
    debug = argv[1] if len(argv) > 1 and int(argv[1])>0 else False
    getTextLines(cv.imread(f, cv.IMREAD_COLOR))
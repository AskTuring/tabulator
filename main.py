from clustering import getTextLines, Box, F
from layout import getTableOutlines
from typing import *
from utils import *
import easyocr
import cv2 as cv
import sys
import os
import time
import csv
import torch

reader = easyocr.Reader(
    ['en'], 
    gpu=torch.cuda.is_available(), 
    quantize=torch.cuda.is_available()
    ) 

def useEasyOcr(src):
    return reader.readtext(
        src,
        batch_size=16,
        width_ths=0.7,
    )
# good ol line detection
def columnByMorph(bw, tlines: List[List[Box]], debug=False) -> List[tuple]:
    thicken = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    vertical = cv.dilate(bw, thicken)
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
        vertical = cv.dilate(vertical, verticalstruct)
    if debug:
        show(f'vert dilate', vertical)
    cnts,_ = cv.findContours(vertical, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cols = []
    for cnt in cnts:
        col = tuple(cv.boundingRect(cnt))
        if isColumn(bw, tlines, col):
            cols.append(col)
    return cols

def rowByMorph(bw, tlines: List[List[Box]], debug=False) -> List[tuple]:
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
        row = tuple(cv.boundingRect(cnt))
        if isRow(bw, tlines, row):
            rows.append(row)
    return rows

# TODO: rule --> column cannot STRIKE through comp
def isColumn(src, tlines: List[List[Box]], col: tuple, thresh=0.33):
    x,_,_,h = col
    if h < len(src)*thresh: return False
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

def isRow(src, tlines: List[List[Box]], row: tuple, thresh=0.75):
    _,y,w,_ = row
    if w < src.shape[1]*thresh: return False
    isTop = True
    isBottom = True
    smallest = min(tlines[0],key=lambda x: x.T)
    greatest = max(tlines[len(tlines)-1], key=lambda x:x.B)
    if smallest.T < y:
        isTop = False
    if greatest.B > y:
        isBottom = False
    midT, midB = src.shape[0]//4, src.shape[0]//4*3
    isInMiddle = y > midT and y < midB
    return (not (isTop or isBottom)) or (isInMiddle)

def addRows(tlines: List[List[Box]], rows: List[tuple]):
    cndRows = [max(line, key=lambda x:x.B) for line in tlines]
    newrows = []
    for cnd in cndRows:
        newrows.append(tuple([0,cnd.B,0,0]))
    rows.extend(newrows)

def fillCells(src, rows, cols, boxes):
    rows = [(0,0,0,0)] + sorted(rows) + [(0,src.shape[0],0,0)]
    cols = [(0,0,0,0)] + sorted(cols) + [(src.shape[1],0,0,0)]
    table = []
    unmatched = set(boxes)
    for i in range(len(rows)-1):
        row = []
        startY = rows[i][1]
        stopY = rows[i+1][1]
        for j in range(len(cols)-1):
            startX = cols[j][0]
            stopX = cols[j+1][0]
            x,y,w,h = startX,startY,stopX-startX,stopY-startY
            cell = []
            matched = set()
            for box in unmatched:
                if (box.cx > x and box.cx < x+w and box.cy > y and box.cy < y+h):
                    matched.add(box) 
                    cell.append(box)
            unmatched -= matched
            inOrder = rasterScan(cell)
            content = concatText(inOrder)
            row.append(content)
        if ''.join(row): table.append(row)
    return table

def saveTable(table, title, idx=None):
    suffix = '' if not idx else f'table_no_{idx}'
    with open(title+suffix+'.csv', 'w') as f:
        thisTable = csv.writer(f)
        for row in table:
            thisTable.writerow(row)
    
def extract(f, debug=True):
    src, title = cv.imread(f, cv.IMREAD_COLOR), os.path.splitext(''.join(f.split('/')[-2:]))[0]
    bwotsu, bwgauss = getBwOtsu(src), getBwGauss(src)
    print('='*20, f'{title}', '='*20)
    s = time.time()
    tables = getTableOutlines(bwgauss, debug=True)
    e = time.time()
    print(f'getTable time: {e-s}')
    for i,table in enumerate(tables):
        x,y,w,h = tuple(table)
        seg_rgb, seg_bwgauss, seg_bwotsu =seg(src,x,y,w,h), seg(bwgauss,x,y,w,h), seg(bwotsu,x,y,w,h) 
        print('-'*20, f'table {i}', '-'*20)
        s = time.time()
        tlines = getTextLines(seg_bwotsu, debug=True)
        e = time.time()
        tlinesTime = e-s
        print('tlines time:',tlinesTime)
        if tlines:
            cols = columnByMorph(seg_bwgauss, tlines, debug=False)
            rows = rowByMorph(seg_bwgauss, tlines, debug=False)
            addRows(tlines, rows)
            drawCols(seg_rgb, cols)
            drawRows(seg_rgb, rows)
            if debug:
                show('drawn', seg_rgb)
            s = time.time()
            boxes = ocr2Boxes(useEasyOcr(seg_rgb))
            e = time.time()
            ocrTime = e-s
            print('ocr time:',ocrTime)
            s = time.time()
            table = fillCells(seg_rgb, rows, cols, boxes)
            e = time.time()
            fillCellTime = e-s
            print('fill time:',fillCellTime)
            print(f'saving table {f}')
            saveTable(table, title, idx=i)
# TODO:
'''
    post-processing: merge "subheader" cells

'''

ROOT = os.getcwd()
TESTDIR1 = os.path.join(ROOT,'data/simatic-st70-complete-english-2022.pdf')
TESTDIR2 = os.path.join(ROOT, 'data/s71200_system_manual_en-US_en-US.pdf')
TESTDIR3 =os.path.join(ROOT,'data/s71200_system_manual_en-US_en-US.pdf[1032:1036]')
if __name__ == '__main__':
    argv = sys.argv[1:]
    f = argv[0]
    if f == 'test':
        for f in os.listdir(TESTDIR1)[::-1]:
            f = os.path.join(TESTDIR1, f)
            extract(f)
    else:
        extract(f)

    
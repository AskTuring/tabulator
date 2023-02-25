import numpy as np
import sys 
import cv2 as cv
import os

def show(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def getTables(f, debug=False):
    src = cv.imread(f, cv.IMREAD_COLOR)
    if debug:
        cv.imshow('src', src)
    # transform image to grayscale
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    if debug:
        show('gray', gray)
    # adaptive threshold at bitwise_not of gray
    # TODO: why does Gaussian work? Can something else work better? 
    gray = cv.bitwise_not(gray) 
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY, 17, -2)
    #_,bw = cv.threshold(gray, 128, 255, cv.THRESH_OTSU)
    if debug: 
        show("base binary", bw)

    horizontal, vertical, text = np.copy(bw),np.copy(bw),np.copy(bw)
    thicken = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    vertical = cv.dilate(text, thicken, iterations=1)

    if debug:
        show('thickened vertical binary', vertical)

    cols = horizontal.shape[1]
    horizontal_size = cols // 3
    horizontalStruct = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    # opening --> erosion followed by dilating to remove "noise"
    # in this case all that is not horizontal lines 
    # TODO: does this remove text information --> will removal be useful?
    horizontal = cv.erode(horizontal, horizontalStruct)
    horizontal = cv.dilate(horizontal, horizontalStruct)
    if debug:
        show('horizontal', horizontal)

    # [vert]
    # TODO: tweak with verticalsize
    verticalsize = 10
    step = 1
    verticalsize += step
    verticalStruct = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    # TODO; tweak iterations
    vertical = opening(vertical, verticalStruct, iters=5)
    if debug:
        show(f'vert at {verticalsize}', vertical)

    verticalStruct = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    vertical = cv.dilate(vertical, verticalStruct, iterations=1)
    if debug:
        show(f'dilation at {verticalsize}', vertical)


    #vertical, horizontal = cv.bitwise_not(vertical),cv.bitwise_not(horizontal)
    # combine vertical and horizontal
    # TODO: what is the purpose of smoothening? 
    vertical,horizontal = smoothen(vertical, debug=debug), smoothen(horizontal, debug=debug)
    overlay = vertical+horizontal
    if debug:
        show('overlay', overlay)
    cont, _ = cv.findContours(overlay, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

   
    table_boxes = []
    for c in cont:
        x, y, w, h = cv.boundingRect(c)
        # remove noise
        if w*h > src.shape[0]*src.shape[1]/64:
            cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)
            table_boxes.append([x,y,w,h])
    if debug:
        show('boxes', src) 
    return table_boxes


def opening(src, struct, iters):
    src = cv.erode(src, struct, iterations=iters)
    src = cv.dilate(src, struct, iterations=iters)
    return src

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

'''
observations: 

need to use table detection to draw "outline" around boxes
for contour detection to work.

TODO:
    - detect multiple tables and get its contours
    - try:
        - extracting hor,vert lines inside table regions
        - extracting hor,vert lines first then concating with table outline
    
    - pick whatever works qualitatively best
    - perform OCR based on contour cell bounding box
    - perform sub-row detection using OCR bounding box info

'''

TESTDIR = '/Users/minjunes/tabulator/data/simatic-st70-complete-english-2022.pdf'

if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        exit()
    if argv[0] == 'test':
        for f in os.listdir(TESTDIR):
            f = os.path.join(TESTDIR, f)
            getTables(f, debug=False)
        exit()

    if len(argv) >= 2:
        debug = argv[1]
    else:
        debug = False

    getTables(argv[0], debug=debug)

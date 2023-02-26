import easyocr
import cv2 as cv
import numpy as np

TEST = '/Users/minjunes/tabulator/data/s71200_system_manual_en-US_en-US.pdf/1.jpg'
def show(name, img):
    cv.imshow(name, img)
    cv.moveWindow(name, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(name)


reader = easyocr.Reader(['en'], gpu=True, quantize=True) # this needs to run only once to load the model into memory
import time
def useEasyOcr(src):
    result = reader.readtext(src)
    r = ''
    for res in result:
        if res[2] > 0.5:
            r += res[1] + " "
        #print(f'word: {res[1]}, conf: {res[2]}')
    return r[:-1]
    

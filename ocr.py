from PIL import Image
import pytesseract as tes
from pytesseract import Output
import sys
import cv2 as cv
import numpy as np

TEST = '/Users/minjunes/tabulator/data/s71200_system_manual_en-US_en-US.pdf/1.jpg'
def show(img, name='default'):
    cv.imshow(name, img)
    cv.moveWindow(name, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(name)

def run(argv):
    if argv:
        f = argv[-1] 
    else:
        f = TEST

    img = cv.imread(f)    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    #show(gray, name='smoothen')
    results = tes.image_to_data(gray, output_type=Output.DICT)

    t = 0
    n = 0
    for i in range(0, len(results['text'])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        
        # We will also extract the OCR text itself along
        # with the confidence of the text localization
        text = results["text"][i]
        conf = int(results["conf"][i])
        
        # filter out weak confidence text localizations
        if conf > 0.9:
            
            # We will display the confidence and text to
            # our terminal
            print("Confidence: {}".format(conf))
            t += conf
            n+=1
            print("Text: {}".format(text))
            print("")
            
            # We then strip out non-ASCII text so we can
            # draw the text on the image We will be using
            # OpenCV, then draw a bounding box around the
            # text along with the text itself
            text = "".join(text).strip()
            cv.rectangle(img,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255), 2)
            
        # After all, we will show the output image
    print(t/n)
    show(img)


if __name__ == '__main__':
    run(sys.argv[1:])


from __future__ import print_function
import cv2
import numpy as np
from crop import crop
from scan import scan
import pytesseract
from PIL import Image
import argparse
from test_accuracy import test_accuracy

def recognize(imgname, output, desired,
              show_intermediate_results=False):
    scan(imgname, show_intermediate_results)
    im = cv2.imread('deskewed.jpg')
    im = cv2.dilate(im, np.ones((2, 2)))
    newimgname = 'no_noise.jpg'
    cv2.imwrite(newimgname, im)
    crop(newimgname, 'scan_res.jpg', show_intermediate_results)
    recognized_text = pytesseract.image_to_string(Image.open('scan_res.jpg'), config="config")
    with open(output, 'w+') as f:
        print(recognized_text, file=f)
    print('Accuracy: ' + str(test_accuracy(scan_res=output, desired=desired)))


if __name__ == '__main__':
    #python .\recognize.py -i photos\chom4.jpg -c texts\chom.txt -o output.txt
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, default='photos/chom3.jpg',
                    help="Path to the image to be scanned")
    ap.add_argument("-o", "--output", required=False, default='output.txt',
                    help="Path for the output text file")
    ap.add_argument("-c", "--check", required=False,  default='texts/chom.txt',
                    help="Path to the file with reference text")
    ap.add_argument("-s", "--show", required=False, default=True,
                    help="Show intermediate results", dest='show', action='store_true')
    args = vars(ap.parse_args())
    recognize(imgname=args['image'],
              output=args['output'],
              desired=args['check'],
              show_intermediate_results=args['show'])

import cv2
import argparse
import numpy as np
from scipy.ndimage import convolve

def parser_args():
    parser = argparse.ArgumentParser(description = 'Blurring')

    parser.add_argument('input_image', type=str)
    parser.add_argument('kernel', type=str)
    parser.add_argument('output_image', type=str)
    parser.add_argument('noise_level', type=float)

    args = parser.parse_args()
    return args

def make_blurred(im, kernel, noise_level=5):
    blurred = convolve(im, kernel) + np.random.normal(0, noise_level, im.shape)
    return blurred

args = parser_args()
im = np.array(cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)).astype('float')
kernel = np.array(cv2.imread(args.kernel, cv2.IMREAD_GRAYSCALE)).astype('float')
kernel /= kernel.sum()
noise = args.noise_level
im_blurred = make_blurred(im, kernel, noise)
cv2.imwrite(args.output_image, im_blurred.astype('int'))
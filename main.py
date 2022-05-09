import argparse
import cv2
import numpy as np
from scipy.ndimage import convolve, shift

def parser_args():
    parser = argparse.ArgumentParser(description = 'Deconvolution')

    parser.add_argument('input_image', type=str)
    parser.add_argument('kernel', type=str)
    parser.add_argument('output_image', type=str)
    parser.add_argument('noise_level', type=float)

    args = parser.parse_args()
    return args

def deblur(u, A, alpha, iterations=100):

    def grad1(z, A):
        return convolve(convolve(z, A), A[::-1, ::-1])  # A_T * A * z

    def grad_tv(z):
        dx = np.sign(shift(z, (0, 1)) - z)
        dx = shift(dx, (0, -1)) - dx
        dy = np.sign(shift(z, (1, 0)) - z)
        dy = shift(dy, (-1, 0)) - dy
        return dx + dy

    mu=0.1
    z = u
    v = np.zeros(u.shape)
    if alpha < 0.1: alpha = 0.1
    grad2 = convolve(u, A[::-1, ::-1])  # A_T * u
    for _ in range(iterations):
        grad = 2 * (grad1(z, A) - grad2) + alpha * grad_tv(z)
        v = mu * v - grad
        z += v
        z[z < 0] = 0
        z[z > 255] = 255
    return z

args = parser_args()
im_blurred = np.array(cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)).astype('float')
kernel = np.array(cv2.imread(args.kernel, cv2.IMREAD_GRAYSCALE)).astype('float')
# the A is normalized so that the sum of the coefficients becomes equal to 1
kernel /= kernel.sum()
im_deblurred = deblur(im_blurred, kernel, args.noise_level / 5)
cv2.imwrite(args.output_image, im_deblurred.astype('int'))

'''
ref = cv2.imread('test.bmp', cv2.IMREAD_GRAYSCALE)
blurred = cv2.imread(args.input_image, cv2.IMREAD_GRAYSCALE)
deblurred = cv2.imread(args.output_image, cv2.IMREAD_GRAYSCALE)
print('psnr min = ', cv2.PSNR(ref, blurred))
print('my_psnr = ', cv2.PSNR(ref, deblurred))
'''
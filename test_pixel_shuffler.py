# -*- coding: utf-8 -*-
import chainer
import numpy
import cv2
import srcgan.models

a = chainer.Variable(numpy.arange(32).reshape(2, 2, 8))


def img2variable(img):
    return chainer.Variable(numpy.array([img.transpose(2, 0, 1)], dtype=numpy.float32))


def clip_img(x):
    return numpy.uint8(0 if x < 0 else (255 if x > 255 else x))
    # return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def variable2img(x):
    img = (numpy.vectorize(clip_img)(x.data[0, :, :, :])).transpose(1, 2, 0)
    return img


import sys

original_img = cv2.imread(sys.argv[1])
img = numpy.dstack((original_img, original_img[:, :, :1]))
img = numpy.dstack((img, img, img))
img = numpy.dstack((original_img, original_img, original_img, original_img))

img_var = img2variable(img)
widened = srcgan.models.pixel_shuffle_upscale(img_var)
widened_img = variable2img(widened)
print(img.shape)
print(widened_img.shape)
print(widened_img[100, 100])
print(widened_img[100, 101])
print(widened_img[100, 102])

cv2.imshow("test", original_img)
cv2.imshow("test2", widened_img)
cv2.waitKey(-1)

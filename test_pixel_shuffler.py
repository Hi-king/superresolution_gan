# -*- coding: utf-8 -*-
import chainer
import numpy
import functools
import cv2

a = chainer.Variable(numpy.arange(32).reshape(2, 2, 8))


def pixel_shuffle_upscale(x: chainer.Variable):
    def channel_to_axis(x: chainer.Variable, axis):
        channels = chainer.functions.separate(x, axis=1)
        result_channel = int(len(channels) / 2)
        w1, w2 = chainer.functions.stack(channels[:result_channel], axis=1), chainer.functions.stack(
            channels[result_channel:], axis=1)
        odds, evens = chainer.functions.separate(w1, axis=axis), chainer.functions.separate(w2, axis=axis)
        width_widened = chainer.functions.stack(
            functools.reduce(lambda x, y: x + y, ([a, b] for a, b in zip(odds, evens)))
            , axis=axis)
        return width_widened

    return channel_to_axis(channel_to_axis(img_var, 2), 3)


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
widened = pixel_shuffle_upscale(img_var)
widened_img = variable2img(widened)
print(img.shape)
print(widened_img.shape)
print(widened_img[100, 100])
print(widened_img[100, 101])
print(widened_img[100, 102])

cv2.imshow("test", original_img)
cv2.imshow("test2", widened_img)
cv2.waitKey(-1)


# print(a.data.shape)
# channels = chainer.functions.separate(width_widened, axis=2)
# w1, w2 = chainer.functions.stack(channels[:result_channel], axis=2), chainer.functions.stack(channels[result_channel:], axis=2)
# print(w1.data.shape)
# print(w2.data.shape)
# odds, evens = chainer.functions.separate(w1, axis=1), chainer.functions.separate(w1, axis=1)
# print(odds[0].data.shape)
# width_widened = chainer.functions.stack(
#     functools.reduce(lambda x, y: x+y, ([a, b] for a, b in zip(odds, evens)))
# , axis=1)
# print(width_widened.data.shape)

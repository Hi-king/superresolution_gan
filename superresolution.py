# -*- coding: utf-8 -*-
import argparse
import chainer
import numpy
import cv2
import srcgan
from PIL import Image


def img2variable(img):
    return chainer.Variable(numpy.array([img.transpose(2, 0, 1)], dtype=numpy.float32))


def clip_img(x):
    return numpy.uint8(0 if x < 0 else (255 if x > 255 else x))
    # return numpy.float32(-1 if x < -1 else (1 if x > 1 else x))


def variable2img(x):
    print(x.data.max())
    print(x.data.min())
    img = (numpy.vectorize(clip_img)(x.data[0, :, :, :])).transpose(1, 2, 0)
    return img


def resize_copy(img):
    dst = numpy.zeros((img.shape[0] * 4, img.shape[1] * 4, img.shape[2]), dtype=img.dtype)
    for x in range(4):
        for y in range(4):
            dst[x::4, y::4, :] = img
    return dst


parser = argparse.ArgumentParser()
parser.add_argument("--modelpath", required=True)
parser.add_argument("--imagepath", required=True)
parser.add_argument("--outputpath")
parser.add_argument("--resizedpath")
parser.add_argument("--bicubicpath")
args = parser.parse_args()
generator = srcgan.models.SRGenerator()
chainer.serializers.load_npz(args.modelpath, generator)
# chainer.serializers.load_hdf5(args.modelpath, generator)

# img = numpy.asarray(Image.open(args.imagepath).resize((96, 96)), dtype=numpy.float32)
img = numpy.asarray(Image.open(args.imagepath), dtype=numpy.float32)
if img.shape[2] == 4:
    img = img[:, :, :3]


img_variable = img2variable(img)
img_variable_sr = generator(img_variable, test=True)
img_sr = variable2img(img_variable_sr)

resized = cv2.cvtColor(resize_copy(img), cv2.COLOR_RGB2BGR)
bicubic = cv2.cvtColor(cv2.resize(img / 256, (img.shape[1]*4, img.shape[0]*4)), cv2.COLOR_RGB2BGR)
if args.outputpath is None:
    cv2.imshow("test", cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))
    cv2.imshow("test2", resized)
    cv2.imshow("test3", bicubic)
    cv2.waitKey(-1)
else:
    cv2.imwrite(args.outputpath, cv2.cvtColor(img_sr, cv2.COLOR_RGB2BGR))
    if args.resizedpath is not None:
        cv2.imwrite(args.resizedpath, resized)
    if args.bicubicpath is not None:
        cv2.imwrite(args.bicubicpath, bicubic)

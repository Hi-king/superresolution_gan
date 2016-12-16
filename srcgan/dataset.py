# -*- coding: utf-8 -*-
import os
import six
import numpy
import math
import cv2
import random
from PIL import Image
from chainer.dataset import dataset_mixin


class PILImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None, root='.'):
        if isinstance(paths, six.string_types):
            with open(paths) as paths_file:
                paths = [path.strip() for path in paths_file]
        self._paths = paths
        self._root = root
        self._resize = resize

    def __len__(self):
        return len(self._paths)

    def get_example(self, i) -> Image:
        path = os.path.join(self._root, self._paths[i])
        original_image = Image.open(path)
        if not self._resize is None:
            return original_image.resize(self._resize)
        else:
            return original_image


class ResizedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, resize=None, root='.', dtype=numpy.float32):
        self.base = PILImageDataset(paths=paths, resize=resize, root=root)
        self._dtype = dtype

    def __len__(self):
        return len(self.base)

    def get_example(self, i) -> numpy.ndarray:
        image = self.base[i]
        image_ary = numpy.asarray(image, dtype=self._dtype)
        if len(image_ary.shape) == 2: # mono
            image_ary = numpy.dstack((image_ary, image_ary, image_ary))
        image_data = image_ary.transpose(2, 0, 1)
        if image_data.shape[0] == 4:  # RGBA
            image_data = image_data[:3]
        return image_data


class PreprocessedImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, cropsize, resize=None, root='.', dtype=numpy.float32):
        self.base = ResizedImageDataset(paths=paths, resize=resize, root=root)
        self._dtype = dtype
        self.cropsize = cropsize

    def __len__(self):
        return len(self.base)

    def get_example(self, i) -> numpy.ndarray:
        image = self.base[i]
        x = random.randint(0, image.shape[1] - self.cropsize)
        y = random.randint(0, image.shape[2] - self.cropsize)

        cropeed_high_res = image[:, x:x + self.cropsize, y:y + self.cropsize]
        cropped_low_res = cv2.resize(cropeed_high_res.transpose(1, 2, 0), dsize=(int(self.cropsize/4), int(self.cropsize/4)),
                                     interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
        return cropped_low_res, cropeed_high_res

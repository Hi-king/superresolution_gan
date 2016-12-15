# -*- coding: utf-8 -*-
import chainer
import math


class SRGeneratorResBlock(chainer.Chain):
    def __init__(self):
        super().__init__(
            c1=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn1=chainer.links.BatchNormalization(96 * 96 * 64),
            c2=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn2=chainer.links.BatchNormalization(96 * 96 * 64),
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = chainer.functions.ReLU(self.bn1(self.c1(x), test=test))
        h = self.bn2(self.c2(h))
        return h + x  # residual


class SRGenerator(chainer.Chain):
    def __init__(self, inputdim=100):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            res1=SRGeneratorResBlock(),
            res2=SRGeneratorResBlock(),
            res3=SRGeneratorResBlock(),
            res4=SRGeneratorResBlock(),
            res5=SRGeneratorResBlock()
        )

    def __call__(self, z, test=False):
        h = chainer.functions.reshape(chainer.functions.relu(self.bn0l(self.l0z(z), test=test)),
                                      (z.data.shape[0], 512, 6, 6))
        h = chainer.functions.relu(self.bn1(self.dc1(h), test=test))
        h = chainer.functions.relu(self.bn2(self.dc2(h), test=test))
        h = chainer.functions.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x

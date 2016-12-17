# -*- coding: utf-8 -*-
import functools
import chainer
import chainer.links.caffe
import math


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

    return channel_to_axis(channel_to_axis(x, 2), 3)


class SRGeneratorResBlock(chainer.Chain):
    def __init__(self):
        super().__init__(
            c1=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn1=chainer.links.BatchNormalization(64),
            c2=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn2=chainer.links.BatchNormalization(64),
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = chainer.functions.relu(self.bn1(self.c1(x), test=test))
        h = self.bn2(self.c2(h))
        return h + x  # residual


class SRGeneratorUpScaleBlock(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv=chainer.functions.Convolution2D(in_channels=64, out_channels=256, ksize=3, stride=1, pad=1,
                                                 wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable):
        h = self.conv(x)
        h = pixel_shuffle_upscale(h)
        h = chainer.functions.relu(h)
        return h


class SRGenerator(chainer.Chain):
    def __init__(self):
        super().__init__(
            first=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            res1=SRGeneratorResBlock(),
            res2=SRGeneratorResBlock(),
            res3=SRGeneratorResBlock(),
            res4=SRGeneratorResBlock(),
            res5=SRGeneratorResBlock(),
            conv_mid=chainer.links.Convolution2D(64, 64, ksize=3, stride=1, pad=1, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn_mid=chainer.links.BatchNormalization(64),
            upscale1=SRGeneratorUpScaleBlock(),
            upscale2=SRGeneratorUpScaleBlock(),
            conv_output=chainer.links.Convolution2D(64, 3, ksize=3, stride=1, pad=1,
                                                    wscale=0.02 * math.sqrt(64 * 3 * 3))
        )

    def __call__(self, x: chainer.Variable, test=False):
        h = first = chainer.functions.relu(self.first(x))

        h = self.res1(h, test=test)
        h = self.res2(h, test=test)
        h = self.res3(h, test=test)
        h = self.res4(h, test=test)
        h = self.res5(h, test=test)
        mid = self.bn_mid(self.conv_mid(h), test=test)

        h = first + mid

        h = self.upscale2(self.upscale1(h))

        h = self.conv_output(h)
        return h


class SRDiscriminator(chainer.Chain):
    def __init__(self):
        super().__init__(
            conv_input=chainer.links.Convolution2D(3, 64, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(3 * 3 * 3)),
            c1=chainer.links.Convolution2D(64, 64, ksize=3, stride=2, pad=0, wscale=0.02 * math.sqrt(64 * 3 * 3)),
            bn1=chainer.links.BatchNormalization(64),
            c2=chainer.links.Convolution2D(64, 128, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            bn2=chainer.links.BatchNormalization(128),
            c3=chainer.links.Convolution2D(128, 128, ksize=3, stride=2, pad=0, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            bn3=chainer.links.BatchNormalization(128),
            c4=chainer.links.Convolution2D(128, 256, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(128 * 3 * 3)),
            bn4=chainer.links.BatchNormalization(256),
            c5=chainer.links.Convolution2D(256, 256, ksize=3, stride=2, pad=0, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            bn5=chainer.links.BatchNormalization(256),
            c6=chainer.links.Convolution2D(256, 512, ksize=3, stride=1, pad=0, wscale=0.02 * math.sqrt(256 * 3 * 3)),
            bn6=chainer.links.BatchNormalization(512),
            c7=chainer.links.Convolution2D(512, 512, ksize=3, stride=2, pad=0, wscale=0.02 * math.sqrt(512 * 3 * 3)),
            bn7=chainer.links.BatchNormalization(512),
            linear1=chainer.links.Linear(in_size=4608, out_size=1024),
            linear2=chainer.links.Linear(in_size=None, out_size=2),
        )

    def __call__(self, x, test=False):
        h = self.conv_input(x)
        h = self.bn1(chainer.functions.elu(self.c1(h)), test=test)
        h = self.bn2(chainer.functions.elu(self.c2(h)), test=test)
        h = self.bn3(chainer.functions.elu(self.c3(h)), test=test)
        h = self.bn4(chainer.functions.elu(self.c4(h)), test=test)
        h = self.bn5(chainer.functions.elu(self.c5(h)), test=test)
        h = self.bn6(chainer.functions.elu(self.c6(h)), test=test)
        h = self.bn7(chainer.functions.elu(self.c7(h)), test=test)
        h = chainer.functions.elu(self.linear1(h))
        h = chainer.functions.sigmoid(self.linear2(h))
        return h


class VGG(object):
    def __init__(self, caffemodelpath):
        self.model = chainer.links.caffe.CaffeFunction(caffemodelpath)

    def forward_layers(self, x, stages=4, average_pooling=False):
        if average_pooling:
            pooling = lambda x: chainer.functions.average_pooling_2d(chainer.functions.relu(x), 2, stride=2)
        else:
            pooling = lambda x: chainer.functions.max_pooling_2d(chainer.functions.relu(x), 2, stride=2)

        ret = []
        y1 = self.model.conv1_2(chainer.functions.relu(self.model.conv1_1(x)))
        ret.append(y1)
        if stages == 0: return ret
        x1 = pooling(y1)

        y2 = self.model.conv2_2(chainer.functions.relu(self.model.conv2_1(x1)))
        ret.append(y2)
        if stages == 1: return ret
        x2 = pooling(y2)

        y3 = self.model.conv3_3(
            chainer.functions.relu(self.model.conv3_2(chainer.functions.relu(self.model.conv3_1(x2)))))
        ret.append(y3)
        if stages == 2: return ret
        x3 = pooling(y3)

        y4 = self.model.conv4_4(chainer.functions.relu(self.model.conv4_3(
            chainer.functions.relu(self.model.conv4_2(chainer.functions.relu(self.model.conv4_1(x3)))))))
        ret.append(y4)
        if stages == 3: return ret
        x4 = pooling(y4)
        y5 = self.model.conv5_4(chainer.functions.relu(self.model.conv5_3(
            chainer.functions.relu(self.model.conv5_2(chainer.functions.relu(self.model.conv5_1(x4)))))))
        ret.append(y5)
        return [y1, y2, y3, y4, y5]

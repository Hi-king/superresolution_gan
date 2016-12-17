# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os
import pickle
import chainer
import numpy

import srcgan


def load_vgg(modelpath) -> srcgan.models.VGG:
    modelname = 'vgg'
    cachepath = "{}.dump".format(modelname)
    if os.path.exists(cachepath):
        nn = pickle.load(open(cachepath, "rb"))
    else:
        nn = srcgan.models.VGG(modelpath)
        with open(cachepath, "wb+") as f:
            pickle.dump(nn, f, 0)
    return nn


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--outdirname", required=True)
parser.add_argument("--use_discriminator", action="store_true")
parser.add_argument("--pretrained_generator")
parser.add_argument("--k_adversarial", type=float, default=1)
parser.add_argument("--k_mse", type=float, default=1)
parser.add_argument("--vgg")
parser.add_argument("--vgg_stage", type=int, default=4)
args = parser.parse_args()

OUTPUT_DIRECTORY = args.outdirname
os.makedirs(OUTPUT_DIRECTORY)

logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

logging.info(args)
if args.pretrained_generator is not None:
    logging.info("pretrained_generator: {}".format(os.path.abspath(args.pretrained_generator)))

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

# paths = glob.glob("{}/*.JPEG".format(args.dataset))
paths = glob.glob(args.dataset)
dataset = srcgan.dataset.PreprocessedImageDataset(paths=paths, cropsize=96, resize=(300, 300))

iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)
# iterator = chainer.iterators.SerialIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

generator = srcgan.models.SRGenerator()
if args.pretrained_generator is not None:
    chainer.serializers.load_npz(args.pretrained_generator, generator)
if args.gpu >= 0:
    generator.to_gpu()

if args.use_discriminator:
    discriminator = srcgan.models.SRDiscriminator()
    if args.gpu >= 0:
        discriminator.to_gpu()
    optimizer_discriminator = chainer.optimizers.Adam()
    optimizer_discriminator.setup(discriminator)

if args.vgg is not None:
    vgg = load_vgg(args.vgg)
    if args.gpu >= 0:
        vgg.model.to_gpu()

optimizer_generator = chainer.optimizers.Adam()
optimizer_generator.setup(generator)

count_processed = 0
sum_loss_generator, sum_loss_generator_adversarial, sum_loss_generator_content = 0, 0, 0
for zipped_batch in iterator:
    low_res = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
    high_res = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))
    super_res = generator(low_res)

    if args.use_discriminator:
        discriminated_from_super_res = discriminator(super_res)
        discriminated_from_high_res = discriminator(high_res)
        loss_generator_adversarial = chainer.functions.softmax_cross_entropy(
            discriminated_from_super_res,
            chainer.Variable(xp.zeros(discriminated_from_super_res.data.shape[0], dtype=xp.int32))
        )
        if args.vgg is None:
            loss_generator_content = chainer.functions.mean_squared_error(
                super_res,
                high_res
            )
        else:
            loss_generator_content = chainer.functions.mean_squared_error(
                vgg.forward_layers(super_res, stages=args.vgg_stage)[args.vgg_stage],
                vgg.forward_layers(high_res, stages=args.vgg_stage)[args.vgg_stage]
            )

        loss_generator = loss_generator_content * args.k_mse + loss_generator_adversarial * args.k_adversarial
        sum_loss_generator_adversarial += chainer.cuda.to_cpu(loss_generator_adversarial.data)
        sum_loss_generator_content += chainer.cuda.to_cpu(loss_generator_content.data)

        loss_discriminator = chainer.functions.softmax_cross_entropy(
            discriminated_from_super_res,
            chainer.Variable(xp.ones(discriminated_from_super_res.data.shape[0], dtype=xp.int32))
        ) + chainer.functions.softmax_cross_entropy(
            discriminated_from_high_res,
            chainer.Variable(xp.zeros(discriminated_from_high_res.data.shape[0], dtype=xp.int32))
        )

        optimizer_generator.zero_grads()
        loss_generator.backward()
        optimizer_generator.update()

        optimizer_discriminator.zero_grads()
        loss_discriminator.backward()
        optimizer_discriminator.update()
    else:
        loss_generator = chainer.functions.mean_squared_error(
            super_res,
            high_res
        )
        optimizer_generator.zero_grads()
        loss_generator.backward()
        optimizer_generator.update()
    sum_loss_generator += chainer.cuda.to_cpu(loss_generator.data)

    report_span = args.batchsize * 10
    save_span = args.batchsize * 1000
    count_processed += len(super_res.data)
    if count_processed % report_span == 0:
        logging.info("processed: {}".format(count_processed))
        # logging.info("accuracy_discriminator: {}".format(sum_accuracy * batchsize / report_span))
        # logging.info("accuracy_classifier: {}".format(sum_accuracy_classifier * batchsize / report_span))
        # logging.info("loss_classifier: {}".format(sum_loss_classifier / report_span))
        # logging.info("loss_discriminator: {}".format(sum_loss_discriminator / report_span))
        logging.info("loss_generator: {}".format(sum_loss_generator / report_span))
        logging.info("loss_generator_adversarial: {}".format(sum_loss_generator_adversarial / report_span))
        logging.info("loss_generator_mse: {}".format(sum_loss_generator_content / report_span))
        sum_loss_generator, sum_loss_generator_adversarial, sum_loss_generator_content = 0, 0, 0
    if count_processed % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "generator_model_{}.npz".format(count_processed)), generator)

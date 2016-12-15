# -*- coding: utf-8 -*-
import argparse
import glob
import logging
import os

import chainer
import numpy

import srcgan

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batchsize", type=int, default=10)
parser.add_argument("--outdirname", required=True)
args = parser.parse_args()

OUTPUT_DIRECTORY = args.outdirname
os.makedirs(OUTPUT_DIRECTORY)

logging.basicConfig(filename=os.path.join(OUTPUT_DIRECTORY, "log.txt"), level=logging.DEBUG)
console = logging.StreamHandler()
logging.getLogger('').addHandler(console)

logging.info(args)

if args.gpu >= 0:
    chainer.cuda.check_cuda_available()
    chainer.cuda.get_device(args.gpu).use()
    xp = chainer.cuda.cupy
else:
    xp = numpy

paths = glob.glob("{}/*.png".format(args.dataset))
dataset = srcgan.dataset.PreprocessedImageDataset(paths=paths, cropsize=96, resize=(300, 300))

iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=args.batchsize, repeat=True, shuffle=True)

generator = srcgan.models.SRGenerator()
if args.gpu >= 0:
    generator.to_gpu()


optimizer_generator = chainer.optimizers.Adam()
optimizer_generator.setup(generator)

count_processed = 0
sum_loss_generator = 0
for zipped_batch in iterator:
    low_res = chainer.Variable(xp.array([zipped[0] for zipped in zipped_batch]))
    high_res = chainer.Variable(xp.array([zipped[1] for zipped in zipped_batch]))

    super_res = generator(low_res)
    loss = chainer.functions.mean_squared_error(
        super_res,
        high_res
    )
    sum_loss_generator += chainer.cuda.to_cpu(loss.data)

    optimizer_generator.zero_grads()
    loss.backward()
    optimizer_generator.update()


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
        sum_loss_discriminator, sum_loss_generator, sum_accuracy, sum_loss_classifier, sum_accuracy_classifier = 0, 0, 0, 0, 0
    if count_processed % save_span == 0:
        chainer.serializers.save_npz(
            os.path.join(OUTPUT_DIRECTORY, "generator_model_{}.npz".format(count_processed)), generator)

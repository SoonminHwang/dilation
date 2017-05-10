#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import caffe
from caffe.proto import caffe_pb2
import os
from os.path import dirname, exists, join
import subprocess

import network
import stat

__author__ = 'Fisher Yu'
__copyright__ = 'Copyright (c) 2016, Fisher Yu'
__email__ = 'i@yf.io'
__license__ = 'MIT'


def make_solver(options):
    solver = caffe_pb2.SolverParameter()

    solver.train_net = options.train_net
    if options.test_net is not None:
        solver.test_net.append(options.test_net)
        solver.test_iter.append(50)
    solver.test_interval = 100
    solver.base_lr = options.lr
    solver.lr_policy = "step"
    solver.gamma = 0.1
    # solver.stepsize = 100000
    solver.stepsize = 20000
    solver.display = 5
    # solver.max_iter = 400000
    solver.max_iter = 50000
    solver.momentum = options.momentum
    solver.weight_decay = 0.0005
    solver.regularization_type = 'L2'
    solver.snapshot = 2000
    solver.solver_mode = solver.GPU
    solver.iter_size = options.iter_size
    solver.snapshot_format = solver.BINARYPROTO
    solver.type = 'SGD'
    solver.snapshot_prefix = options.snapshot_prefix

    return solver


def make_frontend_vgg(options, is_training):
    batch_size = options.train_batch if is_training else options.test_batch
    image_path = options.train_image if is_training else options.test_image
    label_path = options.train_label if is_training else options.test_label
    net = caffe.NetSpec()
    net.data, net.label = network.make_image_label_data(
        image_path, label_path, batch_size,
        is_training, options.crop_size, options.mean)
    last = network.build_frontend_vgg(
        net, net.data, options.classes)[0]
    if options.up:
        net.upsample = network.make_upsample(last, options.classes)
        last = net.upsample
    net.loss = network.make_softmax_loss(last, net.label)
    if not is_training:
        net.accuracy = network.make_accuracy(last, net.label)
    return net.to_proto()


def make_context(options, is_training):
    batch_size = options.train_batch if is_training else options.test_batch
    image_path = options.train_image if is_training else options.test_image
    label_path = options.train_label if is_training else options.test_label
    net = caffe.NetSpec()
    net.data, net.label = network.make_bin_label_data(
        image_path, label_path, batch_size,
        options.label_shape, options.label_stride)
    # last = network.build_context(
    #     net, net.data, options.classes, options.layers)[0]
    last = network.build_context_large(
        net, net.data, options.classes, options.layers)[0]
    if options.up:
        net.upsample = network.make_upsample(last, options.classes)
        last = net.upsample
    net.loss = network.make_softmax_loss(last, net.label)
    if not is_training:
        net.accuracy = network.make_accuracy(last, net.label)
    return net.to_proto()


def make_joint(options, is_training):
    batch_size = options.train_batch if is_training else options.test_batch
    image_path = options.train_image if is_training else options.test_image
    label_path = options.train_label if is_training else options.test_label
    net = caffe.NetSpec()
    net.data, net.label = network.make_image_label_data(
        image_path, label_path, batch_size,
        is_training, options.crop_size, options.mean)
    last = network.build_frontend_vgg(
        net, net.data, options.classes)[0]
    last = network.build_context_large(
        net, last, options.classes, options.layers)[0]
    # last = network.build_context(
    #     net, last, options.classes, options.layers)[0]
    if options.up:
        net.upsample = network.make_upsample(last, options.classes)
        last = net.upsample
    net.loss = network.make_softmax_loss(last, net.label)
    if not is_training:
        net.accuracy = network.make_accuracy(last, net.label)
    return net.to_proto()

def make_joint_bn(options, is_training):
    batch_size = options.train_batch if is_training else options.test_batch
    image_path = options.train_image if is_training else options.test_image
    label_path = options.train_label if is_training else options.test_label
    net = caffe.NetSpec()
    net.data, net.label = network.make_image_label_data_bn(
        image_path, label_path, batch_size,
        is_training, options.crop_size, options.mean)
    last = network.build_frontend_vgg19_bn(
        net, net.data, options.classes)[0]
    last = network.build_context_large_bn(
        net, last, options.classes, options.layers)[0]
    # last = network.build_context(
    #     net, last, options.classes, options.layers)[0]
    if options.up:
        net.upsample = network.make_upsample(last, options.classes)
        last = net.upsample
    net.loss = network.make_softmax_loss(last, net.label)
    if not is_training:
        net.accuracy = network.make_accuracy(last, net.label)
    return net.to_proto()

def make_net(options, is_training):
    return globals()['make_' + options.model](options, is_training)


def make_nets(options):
    train_net = make_net(options, True)
    if options.test_net is None:
        test_net = None
    else:
        test_net = make_net(options, False)
    return train_net, test_net


def process_options(options):
    assert (options.crop_size - 372) % 8 == 0, \
        "The crop size must be a multiple of 8 after removing the margin"
    assert len(options.mean) == 3

    assert options.model == 'context' or options.weights is not None, \
        'Pretrained weights are required for frontend and joint training.'

    assert options.model != 'context' or \
        (options.label_shape is not None and
         len(options.label_shape) == 2), \
        'Please specify the height and weight of label images ' \
        'for computing the loss.'

    assert exists(options.train_image), options.train_image + 'does not exist'
    assert exists(options.train_label), options.train_label + 'does not exist'
    assert exists(options.test_image), options.test_image + 'does not exist'
    assert exists(options.test_label), options.test_label + 'does not exist'

    if options.model == 'frontend':
        options.model += '_vgg'

    # work_dir = "jobs/{}/{}/".format(options.dataset, options.model)
    post_fix = '_' + options.post_fix if not options.post_fix == '' else ''    
    work_dir = os.path.join('jobs', options.dataset, options.model + post_fix)    
    options.work_dir = work_dir

    # work_dir = options.work_dir
    model = options.model
    if not exists(work_dir):
        print('Creating working directory', work_dir)
        os.makedirs(work_dir)
    options.train_net = join(work_dir, model + '_train_net.txt')
    if options.test_batch > 0:
        options.test_net = join(work_dir, model + '_test_net.txt')
    else:
        options.test_net = None
    options.solver_path = join(work_dir, model + '_solver.txt')
    snapshot_dir = join(work_dir, 'snapshots')
    if not exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    options.snapshot_prefix = join(snapshot_dir, model)

    if options.up:
        options.label_stride = 1
    else:
        options.label_stride = 8

    if options.lr == 0:
        if options.model == 'frontend_vgg':
            options.lr = 0.0001
        elif options.model == 'context':
            options.lr = 0.001
        elif options.model == 'joint':
            options.lr = 0.00001

    if options.momentum == 0:
        options.momentum = 0.9

    return options


def train(options):
    import os
    # model_name = 'vgg'
    # job_dir = "jobs/{}/{}/".format(options.dataset, model_name)
    # job_dir = "jobs/{}/{}/".format(options.dataset, options.model)
    # job_file = "{}/train.sh".format(job_dir)

    job_dir = options.work_dir
    job_file = "{}/train.sh".format(job_dir)

    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    max_iter = 0
    snapshot_dir = "{}/snapshots".format(job_dir)
    # Find most recent snapshot.
    for file in os.listdir(snapshot_dir):
      if file.endswith(".solverstate"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(options.model))[1])
        if iter > max_iter:
          max_iter = iter

    train_src_param = '--weights="{}" \\\n'.format(options.weights)
    if options.resume:
        if max_iter > 0:
            train_src_param = '--snapshot="{}/{}_iter_{}.solverstate" \\\n'.format(snapshot_dir, options.model, max_iter)
   
    # Create job file.
    with open(job_file, 'w') as f:
        f.write('{} train \\\n'.format(options.caffe))      
        f.write('--solver="{}" \\\n'.format(options.solver_path))        
        # f.write('--weights="{}" \\\n'.format(options.weights))
        f.write(train_src_param)
        if options.resume:
            f.write('--gpu {} 2>&1 | tee -a {}/train_{}.log\n'.format(options.gpu, job_dir, options.model))
        else:
            f.write('--gpu {} 2>&1 | tee {}/train_{}.log\n'.format(options.gpu, job_dir, options.model))

            import glob, shutil
            parsed_log_file = glob.glob("{}*.train".format(job_dir))
            if len(parsed_log_file) > 0 and os.path.exists(parsed_log_file[0]):
                old_dir = "{}old_log".format(job_dir)
                if not os.path.exists(old_dir):
                    os.makedirs(old_dir)
                shutil.copy(parsed_log_file[0], old_dir)
                os.remove(parsed_log_file[0])
            parsed_log_file = glob.glob("{}*.test".format(job_dir))
            if len(parsed_log_file) > 0 and os.path.exists(parsed_log_file[0]):
                old_dir = "{}old_log".format(job_dir)
                if not os.path.exists(old_dir):
                    os.makedirs(old_dir)
                shutil.copy(parsed_log_file[0], old_dir)
                os.remove(parsed_log_file[0])
    
    os.chmod(job_file, stat.S_IRWXU)
    subprocess.call(job_file, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?',
                        choices=['frontend', 'context', 'joint', 'joint_bn'])
    parser.add_argument('--caffe', default='caffe',
                        help='Path to the caffe binary compiled from '
                             'https://github.com/fyu/caffe-dilation.')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='DB name for creating job directory')
    parser.add_argument('--post_fix', type=str, default='default',
                        help='Post fix for creating job directory')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If true, resume training from latest solverstate.')
    parser.add_argument('--weights', default=None,
                        help='Path to the weights to initialize the model.')
    parser.add_argument('--mean', nargs='*', type=float,
                        default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    # parser.add_argument('--work_dir', default='training/',
    #                     help='Working dir for training.\nAll the generated '
    #                          'network and solver configurations will be '
    #                          'written to this directory, in addition to '
    #                          'training snapshots.')
    parser.add_argument('--train_image', default='', required=True,
                        help='Path to the training image list')
    parser.add_argument('--train_label', default='', required=True,
                        help='Path to the training label list')
    parser.add_argument('--test_image', default='',
                        help='Path to the testing image list')
    parser.add_argument('--test_label', default='',
                        help='Path to the testing label list')
    parser.add_argument('--train_batch', type=int, default=8,
                        help='Training batch size.')
    parser.add_argument('--test_batch', type=int, default=2,
                        help='Testing batch size. If it is 0, no test phase.')
    parser.add_argument('--crop_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0,
                        help='Solver SGD learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Gradient momentum')
    parser.add_argument('--classes', type=int, required=True,
                        help='Number of categories in the data')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU index for training')
    parser.add_argument('--up', action='store_true',
                        help='If true, upsampling the final feature map '
                             'before calculating the loss or accuracy')
    parser.add_argument('--layers', type=int, default=8,
                        help='Used for training context module.\n'
                             'Number of layers in the context module.')
    parser.add_argument('--label_shape', nargs='*', type=int,
                        help='Used for training context module.\n' \
                             'The dimensions of labels for the loss function.')
    parser.add_argument('--iter_size', type=int, default=1,
                        help='Number of passes/batches in each iteration.')

    options = process_options(parser.parse_args())

    train_net, test_net = make_nets(options)
    solver = make_solver(options)
    print('Writing', options.train_net)
    with open(options.train_net, 'w') as fp:
        fp.write(str(train_net))
    if test_net is not None:
        print('Writing', options.test_net)
        with open(options.test_net, 'w') as fp:
            fp.write(str(test_net))
    print('Writing', options.solver_path)
    with open(options.solver_path, 'w') as fp:
        fp.write(str(solver))
    train(options)


if __name__ == '__main__':
    main()

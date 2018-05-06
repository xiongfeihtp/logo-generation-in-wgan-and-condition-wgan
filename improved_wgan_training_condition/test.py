'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: test.py
@time: 2018/5/3 下午3:47
@desc: shanghaijiaotong university
'''
import os, sys

sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf
import tflib as lib
from tflib.picture_ops import build_inputs
from configuration import ModelConfig
from graph_handler import GraphHandler

config = ModelConfig()
DATA_DIR = config.data_dir
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

MODE = config.mode  # dcgan, wgan, wgan-gp, lsgan
DIM = config.dim  # Model dimensionality
CRITIC_ITERS = config.critic_iters  # How many iterations to train the critic for

BATCH_SIZE = config.batch_size  # Batch size. Must be a multiple of N_GPUS
ITERS = config.iters  # How many iterations to train for
LAMBDA = config.lamada  # Gradient penalty lambda hyperparameter

OUTPUT_DIM = config.output_dim  # Number of pixels in each iamge
EVAL_NUMBERS = config.eval_number
lib.print_model_settings(locals().copy())
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    images, labels, _ = build_inputs(config, True)
    session.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    queue_runner = tf.train.start_queue_runners(sess=session, coord=coord)
    for _ in range(10):
        images_val = session.run(images)
        print(images_val)
    coord.request_stop()
    coord.join(queue_runner)
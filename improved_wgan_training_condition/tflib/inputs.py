# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def parse_sequence_example(serialized, image_feature, label_feature, filename_feature):
    """Parses a tensorflow.SequenceExample into an image and caption.

    Args:
      serialized: A scalar string Tensor; a single serialized SequenceExample.
      image_feature: Name of SequenceExample context feature containing image
        data.
      caption_feature: Name of SequenceExample feature list containing integer
        captions.

    Returns:
      encoded_image: A scalar string Tensor containing a JPEG encoded image.
      caption: A 1-D uint64 Tensor with dynamically specified length.
    """
    context, _ = tf.parse_single_sequence_example(
        serialized,
        context_features={
            image_feature: tf.FixedLenFeature([], dtype=tf.string),
            label_feature: tf.FixedLenFeature([], dtype=tf.int64),
            filename_feature: tf.FixedLenFeature([], dtype=tf.string)
        })

    encoded_image = context[image_feature]
    label = context[label_feature]
    image_name = context[filename_feature]

    return encoded_image, label, image_name


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=16,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
    """Prefetches string values from disk into an input queue.

    In training the capacity of the queue is important because a larger queue
    means better mixing of training examples between shards. The minimum number of
    values kept in the queue is values_per_shard * input_queue_capacity_factor,
    where input_queue_memory factor should be chosen to trade-off better mixing
    with memory usage.

    Args:
      reader: Instance of tf.ReaderBase.
      file_pattern: Comma-separated list of file patterns (e.g.
          /tmp/train_data-?????-of-00100).
      is_training: Boolean; whether prefetching for training or eval.
      batch_size: Model batch size used to determine queue capacity.
      values_per_shard: Approximate number of values per shard.
      input_queue_capacity_factor: Minimum number of values to keep in the queue
        in multiples of values_per_shard. See comments above.
      num_reader_threads: Number of reader threads to fill the queue.
      shard_queue_name: Name for the shards filename queue.
      value_queue_name: Name for the values input queue.

    Returns:
      A Queue containing prefetched string values.
    """
    data_files = []
    for pattern in file_pattern.split(","):
        data_files.extend(tf.gfile.Glob(pattern))
    if not data_files:
        tf.logging.fatal("Found no input files matching %s", file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), file_pattern)

    if is_training:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name=shard_queue_name)
        min_queue_examples = values_per_shard * input_queue_capacity_factor
        capacity = min_queue_examples + 100 * batch_size

        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + value_queue_name)
    else:
        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=False, capacity=1, name=shard_queue_name)
        capacity = values_per_shard + 3 * batch_size
        values_queue = tf.FIFOQueue(
            capacity=capacity, dtypes=[tf.string], name="fifo_" + value_queue_name)

    enqueue_ops = []
    for _ in range(num_reader_threads):
        # different thread for enqueue the filename_queue，value is A tuple of Tensors (key, value).
        _, value = reader.read(filename_queue)
        # define a series enqueue ops
        enqueue_ops.append(values_queue.enqueue([value]))
        # give a queueRunner for enqueue the value to values_queue, and add the QueueRunner to GraphKeys.QUEUE_RUNNERS for start easily
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
        values_queue, enqueue_ops))
    tf.summary.scalar(
        "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
        tf.cast(values_queue.size(), tf.float32) * (1. / capacity))
    return values_queue

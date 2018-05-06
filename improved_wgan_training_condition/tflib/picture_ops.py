'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: picture_ops.py
@time: 2018/4/27 上午11:19
@desc: shanghaijiaotong university
'''
from tflib import image_processing
from tflib.inputs import prefetch_input_data, parse_sequence_example
import tensorflow as tf

def process_image(encoded_image, config):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(encoded_image,
                                          height=config.height,
                                          width=config.width,
                                          image_format=config.image_format)

def build_inputs(config, is_training):
    # Prefetch serialized SequenceExample protos.
    reader = tf.TFRecordReader()
    if is_training:
        input_file_pattern = config.input_file_train_pattern
    else:
        input_file_pattern = config.input_file_test_pattern

    input_queue = prefetch_input_data(
        reader,
        input_file_pattern,
        is_training=is_training,
        batch_size=config.batch_size,
        values_per_shard=config.values_per_input_shard,
        # approximate values nums for all shard
        input_queue_capacity_factor=config.input_queue_capacity_factor,
        # queue_capacity_factor for shards
        num_reader_threads=config.num_input_reader_threads)

    # Image processing and random distortion. Split across multiple threads
    # with each thread applying a slightly different color distortions.
    assert config.num_preprocess_threads % 2 == 0
    images_and_label = []
    for thread_id in range(config.num_preprocess_threads):
        # thread
        serialized_sequence_example = input_queue.dequeue()
        encoded_image, image_label, image_name = parse_sequence_example(
            serialized_sequence_example,
            image_feature=config.image_feature_name,
            label_feature=config.label_feature_name,
            filename_feature=config.filename_feature_name)
        # preprocessing, for different thread_id use different distortion function
        image = process_image(encoded_image,config)
        images_and_label.append([image, image_label, image_name])
        # mutil threads preprocessing the image

    queue_capacity = (2 * config.num_preprocess_threads *
                      config.batch_size)
    images, labels, image_names = tf.train.batch_join(
        images_and_label,
        batch_size=config.batch_size,
        capacity=queue_capacity,
        dynamic_pad=True,
        name="batch")
    return images, labels, image_names

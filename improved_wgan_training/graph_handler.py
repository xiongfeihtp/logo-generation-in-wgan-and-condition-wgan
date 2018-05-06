import os
import tensorflow as tf
class GraphHandler(object):
    def __init__(self, config):
        self.config = config
        self.saver = tf.train.Saver(max_to_keep=config.max_to_keep)
        self.writer = tf.summary.FileWriter(config.log_dir)

    def initialize(self, sess):
        if self.config.load_path:
            self._load(sess)

    def save(self, sess, filename, global_step=None):
        saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        saver.save(sess, filename)

    def _load(self, sess):
        config = self.config
        vars_ = {var.name.split(":")[0]: var for var in tf.global_variables()}
        saver = tf.train.Saver(vars_, max_to_keep=config.max_to_keep)
        if config.load_path:
            save_path = config.load_path
        elif config.load_step > 0:
            save_path = os.path.join(
                config.save_dir, "{}_{}.ckpt".format(config.model_name, config.global_step))
        else:
            save_dir = config.save_dir
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "cannot load checkpoint at {}".format(save_dir)
            save_path = checkpoint.model_checkpoint_path
        print("Loading saved model from {}".format(save_path))
        saver.restore(sess, save_path)

    def add_summary(self, summary, global_step):
        self.writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)

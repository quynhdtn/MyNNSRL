from liir.nlp.classifiers.Model import Model
from liir.nlp.nntagger import data_utils
from liir.nlp.nntagger.Seq2SeqModel import Seq2SeqModel




import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf




tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 40000, "French vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

class Seq2SeqTaggerModel(Model):
    def __init__(self):
        Model.__init__(self)


    def makeData(self, X, Y):
        data_set = [[] for _ in _buckets]
        for xsq, ysq in zip(X,Y):
            ysq.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(_buckets):
                  if len(xsq) < source_size and len(ysq) < target_size:
                    data_set[bucket_id].append([xsq, ysq])
                    break
        return data_set




    def create_model(self,session, forward_only):
          """Create translation model and initialize or load parameters in session."""
          model = Seq2SeqModel.Seq2SeqModel(
              FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
              FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
              FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
              forward_only=forward_only)
          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
          if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
          else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
          return model


    def train(self, X, Y):
          """Train a en->fr translation model using WMT data."""
          # Prepare WMT data.
          print("Preparing WMT data in %s" % FLAGS.data_dir)
          en_train, fr_train, en_dev, fr_dev, _, _ = data_utils.prepare_wmt_data(
              FLAGS.data_dir, FLAGS.en_vocab_size, FLAGS.fr_vocab_size)

          with tf.Session() as sess:
            # Create model.
            print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = self.create_model(sess, False)

            # Read data into buckets and compute their sizes.
            print ("Reading development and training data (limit: %d)."
                   % FLAGS.max_train_data_size)
            train_set = self.makeData(X,Y)
            train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
            train_total_size = float(sum(train_bucket_sizes))

            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in range(len(train_bucket_sizes))]

            # This is the training loop.
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            while True:
              # Choose a bucket according to data distribution. We pick a random number
              # in [0, 1] and use the corresponding interval in train_buckets_scale.
              random_number_01 = np.random.random_sample()
              bucket_id = min([i for i in range(len(train_buckets_scale))
                               if train_buckets_scale[i] > random_number_01])

              # Get a batch and make a step.
              start_time = time.time()
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                  train_set, bucket_id)
              _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, False)
              step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
              loss += step_loss / FLAGS.steps_per_checkpoint
              current_step += 1

              # Once in a while, we save checkpoint, print statistics, and run evals.
              if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                  sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0


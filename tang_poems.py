#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import os, sys
import numpy as np
import tensorflow as tf

from model import rnn_model
from utils import process_poems, batch


tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')

tf.app.flags.DEFINE_string('checkpoints_dir', './checkpoints/poems/', 'checkpoints save path')
tf.app.flags.DEFINE_string('train_file', './data/poems.txt', 'train file path')
tf.app.flags.DEFINE_string('model_prefix', 'poems', 'model save prefix')

tf.app.flags.DEFINE_integer('epochs', 50, 'training epochs')

FLAGS = tf.app.flags.FLAGS

start_token = 'G'
end_token = 'E'


def run_training():
    poems_vec, word2id, words = process_poems(FLAGS.train_file)

    x = tf.placeholder(tf.int32, [None, 80])
    y = tf.placeholder(tf.int32, [None, 80])
    l = tf.placeholder(tf.int32, [None])

    end_point = rnn_model(model='lstm', input_data=x, output_data=y, vocab_size=len(words), rnn_size=128,
                          num_layers=2, batch_size=64, lr=FLAGS.lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        start_epoch = 0
        if checkpoint:
            saver.restore(sess, checkpoint)
            print '[INFO] restore from the checkpoint {}'.format(checkpoint)
            start_epoch += int(checkpoint.split('-')[-1])
        print '[INFO] start training...'
        for epoch in range(start_epoch, FLAGS.epochs):
            for x_batch, l_batch, y_batch in batch(FLAGS.batch_size, poems_vec, word2id):
                loss, _, _ = sess.run([end_point['total_loss'], end_point['last_state'], end_point['train_op']],
                                      feed_dict={x: x_batch, y: y_batch})
                print '[INFO] Epoch: {}, training loss: {}'.format(epoch, loss)
            if epoch % 5 == 0:
                saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)


def gen_poem(begin_word):
    batch_size = 1
    print '[INFO] loading corpus from {}'.format(FLAGS.train_file)
    poems_vec, word2id, words = process_poems(FLAGS.train_file)

    x = tf.placeholder(tf.int32, [batch_size, None])

    end_point = rnn_model(model='lstm', input_data=x, output_data=None, vocab_size=len(words), rnn_size=128,
                          num_layers=2, batch_size=64, lr=FLAGS.lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoints_dir)
        saver.restore(sess, checkpoint)

        predict, last_state = sess.run([end_point['prediction'], end_point['last_state']],
                                       feed_dict={x: map(word2id.get, start_token)})

        poem = ''
        word = begin_word
        while word != end_token:
            poem += word
            predict, last_state = sess.run([end_point['prediction'], end_point['last_state']],
                                           feed_dict={x: word2id.get(word, 0),
                                                      end_point['initial_state']: last_state})
            word = to_word(predict, words)

        return poem


def to_word(predict, words):
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(words):
        sample = len(words) - 1
    return words[sample]


def print_poem(poems):
    poems = poems.split(u'。')
    for poem in poems:
        if poem != '' and len(poem) > 10:
            print poem + u'。'


def main(is_train):
    if is_train:
        print '[INFO] train poem...'
        run_training()
    else:
        print '[INFO] write poem...'
        begin_word = raw_input('输入起始字:')
        poems = gen_poem(begin_word)
        print_poem(poems)


if __name__ == '__main__':
    tf.app.run()














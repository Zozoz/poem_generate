#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


import tensorflow as tf


def rnn_model(model, input_data, seq_len, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=64, lr=0.01):
    end_point = {}
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    elif model == 'lstm':
        cell_fun = tf.contrib.rnn.BasicLSTMCell

    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    with tf.device('/gpu:0'):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform([vocab_size + 1, rnn_size], -1.0, 1.0))
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, seq_len, initial_state=initial_state)
    outputs = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.get_variable('weight', initializer=tf.truncated_normal([rnn_size, vocab_size + 1]))
    biases = tf.get_variable('bias', initializer=tf.zeros([vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(outputs, weights), bias=biases)

    if output_data is not None:
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(lr).minimize(total_loss)

        end_point['initial_state'] = initial_state
        end_point['outputs'] = outputs
        end_point['train_op'] = train_op
        end_point['total_loss'] = total_loss
        end_point['loss'] = loss
        end_point['last_state'] = last_state
    else:
        prediction = tf.nn.softmax(logits)
        end_point['initial_state'] = initial_state
        end_point['last_state'] = last_state
        end_point['prediction'] = prediction

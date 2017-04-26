#!/usr/bin/env python
# encoding: utf-8
# @author: newbie
# email: zhengshiliang0@gmail.com


from collections import Counter
import os
import sys

import numpy as np


start_token = 'G'
end_token = 'E'


def process_poems(file_name):
    poems = []
    for line in open(file_name):
        title, content = line.decode('utf8').strip().split(':')
        content = content.replace(' ', '')
        if len(content) < 5 or len(content) > 79:
            continue
        content = start_token + content + end_token
        poems.append(content)
    # 统计词频
    poems = sorted(poems, key=lambda d: len(d))
    word_count = Counter()
    for poem in poems:
        for word in poem:
            word_count[word] += 1
    count_pairs = sorted(word_count.items(), key=lambda d: -d[1])
    words, _ = zip(*count_pairs)
    # 取常用词 & 转化古诗序列为向量
    words = (' ',) + words[:len(words)]
    word2id = dict(zip(words, range(len(words))))
    poems_vec = [map(lambda w: word2id.get(w, 0), poem) for poem in poems]

    return poems_vec, word2id, words


def batch(batch_size, poems_vec, word2id):
    np.random.shuffle(poems_vec)
    for i in range(len(poems_vec) / batch_size):
        s = i * batch_size
        e = s + batch_size
        batches = poems_vec[s:e]
        x_batch = np.zeros((batch_size, 80))
        len_batch = []
        for j in range(len(poems_vec)):
            x_batch[i, :len(batches[j])] = batches[j]
            len_batch.append(len(batches[j]))
        y_batch = np.copy(x_batch)
        y_batch[:, :-1] = x_batch[:, 1:]
        yield x_batch, len_batch, y_batch







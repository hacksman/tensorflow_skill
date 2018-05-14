# -*- coding: utf-8 -*-
# @Time    : 5/14/18 8:31 AM

import tensorflow as tf
import numpy as np


def emb():
    """
     embedding_lookup
    """
    # embeding_demo = np.random.random_integers(1, 3, [1, 10])
    embeding_demo = np.random.random_integers(1, 3, [10, 3])
    embeding_look = tf.nn.embedding_lookup(embeding_demo, [2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(embeding_look))
        print(embeding_demo)


if __name__ == '__main__':
    emb()

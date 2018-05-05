# -*- coding: utf-8 -*-
# @Time    : 5/5/18 7:40 AM


import tensorflow as tf


# dropout
def dropout_demo():
    '''
    :param: x, keep_prob, noise_shape=None, seed=None, name=None
    关键参数：keep_prob为保持一定比率的值，另外一个作用，是将数据中不丢弃的数，变为原来的1/keep_prob倍
    
    实验结果：keep_prob虽然表示保持一定比率的值，但不代表绝对的。貌似是均值概率是这么多也就是说，如下16个元素，kee_prob为0.6，
            理论上应该保持16*0.6=9.6，也就是9个或10个，但实验的时候发现，元素可能保持7个或13个等更多情况
    
    参考资料：https://www.jianshu.com/p/c9f66bc8f96c 
    '''

    dropout = tf.placeholder(tf.float32)

    x = tf.Variable(tf.ones([4, 4]))
    y = tf.nn.dropout(x, dropout)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(y, feed_dict={dropout: 0.6}))

if __name__ == '__main__':
    dropout_demo()




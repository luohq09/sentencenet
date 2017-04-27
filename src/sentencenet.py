import tensorflow as tf
import numpy as np


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor sentences.
      positive: the embeddings for the positive sentences.
      negative: the embeddings for the negative sentences.
      alpha: dist margin

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


def get_label_count(labels):
    d = {}
    for i in xrange(labels.shape[0]):
        if labels[i] in d:
            d[labels[i]] = d[labels[i]] + 1
        else:
            d[labels[i]] = 1

    ret = np.zeros(shape=labels.shape[0], dtype=np.float32)
    for i in xrange(labels.shape[0]):
        ret[i] = d[labels[i]]
    return ret


def center_loss(embeddings, labels, label_counts, alfa, num_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    embedding_size = embeddings.get_shape()[1]
    centers = tf.get_variable(
        'centers',
        [num_classes, embedding_size],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0),
        trainable=False)
    labels = tf.reshape(labels, [-1])
    # label_count = get_label_count(labels)
    centers_batch = tf.gather(centers, labels)
    diff = alfa * (centers_batch - embeddings)
    centers = tf.scatter_sub(centers, labels, diff / (label_counts + 1.))
    loss = tf.nn.l2_loss(embeddings - centers_batch) / embeddings.shape[0]
    return loss, centers


def get_optimizer(name, learning_rate):
    if name == 'ADAGRAD':
        opt = tf.train.AdagradOptimizer(learning_rate)
    elif name == 'ADADELTA':
        opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
    elif name == 'ADAM':
        opt = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999)
    elif name == 'RMSPROP':
        opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
    elif name == 'MOM':
        opt = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    else:
        raise ValueError('Invalid optimization algorithm')

    return opt


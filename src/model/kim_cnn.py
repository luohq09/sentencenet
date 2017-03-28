import tensorflow as tf
#import tensorflow.contrib as tfcontrib


class KimCNN(object):
    def __init__(self, sequence_length, filter_sizes, num_filters,
                 pretrained_word_embeddings, sentence_embedding_size, l2_reg_lambda=0.0):
        # Placeholders for input, dropout and pretrained_embedding
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.l2_loss = tf.constant(0.0)

        word_embedding_size = len(pretrained_word_embeddings[0])
        # Embedding layer
        word_embedding_w = tf.Variable(
            initial_value=tf.constant(0.0, shape=[len(pretrained_word_embeddings), word_embedding_size]),
            trainable=False, name="word_embedding_w")
        word_embedding_init = word_embedding_w.assign(pretrained_word_embeddings)

        with tf.control_dependencies([word_embedding_init]):
            embedded_words = tf.nn.embedding_lookup(word_embedding_w, self.input_x)
            embedded_words_expanded = tf.expand_dims(embedded_words, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, word_embedding_size, 1, num_filters]
                    weight = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="weight")
                    bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bias")
                    conv = tf.nn.conv2d(input=embedded_words_expanded,
                                        filter=weight,
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="conv")
                    # Apply nonlinearity
                    activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        activation,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

                    if l2_reg_lambda > 0.0:
                        self.l2_loss += tf.nn.l2_loss(weight)
                        self.l2_loss += tf.nn.l2_loss(bias)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

            # Final normalized sentence embedding
            with tf.name_scope("output"):
                weight = tf.Variable(
                    initial_value=tf.truncated_normal([num_filters_total, sentence_embedding_size], stddev=0.1),
                    name="weight")
                bias = tf.Variable(tf.constant(0.1, shape=[sentence_embedding_size]), name="bias")
                sentence_embeddings = tf.nn.xw_plus_b(h_drop, weight, bias, name="sentence_embeddings")
                self.normalized_sentence_embeddings = tf.nn.l2_normalize(
                    sentence_embeddings, 1, 1e-10, name="normalized_sentence_embeddings")

                if l2_reg_lambda > 0.0:
                    self.l2_loss += tf.nn.l2_loss(weight)
                    self.l2_loss += tf.nn.l2_loss(bias)
                    self.l2_loss *= l2_reg_lambda

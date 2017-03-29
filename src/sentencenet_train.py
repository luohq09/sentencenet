import tensorflow as tf
import numpy as np
import itertools
import os
import time
import datetime

from model.kim_cnn import KimCNN
import pretrained_word_embedding
import data_helper
import sentencenet
import sentencenet_evaluate

# Data loading params
tf.app.flags.DEFINE_string("pretrained_word_embedding_file", "", "Data source for the pretrained word embeddings")
tf.app.flags.DEFINE_string("train_data_file", "", "Data source for the train data")
tf.app.flags.DEFINE_string("dev_data_file", "", "Data source for the dev data")

tf.app.flags.DEFINE_string("model_restore_dir", None, "Directory containing checkpoints used to restore"
                                                      " the model (default None)")

# Model Hyperparameters
tf.app.flags.DEFINE_float("alpha", 0.2, "Positive to negative triplet distance margin (default: 0.2)")
tf.app.flags.DEFINE_integer("sequence_length", 20, "Fixed length of sentence (default: 20)")
tf.app.flags.DEFINE_integer("sentence_embedding_size", 128, "Dimensionality of the sentence embedding (default: 128)")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Initial learning rate (default: 0.1)")
tf.app.flags.DEFINE_integer("learning_rate_decay_epochs", 50,
                            "Number of epochs between learning rate decay (default: 50)")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 1.0, "Learning rate decay factor (default: 1.0)")
tf.app.flags.DEFINE_integer("word_embedding_size", 50, "Dimensionality of the word embedding (default: 50)")

tf.app.flags.DEFINE_float("select_alpha", 0.2, "distance margin used to select the negative for triplet (default: 0.2)")

# Training parameters
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of classes per batch (default: 64)")
tf.app.flags.DEFINE_integer("max_sentences_per_class", 10, "Maximum number of sentences per class (default: 10)")
tf.app.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.app.flags.DEFINE_string("optimizer", "ADAGRAD", "The optimization algorithm to use: ['ADAGRAD', 'ADADELTA',"
                                                   " 'ADAM', 'RMSPROP', 'MOM'] (default: 'ADAGRAD')")
tf.app.flags.DEFINE_boolean("word_embedding_static", True,
                            "Whether to keep word embeddings static during training (default: True)")

tf.app.flags.DEFINE_string("out_dir", os.path.curdir,
                           "Output directory for model and summaries (default: 'current dir')")

FLAGS = tf.app.flags.FLAGS

FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def main(argv=None):
    word_number_dict, word_embeddings = pretrained_word_embedding.load_word_embedding(
        FLAGS.pretrained_word_embedding_file, np.zeros(FLAGS.word_embedding_size))
    train_sentence_classes = data_helper.load_sentences(FLAGS.train_data_file, word_number_dict, FLAGS.sequence_length)
    dev_sentence_classes = data_helper.load_sentences(FLAGS.dev_data_file, word_number_dict, FLAGS.sequence_length)

    num_batches_per_epoch = int((len(train_sentence_classes)-1)/FLAGS.batch_size) + 1

    global_step = tf.Variable(0, name="global_step", trainable=False)

    sess = tf.Session()

    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
    learning_rate = tf.train.exponential_decay(learning_rate=learning_rate_placeholder,
                                               global_step=global_step,
                                               decay_steps=FLAGS.learning_rate_decay_epochs*num_batches_per_epoch,
                                               decay_rate=FLAGS.learning_rate_decay_factor,
                                               staircase=True)
    net = KimCNN(
        sequence_length=FLAGS.sequence_length,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        pretrained_word_embeddings=word_embeddings,
        sentence_embedding_size=FLAGS.sentence_embedding_size,
        word_embedding_static=FLAGS.word_embedding_static,
        l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Define Training procedure
    anchor, positive, negative = tf.unstack(
        tf.reshape(net.normalized_sentence_embeddings, [-1, 3, FLAGS.sentence_embedding_size]), 3, 1)
    triplet_loss = sentencenet.triplet_loss(anchor, positive, negative, FLAGS.alpha)
    total_loss = triplet_loss + net.l2_loss
    optimizer = sentencenet.get_optimizer(FLAGS.optimizer, learning_rate)
    grads_and_vars = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(FLAGS.out_dir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Summaries for losses
    triplet_loss_summary = tf.summary.scalar("triplet_loss", triplet_loss)
    total_loss_summary = tf.summary.scalar("total_loss", total_loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([triplet_loss_summary, total_loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    if FLAGS.model_restore_dir is None:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(net.word_embedding_init)
    else:
        checkpoint_file_path = tf.train.latest_checkpoint(FLAGS.model_restore_dir)
        saver.restore(sess, checkpoint_file_path)

    with sess.as_default():
        batches = data_helper.batch_iter(train_sentence_classes, FLAGS.batch_size, FLAGS.num_epochs, True)
        for batch in batches:
            # Select triplets based on the embeddings
            print('Selecting suitable triplets for training')
            triplet_sentences, num_anchor_pos = select_triplets(sess=sess,
                                                                net=net,
                                                                sentence_classes=batch,
                                                                max_sentences_per_class=FLAGS.max_sentences_per_class,
                                                                alpha=FLAGS.select_alpha)
            print ("num_anchor_pos, num_triplets = (%d, %d)" % (num_anchor_pos, len(triplet_sentences)))

            if len(triplet_sentences) > 0:
                # flatten triplet_sentences
                train_sentences = list(itertools.chain(*triplet_sentences))
                train_sentences = np.asarray(train_sentences)

                # train step
                feed_dict = {
                    net.input_x: train_sentences,
                    net.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    learning_rate_placeholder: FLAGS.learning_rate
                }
                _, step, summaries, current_triplet_loss, current_total_loss = sess.run(
                    [train_op, global_step, train_summary_op, triplet_loss, total_loss],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, triplet_loss {:g}, total_loss {:g}".format(
                    time_str, step, current_triplet_loss, current_total_loss))
                train_summary_writer.add_summary(summaries, step)

                # save checkpoint
                if int(step) % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

                # evaluate
                if int(step) % FLAGS.evaluate_every == 0:
                    min_accuracy, avg_accuracy = sentencenet_evaluate.evaluate(sess, net,
                                                                               dev_sentence_classes,
                                                                               train_sentence_classes)
                    time_str = datetime.datetime.now().isoformat()
                    print ("\nEvaluation-{}: step {}, min_accuracy {:g}, avg_accuracy {:g}".
                           format(time_str, step, min_accuracy, avg_accuracy))
                    print("")

        # final save and evaluate
        step = tf.train.global_step(sess, global_step)
        path = saver.save(sess, checkpoint_prefix, global_step=step)
        print("Saved model checkpoint to {}\n".format(path))

        min_accuracy, avg_accuracy = sentencenet_evaluate.evaluate(sess, net,
                                                                   dev_sentence_classes,
                                                                   train_sentence_classes)
        time_str = datetime.datetime.now().isoformat()
        print ("\nEvaluation-{}: step {}, min_accuracy {:g}, avg_accuracy {:g}".
               format(time_str, step, min_accuracy, avg_accuracy))
        print("")


def select_triplets(sess, net, sentence_classes, max_sentences_per_class, alpha):
    flatten_sentences = []
    lengths = []
    for sentence_class in sentence_classes:
        num_sentence = len(sentence_class.sentences)
        if num_sentence > 1:
            shuffle_indices = np.random.permutation(np.arange(num_sentence))
            sentences = sentence_class.sentences[shuffle_indices[0:min(num_sentence, max_sentences_per_class)]]
            flatten_sentences.extend(sentences)
            lengths.append(len(sentences))

    feed_dict = {
        net.input_x: np.asarray(flatten_sentences),
        net.dropout_keep_prob: 1.
    }
    sentence_embeddings = sess.run(net.normalized_sentence_embeddings, feed_dict)

    triplet_sentences = []
    start_index = 0
    num_anchor_pos = 0
    for length in lengths:
        for anchor_index in xrange(start_index, start_index + length - 1):
            neg_dists_sqr = np.sum(np.square(sentence_embeddings[anchor_index] - sentence_embeddings), 1)
            for pos_index in xrange(anchor_index + 1, start_index + length):
                num_anchor_pos += 1

                pos_dist_sqr = np.sum(np.square(sentence_embeddings[anchor_index] - sentence_embeddings[pos_index]))
                neg_dists_sqr[start_index:start_index + length] = 1000.
                all_negs = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]

                num_all_negs = all_negs.shape[0]
                if num_all_negs > 0:
                    random_index = np.random.randint(num_all_negs)
                    neg_index = all_negs[random_index]
                    triplet_sentences.append((flatten_sentences[anchor_index],
                                              flatten_sentences[pos_index],
                                              flatten_sentences[neg_index]))

        start_index += length

    np.random.shuffle(triplet_sentences)
    return triplet_sentences, num_anchor_pos


if __name__ == '__main__':
    tf.app.run()


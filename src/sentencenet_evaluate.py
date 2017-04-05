import numpy as np
import tensorflow as tf
import datetime

from model.kim_cnn import KimCNN
import pretrained_word_embedding
import data_helper

if __name__ == '__main__':
    tf.app.flags.DEFINE_integer("top_k", 1, "Select top k candidates (default: 1)")

    # Data loading params
    tf.app.flags.DEFINE_string("pretrained_word_embedding_file", "", "Data source for the pretrained word embeddings")
    tf.app.flags.DEFINE_string("train_data_file", "", "Data source for the train data")
    tf.app.flags.DEFINE_string("dev_data_file", "", "Data source for the dev data")
    tf.app.flags.DEFINE_string("model_restore_dir", None,
                               "Directory containing checkpoints used to restore the model (default None)")

    tf.app.flags.DEFINE_integer("sequence_length", 20, "Fixed length of sentence (default: 20)")
    tf.app.flags.DEFINE_integer("sentence_embedding_size", 128,
                                "Dimensionality of the sentence embedding (default: 128)")
    tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.app.flags.DEFINE_integer("word_embedding_size", 50, "Dimensionality of the word embedding (default: 50)")

    FLAGS = tf.app.flags.FLAGS


def flatten_sentences(sentence_classes):
    sentences = []
    for sentence_class in sentence_classes:
        sentences.extend(sentence_class.sentences)
    return np.asarray(sentences)


def generate_embeddings(sess, net, sentences):
    feed_dict = {
        net.input_x: sentences,
        net.dropout_keep_prob: 1.
    }
    return sess.run(net.normalized_sentence_embeddings, feed_dict)


def classify(dev_embedding, train_embeddings, train_sentence_classes, top_k=1):
    dists_sqr = np.sum(np.square(dev_embedding - train_embeddings), 1)
    index = 0
    top_k = max(min(top_k, len(train_sentence_classes)), 1)

    min_min_dists = np.full(top_k, 10000.)
    min_avg_dists = np.full(top_k, 10000.)
    min_dist_class_names = np.empty(top_k, 'a128')
    avg_dist_class_names = np.empty(top_k, 'a128')
    for train_sentence_class in train_sentence_classes:
        min_dist = 1000.
        avg_dist = 0.
        for i in xrange(len(train_sentence_class.sentences)):
            dist_sqr = dists_sqr[index]
            if dist_sqr < min_dist:
                min_dist = dist_sqr
            avg_dist += dist_sqr
            index += 1
        avg_dist /= len(train_sentence_class.sentences)

        arg_max_index = np.argmax(min_min_dists)
        if min_dist < min_min_dists[arg_max_index]:
            min_min_dists[arg_max_index] = min_dist
            min_dist_class_names[arg_max_index] = train_sentence_class.name

        arg_max_index = np.argmax(min_avg_dists)
        if avg_dist < min_avg_dists[arg_max_index]:
            min_avg_dists[arg_max_index] = avg_dist
            avg_dist_class_names[arg_max_index] = train_sentence_class.name
    return min_dist_class_names, min_min_dists, avg_dist_class_names, min_avg_dists


def evaluate(sess, net, dev_sentence_classes, train_sentence_classes, top_k=1):
    # flatten sentences
    dev_flatten_sentences = flatten_sentences(dev_sentence_classes)
    train_flatten_sentences = flatten_sentences(train_sentence_classes)

    # generate embeddings
    dev_embeddings = generate_embeddings(sess, net, dev_flatten_sentences)
    train_embeddings = generate_embeddings(sess, net, train_flatten_sentences)

    avg_pos_num = 0
    min_pos_num = 0
    index = 0
    for dev_sentence_class in dev_sentence_classes:
        for i in xrange(len(dev_sentence_class.sentences)):
            min_dist_class_names, _, avg_dist_class_names, _ = classify(
                dev_embeddings[index], train_embeddings, train_sentence_classes, top_k)
            index += 1
            if dev_sentence_class.name in min_dist_class_names:
                min_pos_num += 1
            if dev_sentence_class.name in avg_dist_class_names:
                avg_pos_num += 1
    return float(min_pos_num) / len(dev_flatten_sentences), float(avg_pos_num) / len(dev_flatten_sentences)


def main(argv=None):
    word_number_dict, word_embeddings = pretrained_word_embedding.load_word_embedding(
        FLAGS.pretrained_word_embedding_file, np.zeros(FLAGS.word_embedding_size))
    train_sentence_classes = data_helper.load_sentences(FLAGS.train_data_file, word_number_dict, FLAGS.sequence_length)
    dev_sentence_classes = data_helper.load_sentences(FLAGS.dev_data_file, word_number_dict, FLAGS.sequence_length)

    net = KimCNN(
        sequence_length=FLAGS.sequence_length,
        filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
        num_filters=FLAGS.num_filters,
        pretrained_word_embeddings=word_embeddings,
        sentence_embedding_size=FLAGS.sentence_embedding_size,
        word_embedding_static=True)

    sess = tf.Session()
    saver = tf.train.Saver()
    if FLAGS.model_restore_dir is None:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(net.word_embedding_init)
    else:
        checkpoint_file_path = tf.train.latest_checkpoint(FLAGS.model_restore_dir)
        saver.restore(sess, checkpoint_file_path)

    min_accuracy, avg_accuracy = evaluate(sess, net, dev_sentence_classes, train_sentence_classes, FLAGS.top_k)
    time_str = datetime.datetime.now().isoformat()
    print ("\nEvaluation-{}: min_accuracy {:g}, avg_accuracy {:g}".
           format(time_str, min_accuracy, avg_accuracy))
    print("")

if __name__ == '__main__':
    tf.app.run()

import numpy as np
import tensorflow as tf

from model.kim_cnn import KimCNN
import pretrained_word_embedding
import data_helper
import sentencenet_evaluate as evaluate


if __name__ == '__main__':
    tf.app.flags.DEFINE_integer("top_k", 1, "Select top k candidates (default: 1)")

    # Data loading params
    tf.app.flags.DEFINE_string("pretrained_word_embedding_file", "", "Data source for the pretrained word embeddings")
    tf.app.flags.DEFINE_string("train_data_file", "", "Data source for the train data")
    tf.app.flags.DEFINE_string("dev_data_file", "", "Data source for the dev data")
    tf.app.flags.DEFINE_string("model_restore_dir", None,
                               "Directory containing checkpoints used to restore the model (default None)")

    tf.app.flags.DEFINE_string("output_file", "./out.txt", "Output file path (default: ./out.txt)")

    tf.app.flags.DEFINE_integer("sequence_length", 20, "Fixed length of sentence (default: 20)")
    tf.app.flags.DEFINE_integer("sentence_embedding_size", 128,
                                "Dimensionality of the sentence embedding (default: 128)")
    tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
    tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
    tf.app.flags.DEFINE_integer("word_embedding_size", 50, "Dimensionality of the word embedding (default: 50)")

    FLAGS = tf.app.flags.FLAGS


def load_dev_sentences(sentence_path, word_number_dict, fixed_sentence_length):
    sentences = []

    # statistics info
    num_trimmed = 0

    with open(sentence_path) as fn:
        for line in fn:
            splits = line.split("=====>")
            words = splits[1].strip().split(" ")

            sentence = np.zeros(fixed_sentence_length)
            idx = 0
            for word in words:
                if idx >= fixed_sentence_length:
                    num_trimmed += 1
                    break
                if word in word_number_dict:
                    sentence[idx] = word_number_dict[word]
                    idx += 1

            sentences.append((splits[0].strip(), sentence))

    print("%s:\nnum_sentences: %d, trimmed: %d" % (sentence_path, len(sentences), num_trimmed))
    return sentences


def main(argv=None):
    word_number_dict, word_embeddings = pretrained_word_embedding.load_word_embedding(
        FLAGS.pretrained_word_embedding_file, np.zeros(FLAGS.word_embedding_size))
    train_sentence_classes = data_helper.load_sentences(FLAGS.train_data_file, word_number_dict, FLAGS.sequence_length)
    dev_sentences = load_dev_sentences(FLAGS.dev_data_file, word_number_dict, FLAGS.sequence_length)

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

    train_flatten_sentences = evaluate.flatten_sentences(train_sentence_classes)
    dev_flatten_sentences = [pair[1] for pair in dev_sentences]
    dev_flatten_sentences = np.asarray(dev_flatten_sentences)

    # generate embeddings
    dev_embeddings = evaluate.generate_embeddings(sess, net, dev_flatten_sentences)
    train_embeddings = evaluate.generate_embeddings(sess, net, train_flatten_sentences)

    with open(FLAGS.output_file, "w") as out_fn:
        for i in xrange(len(dev_sentences)):
            min_dist_class_names, min_min_dists, _, _ = evaluate.classify(
                dev_embeddings[i], train_embeddings, train_sentence_classes, FLAGS.top_k)
            class_dists = [(min_dist_class_names[index], min_min_dists[index])
                           for index in xrange(len(min_dist_class_names))]
            class_dists = sorted(class_dists, cmp=lambda x, y: cmp(x[1], y[1]))

            # write to the output file
            out_fn.write("<question=\"{}\";id=null;entity=>\nquestion:\n{}semantics:\n"
                         .format(dev_sentences[i][0], dev_sentences[i][0]))
            for class_dist in class_dists:
                out_fn.write("null\t###\t{}\t###\t{:g}\n".format(class_dist[0], class_dist[1]))
            out_fn.write("\n")

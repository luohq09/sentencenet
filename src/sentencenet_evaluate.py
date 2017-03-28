import numpy as np


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


def classify(dev_embedding, train_embeddings, train_sentence_classes):
    dists_sqr = np.sum(np.square(dev_embedding - train_embeddings), 1)
    index = 0
    min_min_dist = 10000.
    min_avg_dist = 10000.
    min_dist_class_name = ""
    avg_dist_class_name = ""
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
        if min_dist < min_min_dist:
            min_min_dist = min_dist
            min_dist_class_name = train_sentence_class.name
        if avg_dist < min_avg_dist:
            min_avg_dist = avg_dist
            avg_dist_class_name = train_sentence_class.name
    return min_dist_class_name, avg_dist_class_name


def evaluate(sess, net, dev_sentence_classes, train_sentence_classes):
    # flatten sentences
    dev_flatten_sentences = flatten_sentences(dev_sentence_classes)
    train_flatten_sentences = flatten_sentences(train_sentence_classes)

    # generate embeddings
    dev_embeddings = generate_embeddings(sess, net, dev_flatten_sentences)
    train_embeddings = generate_embeddings(sess, net, train_sentence_classes)

    avg_pos_num = 0
    min_pos_num = 0
    index = 0
    for dev_sentence_class in dev_sentence_classes:
        for i in xrange(len(dev_sentence_class.sentences)):
            min_dist_class_name, avg_dist_class_name = classify(dev_embeddings[index],
                                                                train_embeddings,
                                                                train_sentence_classes)
            if min_dist_class_name == dev_sentence_class.name:
                min_pos_num += 1
            if avg_dist_class_name == dev_sentence_class.name:
                avg_pos_num += 1
    return float(min_pos_num) / len(dev_flatten_sentences), float(avg_pos_num) / len(dev_flatten_sentences)

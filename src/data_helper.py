import numpy as np


class SentenceClass:
    def __init__(self, name, sentences):
        self.name = name
        self.sentences = sentences


def load_sentences(sentence_path,
                   word_number_dict,
                   fixed_sentence_length):
    sentence_classes = []
    sentence_class = None
    sentences = []
    trimmed = 0

    # -1: initial; 0: new class; 1: loading sentences
    state = -1
    with open(sentence_path) as fn:
        for line in fn:
            if line.startswith("<semantic"):
                # new class
                if not (sentence_class is None) and len(sentences) > 0:
                    sentence_class.sentences = np.asarray(sentences)
                    sentence_classes.append(sentence_class)

                class_name = line.split(";")[1].split("=")[1].strip()
                sentences = []
                sentence_class = SentenceClass(class_name, None)
                state = 0
            elif state == 1:
                # parse new sentence
                words = line.strip().split(" ")
                sentence = np.zeros(fixed_sentence_length)
                idx = 0
                for word in words:
                    if idx >= fixed_sentence_length:
                        trimmed += 1
                        break
                    if word in word_number_dict:
                        sentence[idx] = word_number_dict[word]
                        idx += 1
                sentences.append(sentence)
            elif state == 0 and line.startswith("questions:"):
                state = 1

        if not (sentence_class is None) and len(sentences) > 0:
            sentence_class.sentences = np.asarray(sentences)
            sentence_classes.append(sentence_class)

        print("trimmed: %d" % trimmed)
        return sentence_classes


def batch_iter(sentence_classes, batch_size, num_epochs, shuffle=True):
    sentence_classes = np.array(sentence_classes)
    sentence_classes_size = len(sentence_classes)
    num_batches_per_epoch = int((sentence_classes_size-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(sentence_classes_size))
            shuffled_sentence_classes = sentence_classes[shuffle_indices]
        else:
            shuffled_sentence_classes = sentence_classes
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, sentence_classes_size)
            yield shuffled_sentence_classes[start_index:end_index]


# import time
# import pretrained_embedding
# start_time = time.time()
# word_dict, embeddings = pretrained_embedding.load_embedding(
#     "/Users/luohuaqing/sentence_data/data/xiaoshuo_cbow8.txt", np.zeros(50))
# s_classes = load_sentences("/Users/luohuaqing/sentence_data/data/paraphrase/lkl-2016-11-29-train-seg.txt", word_dict, 25)
# print("s_classes len: %d" % len(s_classes))
# for s_class in s_classes:
#     print("class name: %s. len: %d" % (s_class.name, len(s_class.sentences)))
# print("--- %s seconds ---" % (time.time() - start_time))

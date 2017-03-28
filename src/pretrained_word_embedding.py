import numpy as np


def load_word_embedding(embedding_path, padding_embedding):
    """
    embedding_path: file path to load pretrained embeddings
    padding_embedding: embedding used for padding word, placed at embeddings[0]
    """
    word_number_dict = {}
    embeddings = [padding_embedding]
    embedding_size = len(padding_embedding)
    with open(embedding_path) as fn:
        idx = 1
        for line in fn:
            word, embedding_str = line.split("\t")

            word_number_dict[word] = idx
            idx += 1

            embedding = np.array(embedding_str.split(" ", embedding_size), np.float32)
            embeddings.append(embedding)

    embeddings = np.asarray(embeddings)
    return word_number_dict, embeddings

# import time
# start_time = time.time()
# word_number_dict, embeddings = load_embedding("/Users/luohuaqing/sentence_data/data/xiaoshuo_cbow8.txt", np.zeros(50))
# print("word_number_dict len: %d" % len(word_number_dict))
# print("embeddings len: %d" % len(embeddings))
# print("--- %s seconds ---" % (time.time() - start_time))

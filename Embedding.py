import parameters
import numpy as np
import logging
# generate pre-trained embedding


class Embedding:

    class __Embedding:
        def __init__(self, path):
            self.path = path
            self.word_to_id_map, self.embedding_numpy = self.load_embedding()
            self.pad_index = self.embedding_numpy.shape[0] - 1

        def get_embedding_numpy(self):
            return self.embedding_numpy

        def get_word_embedding(self, word, unknown_word="UNK"):
            return self.embedding_numpy[self.get_word_index(word, unknown_word)]

        def get_word_index(self, word, unknown_word="UNK"):
            if word in self.word_to_id_map:
                return self.word_to_id_map[word]
            else:
                return self.word_to_id_map[unknown_word]

        def get_pad_index(self):
            return self.pad_index

        def load_embedding(self):
            embedding_file = open(self.path)
            next(embedding_file)
            word_to_id_map = {}
            embedding_list = []
            for (M, line) in enumerate(embedding_file):
                values = line.split()
                word = values[0]
                coef = np.asarray(values[1:], dtype='float32')
                word_to_id_map[word] = M
                embedding_list.append(coef)
            embedding_file.close()
            # add one additional row at the end of embedding numpy array to be used for padding
            embedding_list.append(np.zeros_like(coef))
            return word_to_id_map, np.stack(embedding_list)

    instance = None

    def __init__(self, path=parameters.embedding_file):
        if not Embedding.instance:
            Embedding.instance = Embedding.__Embedding(path)

    def get_word_embedding(self, word, unknown_word="UNK"):
        return self.instance.get_word_embedding(word, unknown_word)

    def get_word_index(self, word, unknown_word="UNK"):
        return self.instance.get_word_index(word, unknown_word)

    def get_embedding_numpy(self):
        return self.instance.get_embedding_numpy()

    def get_pad_index(self):
        return self.instance.get_pad_index()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    embedding_1 = Embedding()
    embedding_2 = Embedding()
    logging.info(Embedding().get_word_embedding("haioncviae"))

if __name__ == "__main__":
    main()

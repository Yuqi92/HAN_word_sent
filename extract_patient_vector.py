# extract patient vector of obesity from two-level HAN
import G3_parameters
import tensorflow as tf
import logging
from Embedding import Embedding
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from utility import HAN_model

logging.basicConfig(filename=G3_parameters.log_file_name, level=logging.INFO)

train_csv = pd.read_csv(G3_parameters.obesity_train_label)

train_patient_name = list(train_csv['patient_id'])
n_train = len(train_patient_name)

def generate_token_embedding(subject_id):
    x_doc = np.ones([G3_parameters.max_document_length,
                      G3_parameters.max_sentence_length], dtype=np.int32) * Embedding().get_pad_index()
    patient_file = G3_parameters.obesity_train_file_folder + str(subject_id) + '.txt'

    word_length_per_doc = []
    text = open(patient_file)
    num_sent_per_doc = 0
    for current_sent_index, line in enumerate(text):
        num_sent_per_doc += 1
        if current_sent_index < G3_parameters.max_document_length:
            strip_line = line.strip()
            word_list = strip_line.split()
            num_word_per_sent = len(word_list)
            if num_word_per_sent <= G3_parameters.max_sentence_length:
                word_length_per_doc.append(num_word_per_sent)
            else:
                word_length_per_doc.append(G3_parameters.max_sentence_length)
            for current_word_index, word in enumerate(word_list):
                    if current_word_index < G3_parameters.max_sentence_length:
                        x_doc[current_sent_index][current_word_index] = Embedding().get_word_index(word=word.lower(),
                                                                                                   unknown_word="UNK")
        else:
            num_sent_per_doc = G3_parameters.max_document_length
            continue
    num_word_per_doc = word_length_per_doc + [0] * (G3_parameters.max_document_length - num_sent_per_doc)
    text.close()
    return x_doc, num_sent_per_doc, num_word_per_doc

def load_x_data_for_HAN(patient_name, keep_prob, input_x, sentence_lengths, word_lengths, dropout_keep_prob,
                        is_training_ph, is_training_value):

    generate_token_embedding_results = []
    for i in patient_name:
        generate_token_embedding_results.append(generate_token_embedding(i))

    tmp_x = np.zeros([len(generate_token_embedding_results),
                            G3_parameters.max_document_length,
                            G3_parameters.max_sentence_length], dtype=np.float32)
    sent_l = []
    word_l = []

    for (M, r) in enumerate(generate_token_embedding_results):

        tmp_x[M] = r[0]
        sent_l.append(r[1])
        word_l.append(r[2])

    sent_l = np.asarray(sent_l)
    word_l = np.stack(word_l)
    feed_dict = {input_x: tmp_x,
                 sentence_lengths: sent_l,
                 word_lengths: word_l,
                 dropout_keep_prob: keep_prob,
                 is_training_ph: is_training_value}

    return feed_dict

#define placeholder

input_ys = []
for i in range(G3_parameters.multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None,G3_parameters.num_classes], name="input_y"+str(i)))

input_x = tf.placeholder(tf.int32,
                         [None, G3_parameters.max_document_length, G3_parameters.max_sentence_length],name="input_x")
is_training = tf.placeholder(tf.bool, [], name="is_training")
word_lengths = tf.placeholder(shape=(None, G3_parameters.max_document_length), dtype=tf.int32, name='word_lengths')
sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')
dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")

_, _, patient_vector = HAN_model(input_x,input_ys,word_lengths,sentence_lengths, 
	is_training, dropout_keep_prob,
	Embedding().get_embedding_numpy())


saver = tf.train.Saver()
num_batch = int(math.ceil(n_train / G3_parameters.n_batch))

with tf.Session() as sess:
    saver.restore(sess, G3_parameters.deep_learning_model)

    for i in tqdm(range(num_batch)):
        tmp_train_patient_name = train_patient_name[i*G3_parameters.n_batch:min((i+1)*G3_parameters.n_batch, n_train)]

        feed_dict = load_x_data_for_HAN(tmp_train_patient_name, G3_parameters.drop_out_train, input_x,
                    sentence_lengths, word_lengths, dropout_keep_prob,is_training, True)

        tmp_patient_vector = sess.run(patient_vector, feed_dict=feed_dict)

        for j in range(len(tmp_train_patient_name)):
            np.save(G3_parameters.obesity_ptvector_directory + tmp_train_patient_name[j] + ".npy", tmp_patient_vector[j])


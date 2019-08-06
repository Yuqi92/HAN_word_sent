import logging
import G3_parameters
import numpy as np
import pandas as pd
import math
from random import shuffle
from bn_lstm import BNLSTMCell
from model_components import task_specific_attention, bidirectional_rnn
from tqdm import tqdm
import glob
from sklearn.metrics import roc_auc_score
from Embedding import Embedding
import tensorflow as tf
from G3_utility import split_train_dev, assign_patient_subject_id, generate_task_label, \
    HAN_model, load_x_data_for_HAN, test_dev_auc

logging.basicConfig(filename=G3_parameters.log_file_name, level=logging.INFO, format='%(asctime)s %(message)s')

result_csv = pd.read_csv(G3_parameters.label_csv)

train_index, dev_index = split_train_dev(len(result_csv))
label = 'ICD9_CODE'

train_label = result_csv[label].iloc[train_index]
dev_label = result_csv[label].iloc[dev_index]


train_patient_name = assign_patient_subject_id(result_csv,train_index) # subject_id
dev_patient_name = assign_patient_subject_id(result_csv,dev_index)

y_train_task = generate_task_label(train_label)
y_dev_task = generate_task_label(dev_label)

n_train = len(train_patient_name)
n_dev = len(dev_patient_name)

logging.info("number of train " + str(n_train))

num_train_batch = int(math.ceil(n_train / G3_parameters.n_batch))
num_dev_batch = int(math.ceil(n_dev / G3_parameters.n_batch))

input_ys = []
for i in range(G3_parameters.multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None,G3_parameters.num_classes], name="input_y"+str(i)))

input_x = tf.placeholder(tf.int32,
                         [None, G3_parameters.max_document_length, G3_parameters.max_sentence_length],
                         name="input_x")
is_training = tf.placeholder(tf.bool, [], name="is_training")
word_lengths = tf.placeholder(shape=(None, G3_parameters.max_document_length), dtype=tf.int32, name='word_lengths')
sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')
dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")

optimize, scores_soft_max_list, _ = HAN_model(
    input_x,input_ys,word_lengths,sentence_lengths, is_training, dropout_keep_prob,
    Embedding().get_embedding_numpy()
)


saver = tf.train.Saver()

with tf.Session() as sess:
    if G3_parameters.restore:
        saver.restore(sess, G3_parameters.deep_learning_model)
    else:
        sess.run(tf.global_variables_initializer())

    shuf_train_ind = np.arange(n_train)
    max_auc = 0
    current_early_stop_times = 0

    while True:
        shuffle(shuf_train_ind)
        train_patient_name = train_patient_name[shuf_train_ind]
        for i in range(len(y_train_task)):
            y_train_task[i] = y_train_task[i][shuf_train_ind]


        for i in tqdm(range(num_train_batch)):
            tmp_train_patient_name = train_patient_name[i*G3_parameters.n_batch:min((i+1)*G3_parameters.n_batch, n_train)]
            tmp_y_train = []
            for t in y_train_task:
                tmp_y_train.append(t[i*G3_parameters.n_batch:min((i+1)*G3_parameters.n_batch, n_train)])

            feed_dict = load_x_data_for_HAN(tmp_train_patient_name, G3_parameters.drop_out_train, input_x,
                    sentence_lengths, word_lengths, dropout_keep_prob,is_training, True)

            for (M, input_y) in enumerate(input_ys):
                feed_dict[input_y] = tmp_y_train[M]
            
            # logging.info("start to train")
            sess.run([optimize], feed_dict=feed_dict)

        dev_auc,_ = test_dev_auc(num_dev_batch, y_dev_task, dev_patient_name, n_dev, sess,
                                   input_x, sentence_lengths, word_lengths,
                                 dropout_keep_prob, scores_soft_max_list, test_output_flag=False,
                                 is_training_ph=is_training, is_training_value=False)
        logging.info("Dev AUC: {}".format(dev_auc))

        if dev_auc > max_auc:
            save_path = saver.save(sess, G3_parameters.deep_learning_model)
            logging.info("- new best score!")
            max_auc = dev_auc
            current_early_stop_times = 0
        else:
            current_early_stop_times += 1
        if current_early_stop_times >= G3_parameters.early_stop_times:
            logging.info("- early stopping {} epochs without improvement".format(current_early_stop_times))
            break



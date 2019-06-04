import numpy as np
import pandas as pd
import parameters
import tensorflow as tf
import tensorflow.contrib.layers as layers
import logging

from Embedding import Embedding
from sklearn.metrics import roc_auc_score, roc_curve
from bn_lstm import BNLSTMCell
from model_components import task_specific_attention, bidirectional_rnn


def split_train_test_dev(n_patient):
    if not parameters.load_index:

        index_list = np.arange(n_patient)
        np.random.shuffle(index_list)
        n_dev = len(index_list) // 10
        dev_index = index_list[:n_dev]
        test_index = index_list[n_dev:(2*n_dev)]
        train_index = index_list[(2*n_dev):]

        np.save(parameters.index_dev_path, dev_index)
        np.save(parameters.index_test_path, test_index)
        np.save(parameters.index_train_path, train_index)

    else:
        logging.info("Load index")
        dev_index = np.load(parameters.index_dev_path)
        train_index = np.load(parameters.index_train_path)
        test_index = np.load(parameters.index_test_path)
    return train_index, test_index, dev_index

def transfer_string_list_icd9(cell):
    new_cell_list = []
    cell_nob = cell[1:-1]
    cell_list = cell_nob.split(',')
    for c in cell_list:
        c = c.strip()
        c = c[1:-1]
        new_cell_list.append(c)
    return new_cell_list

def generate_task_label(y_series):
    labels = []
    for task in parameters.multi_task_list:
        label = []
        for index,y in y_series.iteritems():
            y = transfer_string_list_icd9(y)
            if task in y:
                label.append([0,1])
            else:
                label.append([1,0])
        label = np.asarray(label)
        labels.append(label)
    return labels



def assign_patient_subject_id(result_csv,index):
    patient_name = np.asarray(result_csv["SUBJECT_ID"].iloc[index])
    return patient_name

pid_subjectid = pd.read_csv('./subject.csv')
def _find_pid(sid):
    pid = pid_subjectid.loc[pid_subjectid['subject_id'] == sid, 'pid'].iloc[0]
    return pid

#TODO: word_l: 1d [n_sent]
def generate_token_embedding(subject_id):
    pid = _find_pid(subject_id)
    x_doc = np.ones([parameters.max_document_length,
                      parameters.max_sentence_length], dtype=np.int32) * Embedding().get_pad_index()
    current_sentence_ind = 0
    word_length_per_sent = []
    if pid is not None:
        f = open(parameters.note_directory + 'patient'+ str(pid) + '.txt')
        categories_id_per_file = []
        waiting_for_new_sentence_flag = True
        for line in f:
            strip_line = line.strip()
            if len(strip_line) == 0:
                waiting_for_new_sentence_flag = True
                word_length_per_sent.append(current_word_ind)
                if current_word_ind > 0:
                    current_sentence_ind += 1
                    if current_sentence_ind >= parameters.max_document_length:
                        break
                else:
                    logging.warning("Continues blank line in file: " + pid)
                # add something to x_token
                continue
            if waiting_for_new_sentence_flag:  # is new category line
                categories_id_per_file.append(int(strip_line))
                waiting_for_new_sentence_flag = False
                # x_sentence = np.zeros([HP.n_max_word_num,
                #                       HP.embedding_size], dtype=np.float32)
                current_word_ind = 0
            else:  # is new word line
                if current_word_ind < parameters.max_sentence_length: # Do not load real embedding, load index 
                    x_doc[current_sentence_ind][current_word_ind] = Embedding().get_word_index(
                        word=strip_line, unknown_word="UNK"
                    )
                    current_word_ind += 1
        if not waiting_for_new_sentence_flag:
            logging.warning("Do not find new line at the bottom of the file: " + str(pid) + ". Which will cause one ignored sent")
        f.close()
        number_of_sentences = len(categories_id_per_file)
        # categories_id_per_file = categories_id_per_file + [0]*(parameters.max_document_length-number_of_sentences)
        number_of_words = word_length_per_sent + [0]*(parameters.max_document_length-number_of_sentences)
        return x_doc, number_of_sentences, number_of_words
    else:
        logging.info("no existing note for this patient " + str(subject_id))

def load_x_data_for_simple(patient_name, input_x):
    p_vector_list = []
    for p in patient_name:
        p_np = np.load(parameters.patient_vector_directory + str(p) + ".npy")
        p_vector_list.append(p_np)
    tmp_x = np.stack(p_vector_list)
    feed_dict = {input_x: tmp_x}
    return feed_dict



#TODO: word_l: 2d the same with category_id_per_file
def load_x_data_for_HAN(patient_name, keep_prob, input_x, sentence_lengths, word_lengths, dropout_keep_prob,
                        is_training_ph, is_training_value):

    generate_token_embedding_results = []
    for i in patient_name:
        generate_token_embedding_results.append(generate_token_embedding(i))

    tmp_x = np.zeros([len(generate_token_embedding_results),
                            parameters.max_document_length,
                            parameters.max_sentence_length], dtype=np.float32)
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

# Perceptron for target task
def simple_model(input_x, input_ys):
    # input_x : n_batch * document_filter_size
    total_loss = 0
    scores_soft_max_list = []
    for (M,input_y) in enumerate(input_ys):
        with tf.name_scope("task"+str(M)):

            W_fully = tf.Variable(tf.truncated_normal([parameters.document_num_filters, parameters.document_num_filters], stddev=0.1), name="W_fully")
            b_fully = tf.Variable(tf.constant(0.1, shape=[parameters.document_num_filters]), name="b_fully")
            scores_2 = tf.nn.xw_plus_b(input_x, W_fully, b_fully) # n_batch * document_num_filters

            with tf.name_scope("dropout_second"):
                scores_drop = tf.nn.dropout(scores_2, 0.8)

            W = tf.Variable(tf.truncated_normal([parameters.document_num_filters, parameters.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[parameters.num_classes]), name="b")

            scores = tf.nn.xw_plus_b(scores_drop, W, b)
            # scores has shape: [n_batch, num_classes]
            scores_soft_max = tf.nn.softmax(scores)
            scores_soft_max_list.append(scores_soft_max)   # scores_soft_max_list shape:[multi_size, n_batch, num_classes]
            # caculate loss
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            loss_avg = tf.reduce_mean(losses)
            total_loss += loss_avg

    optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
    optimize = optimizer.minimize(total_loss)
    scores_soft_max_list = tf.stack(scores_soft_max_list, axis=0)

    return optimize, scores_soft_max_list



# word_lengths: placeholder of number of words in per sentence; the shape is [n-doc, n_sent]
# sentence_lengths: placeholder of number of sentences in per doc: the shape is [n_doc]
# more understandings might be in HAN class 
def HAN_model(input_x,input_ys, word_lengths, sentence_lengths, is_training, dropout_keep_prob, embedding_numpy):

    word_embeddings = tf.get_variable(
            name="word_embedding",
            dtype=tf.float32,
            shape=embedding_numpy.shape,
            initializer=tf.constant_initializer(embedding_numpy),
            trainable=False)
    input_x = tf.nn.embedding_lookup(word_embeddings, input_x)
    # shape: [n_batch,n_sent,n_word,embed_size]

    # ============================================= word_level AN ===============================================#
    word_level_inputs = tf.reshape(input_x, [-1, parameters.max_sentence_length, parameters.embedding_size]) #reshape to 3D

    #shape of word_lengths: 2D [n_batch, n_sent]
    word_level_lengths = tf.reshape(word_lengths, [-1])  # reshape to 1D

    with tf.variable_scope("word") as scope:
        word_fw_cell = BNLSTMCell(100, is_training)
        word_bw_cell = BNLSTMCell(100, is_training)
        word_encoder_output, _ = bidirectional_rnn(
            word_fw_cell, word_bw_cell,
            word_level_inputs, 
            word_level_lengths,
            scope=scope)

        with tf.variable_scope('attention') as scope:
              word_level_output = task_specific_attention(
                word_encoder_output,
                parameters.word_output_size,
                scope=scope)

        with tf.name_scope("dropout"):
            word_level_output = tf.nn.dropout(word_level_output, dropout_keep_prob)

      # shape of word_level_output: 2D [n_batch*n_sent, word_output_size]
# ============================================= sent_level HAN ===============================================#
    sentence_level_inputs = tf.reshape(word_level_output,
        [-1, parameters.max_document_length, parameters.word_output_size]) # reshape to 3D
    # sentence_lengths:n_batch
    with tf.variable_scope('sentence') as scope:
        sentence_fw_cell = BNLSTMCell(100, is_training)
        sentence_bw_cell = BNLSTMCell(100, is_training)
        sentence_encoder_output, _ = bidirectional_rnn(
          sentence_fw_cell, sentence_bw_cell,
          sentence_level_inputs, 
          sentence_lengths, 
          scope=scope) 

        with tf.variable_scope('attention') as scope:
          sentence_level_output = task_specific_attention(
            sentence_encoder_output, 
            parameters.sentence_output_size, 
            scope=scope) 

        patient_vector = sentence_level_output
        with tf.name_scope("dropout"):
            sentence_level_output = tf.nn.dropout(sentence_level_output, dropout_keep_prob)   # shape: n_batch * sentence_output_size


    total_loss = 0
    scores_soft_max_list = []
    for (M,input_y) in enumerate(input_ys):
        with tf.name_scope("task"+str(M)):

            W = tf.Variable(tf.truncated_normal([parameters.sentence_output_size, parameters.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[parameters.num_classes]), name="b")

            scores = tf.nn.xw_plus_b(sentence_level_output, W, b)
            # scores has shape: [n_batch, num_classes]
            scores_soft_max = tf.nn.softmax(scores)
            scores_soft_max_list.append(scores_soft_max)  # scores_soft_max_list shape:[multi_size, n_batch, num_classes]
            # predictions = tf.argmax(scores, axis=1, name="predictions")
            # predictions has shape: [None, ]. A shape of [x, ] means a vector of size x
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            # losses has shape: [None, ]
            # include target replication
            # total_loss += losses
            loss_avg = tf.reduce_mean(losses)
            total_loss += loss_avg
    # avg_loss = tf.reduce_mean(total_loss)
    # optimize function
    optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
    optimize = optimizer.minimize(total_loss)
    scores_soft_max_list = tf.stack(scores_soft_max_list, axis=0)
    # correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    # accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="accuracy")

    return optimize, scores_soft_max_list, patient_vector





def test_dev_auc(num_batch, y_task, patient_name, n, sess,
                 input_x, sent_lengths, word_lengths,
                 dropout_keep_prob, scores_soft_max_list, test_output_flag,
                 is_training_ph=None, is_training_value=None):

    y_total_task_label = []
    predictions = []

    seperate_pre = {}
    y_seperate_task_label = {}
    auc_per_task = {}



    for m in range(parameters.multi_size):
        seperate_pre[m] = []
        y_seperate_task_label[m] = []
    for i in range(num_batch):
        tmp_patient_name = patient_name[i*parameters.n_batch:min((i+1)*parameters.n_batch, n)]
        for (y_i, y) in enumerate(y_task):
            tmp_y_task = y[i*parameters.n_batch:min((i+1)*parameters.n_batch, n)]
            y_total_task_label.extend(np.argmax(tmp_y_task, axis=1).tolist())
            y_seperate_task_label[y_i].extend(np.argmax(tmp_y_task, axis=1).tolist())

        if parameters.model_type == "HAN":
            feed_dict = load_x_data_for_HAN(tmp_patient_name, 1.0, input_x, sent_lengths, word_lengths,
                                            dropout_keep_prob, is_training_ph, is_training_value)
        elif parameters.model_type == "Perceptron":
            feed_dict = load_x_data_for_simple(tmp_patient_name, input_x)
        else:
            logging.error("not support model type")
            feed_dict = None

        pre = sess.run(scores_soft_max_list, feed_dict=feed_dict)
        for m in range(parameters.multi_size):
            pre_slice = pre[m, :]
            pre_pos = pre_slice[:, 1]
            seperate_pre[m].extend(pre_pos.tolist())
        pre = pre.reshape(-1, parameters.num_classes)  # [3*n_batch,2]  in one batch: task1+task2+task3
        pre = pre[:, 1]  # get probability of positive class
        predictions.extend(pre.tolist())

        auc = roc_auc_score(np.asarray(y_total_task_label), np.asarray(predictions))

    if test_output_flag:
        for m in range(parameters.multi_size):
            auc_per_task[m] = roc_auc_score(np.asarray(y_seperate_task_label[m]), np.asarray(seperate_pre[m]))

    else:
        logging.info("Dev finished")

    return auc, auc_per_task


import logging
import parameters
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from Embedding import Embedding
import tensorflow as tf
from utility import split_train_test_dev, assign_patient_subject_id,\
generate_task_label, test_dev_auc, load_x_data_for_simple, \
load_x_data_for_HAN, HAN_model


logging.basicConfig(filename=parameters.log_file_name, level=logging.INFO, format='%(asctime)s %(message)s')

result_csv = pd.read_csv(parameters.label_csv)

train_index, test_index, dev_index = split_train_test_dev(len(result_csv))

label = 'dead_after_disch_date'
train_label = result_csv[label].iloc[train_index]
test_label = result_csv[label].iloc[test_index]
dev_label = result_csv[label].iloc[dev_index]

dev_patient_name = assign_patient_subject_id(result_csv,dev_index)
test_patient_name = assign_patient_subject_id(result_csv,test_index)  # patient_250, patient251, ...
train_patient_name = assign_patient_subject_id(result_csv,train_index)  # subject_id

y_train_task = generate_task_label(train_label)
y_dev_task = generate_task_label(dev_label)
y_test_task = generate_task_label(test_label)

n_train = len(train_patient_name)
n_dev = len(dev_patient_name)
n_test = len(test_patient_name)

# train CNN model
num_train_batch = int(math.ceil(n_train / parameters.n_batch))
num_dev_batch = int(math.ceil(n_dev / parameters.n_batch))
num_test_batch = int(math.ceil(n_test / parameters.n_batch))
is_training = None

# define placeholder:
input_ys = []
for i in range(parameters.multi_size):
    input_ys.append(tf.placeholder(tf.int32, [None,parameters.num_classes], name="input_y"+str(i)))

if parameters.model_type == "HAN":
    input_x = tf.placeholder(tf.int32,
                         [None, parameters.max_document_length, parameters.max_sentence_length],
                         name="input_x")
    is_training = tf.placeholder(tf.bool, [], name="is_training")
    word_lengths = tf.placeholder(shape=(None, parameters.max_document_length), dtype=tf.int32, name='word_lengths')
    sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')
    dropout_keep_prob = tf.placeholder(tf.float32, [], name="dropout_keep_prob")

    optimize, scores_soft_max_list, _ = HAN_model(
        input_x,input_ys,word_lengths,sentence_lengths, is_training, dropout_keep_prob,
        Embedding().get_embedding_numpy()
    )

#
# elif parameters.model_type == "Perceptron":
#     input_x = tf.placeholder(tf.float32,
#                          [None, parameters.sentence_output_size],
#                          name="input_x")
#     dropout_keep_prob = None
#     word_lengths = None
#     sentence_lengths = None
#     optimize, scores_soft_max_list = simple_model(input_x, input_ys)

else:
    logging.error("unsupport model type")
    optimize = None
    scores_soft_max_list = None


np.random.seed(20)
saver = tf.train.Saver()

with tf.Session() as sess:
    if parameters.restore:
        saver.restore(sess, parameters.deep_learning_model)
    else:
        sess.run(tf.global_variables_initializer())

    shuf_train_ind = np.arange(n_train)
    max_auc = 0
    current_early_stop_times = 0

    while True:
        np.random.shuffle(shuf_train_ind)
        train_patient_name = train_patient_name[shuf_train_ind]
        for i in range(len(y_train_task)):
            y_train_task[i] = y_train_task[i][shuf_train_ind]


        for i in tqdm(range(num_train_batch)):
            tmp_train_patient_name = train_patient_name[i*parameters.n_batch:min((i+1)*parameters.n_batch, n_train)]
            tmp_y_train = []
            for t in y_train_task:
                tmp_y_train.append(t[i*parameters.n_batch:min((i+1)*parameters.n_batch, n_train)])

            if parameters.model_type == "HAN":
                feed_dict = load_x_data_for_HAN(tmp_train_patient_name, parameters.drop_out_train, input_x, 
                    sentence_lengths, word_lengths, dropout_keep_prob,
                                                is_training, True)
            # elif parameters.model_type == "Perceptron":
            #     feed_dict = load_x_data_for_simple(tmp_train_patient_name, input_x)
            else:
                logging.error("unsupported model type")
                feed_dict = None

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
            save_path = saver.save(sess, parameters.deep_learning_model)
            logging.info("- new best score!")
            max_auc = dev_auc
            current_early_stop_times = 0
        else:
            current_early_stop_times += 1
        if current_early_stop_times >= parameters.early_stop_times:
            logging.info("- early stopping {} epochs without improvement".format(current_early_stop_times))
            break

    test_auc,test_auc_per_task = test_dev_auc(num_test_batch, y_test_task, test_patient_name, n_test, sess,
                            input_x, sentence_lengths, word_lengths, dropout_keep_prob, scores_soft_max_list,
                                              test_output_flag=True, is_training_ph=is_training, is_training_value=False)
    logging.info("Test total AUC: {}".format(test_auc))
    logging.info("Multi-task list: " + str(parameters.tasks_dead_date))
    logging.info("Test total AUC: {}".format(test_auc_per_task))

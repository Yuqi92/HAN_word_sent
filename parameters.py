
load_index = True
index_path = './index/'
index_dev_path = index_path + 'dev.index.npy'
index_test_path = index_path + 'test.index.npy'
index_train_path = index_path + 'train.index.npy'

restore = False
model_path = './results/3_task_dead/'
deep_learning_model = model_path + "model.weights/model.ckpt"

max_document_length = 1000 # how many sentences in document(max): n_sent
max_sentence_length = 25  # how many words in the sentence(max): n_word
embedding_size = 100

task = 'dead' # option:  dead; LOS; ICD9
tasks_icd_9 = []
tasks_dead_date = [0, 31, 366]
tasks_los_date = []

multi_size = len(tasks_dead_date)

embedding_file = './embeddings/mimic.k100.w2v'
label_csv = './final_label_mortality_LOS_icd.csv'
note_directory = './patient_files/'

log_file_name = model_path + 'log.log'




n_batch = 32
early_stop_times = 5
num_classes = 2
model_type = "HAN"
drop_out_train = 0.8
learning_rate = 0.001


word_output_size = 100
sentence_output_size = 100


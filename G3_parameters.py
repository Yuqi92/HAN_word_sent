load_index = True
restore = False


max_document_length = 640 # how many sentences in document(max): n_sent
max_sentence_length = 25  # how many words in the sentence(max): n_word
embedding_size = 100

task = 'icd' # option:  dead; LOS; ICD9
tasks_icd_9 = ["G3"]
tasks_dead_date = []
tasks_los_date = []

multi_size = len(tasks_icd_9)

embedding_file = './embeddings/mimic.k100.w2v'
label_csv = './final_label_mortality_LOS_icd.csv'
pretrain_note_dir = './patient_files/'

index_path = './index_obesity/'
index_dev_path = index_path + 'dev.index.npy'
index_train_path = index_path + 'train.index.npy'

model_path = './mimic_G3_pretrained/'
deep_learning_model = model_path + "model.weights/model.ckpt"


word_output_size = 100
sentence_output_size = 100


drop_out_train = 0.8
num_classes = 2
learning_rate = 0.001
early_stop_times = 5
n_batch = 50

log_file_name = model_path + 'pretrain_G3.log'

obesity_train_label = '/obesity_2008/train_dev_label_nofill.csv'
obesity_train_file_folder = '/obesity_2008/train_dev/'

obesity_ptvector_directory = '/patient_vector/'

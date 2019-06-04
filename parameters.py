
embedding_file = './embeddings/mimic.k100.w2v'
model_path = './results/uni_task/G17_HAN/'
deep_learning_model = model_path + "model.weights/model.ckpt"

log_file_name = model_path + 'log'
label_csv = './label_2.csv'
load_index = True
index_path = './index'

note_directory = './entire_file/'

category = ['pad', 'Respiratory', 'ECG','Radiology','Nursing/other','Rehab Services','Nutrition','Pharmacy','Social Work',
            'Case Management','Physician','General','Nursing','Echo','Consult']

#multi_task_list = ["G7","G3","G8","G10","G9","G17"]
multi_task_list = ["G17"]

category_id = {cate: idx for idx, cate in enumerate(category)}
n_category = len(category)

index_train_path = index_path + '/train.npy'
index_dev_path = index_path + '/dev.npy'
index_test_path = index_path + '/test.npy'

n_batch = 32
early_stop_times = 5
multi_size = len(multi_task_list)
num_classes = 2

model_type = "HAN"

max_document_length = 1000
max_sentence_length = 25
embedding_size = 100

drop_out_train = 0.8
document_filter_size = 3


learning_rate = 0.001

restore = False

#patient_vector_directory = './patient_vector/G_7_3_8_10_9_2/'


# HAN:

word_output_size = 100
sentence_output_size = 50
plot = False

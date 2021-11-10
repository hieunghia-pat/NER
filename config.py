## model configuration
embedding_dim = 300
rnn_size = 512
smoothing = 0.2
num_head = 8
d_model = 512
dff = 2048
num_layers = 8

## dataset configuration
json_file = "ner_dataset.json"
json_file_train_vi = "phonert_train.json"
json_file_val_vi = "phonert_val.json"
json_file_test_vi = "phonert_test.json"
vectors = ["phow2v.word.300d"]

## training configuration
batch_size = 64
num_workers = 0
initial_lr = 3e-3
lr_halflife = 50000
epochs = 10
model_checkpoint = "saved_models"
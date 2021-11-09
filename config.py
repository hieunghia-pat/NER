## model configuration
embedding_dim = 512
rnn_size = 512
smoothing = 0.2

## dataset configuration
json_file = "ner_dataset.json"

## training configuration
batch_size = 64
num_workers = 0
initial_lr = 3e-3
lr_halflife = 50000
epochs = 10
model_checkpoint = "saved_models"
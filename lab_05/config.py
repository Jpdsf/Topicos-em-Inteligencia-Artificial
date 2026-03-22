# Hiperparâmetros centralizados para o Lab 05
DATA_NUM_SAMPLES = 1000
DATA_SRC_LANG = "en"
DATA_TGT_LANG = "de"
DATA_DATASET_NAME = "bentrevett/multi30k"
DATA_SPLIT = "train"

# Tokenizer
TOKENIZER_MODEL = "bert-base-multilingual-cased"
TOKENIZER_MAX_LENGTH = 64
TOKENIZER_PAD_ID = 0

# Model
MODEL_VOCAB_SIZE = 119547  # bert-base-multilingual-cased
MODEL_D_MODEL = 128
MODEL_NUM_HEADS = 4
MODEL_NUM_LAYERS = 2
MODEL_D_FF = 256
MODEL_DROPOUT = 0.1

# Training
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-3
TRAIN_BATCH_SIZE = 32

# Overfitting Test
OVERFIT_EPOCHS = 300
OVERFIT_LR = 1e-3
OVERFIT_MAX_NEW_TOKENS = 50

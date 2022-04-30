import os


DATA_DIR = os.environ.get('DATA_DIR')
MODEL_DIR = os.environ.get('MODEL_DIR')
RESUME_FROM = os.environ.get('RESUME_FROM')
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', 'bentenmann')
WANDB_PROJECT_NAME = os.environ.get('WANDB_PROJECT_NAME', 'protein-classification-development')

DATA_FILES = ['source.npy', 'target.npy']
DATA_SPLITS = ['train', 'dev', 'test']
SOURCE_COLUMN = 'sequence'
TARGET_COLUMN = 'family_accession'
SEQUENCE_LENGTH = 256

NUM_TOKENS = 25
NUM_CLASSES = 17_929
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-5
ALPHA = 1.0
GAMMA = 1.5
BATCH_SIZE = 64
NUM_EPOCHS = 80
MODEL_CONF = dict(
    num_embeddings=NUM_TOKENS,
    embedding_dim=64,
    residual_block_def={
        'input_features': 64,
        'block_features': 128,
        'kernel_size': (9,),
        'dilation': 3,
        'padding': 'same'
    },
    n_residual_blocks=4,
    num_labels=NUM_CLASSES
)
MODEL_EVAL_CONF = MODEL_CONF.copy()
MODEL_EVAL_CONF['residual_block_def']['use_running_avg'] = True

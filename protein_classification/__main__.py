from os import environ
from pathlib import Path

import torch
import srsly
import wandb
from torch import nn

from . import model as mdl
from .constants import DEVICE
from .execution import (
    OptimizerStep,
    training,
    testing,
    validation
)
from .utils import (
    load_dataloader,
    load_tokenizer,
    now
)


def main():
    """
    Entrypoint for model training and benchmarking script, accessed via the command line. Refer to `__init__.py` for
    variable definitions.
    """
    config_path = Path(environ.get('CONFIG_MAP'))
    config = srsly.read_yaml(config_path)

    model_conf = config.get('model', {})
    model_param = model_conf.get('param', {})
    # gets the model object we want; gives us more flexibility training different models
    model = getattr(mdl, model_conf.get('name'))(**model_param)
    model.to(DEVICE)

    label_map = srsly.read_json(environ.get('LABEL_MAP'))
    tokenizer = load_tokenizer(environ.get('TOKEN_MAP'))

    env = config.get('env', {})
    loader_map = {'tokenizer': tokenizer,
                  'label_map': label_map,
                  'batch_size': env.get('batch_size'),
                  'max_length': model_param.get('seq_len')}

    train_loader = load_dataloader(environ.get('TRAIN_DATA'), **loader_map)
    dev_loader = load_dataloader(environ.get('DEV_DATA'), **loader_map)
    test_loader = load_dataloader(environ.get('TEST_DATA'), **loader_map)

    if 'LABEL_WEIGHTS' in environ:
        lw = srsly.read_json(environ.get('LABEL_WEIGHTS'))
        weights = torch.tensor(lw.get('weights'), dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    optim_param = config.get('optim', {'lr': 1e-3})
    optim = torch.optim.Adam(model.parameters(), **optim_param)
    optimizer = OptimizerStep(optim)

    wandb.login(key=environ.get('WANDB_API_KEY'))
    wandb.init(entity=environ.get('WANDB_ENTITY'), project=environ.get('WANDB_PROJECT'))

    epochs = env.get('epochs')
    for _ in range(epochs):
        model = training(model, train_loader, loss_fn, optimizer)
        model = validation(model, dev_loader, loss_fn)

    model = testing(model, test_loader)
    output_path = Path(environ.get('SAVE_PATH'))

    torch.save(model.state_dict(), output_path / f'{now()}.bin')


if __name__ == '__main__':
    main()

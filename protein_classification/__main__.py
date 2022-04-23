import re
from os import environ
from pathlib import Path

import torch
import srsly
import wandb
from torch import nn
from transformers import AutoTokenizer

from . import model as mdl
from .constants import DEVICE
from .execution import (
    OptimizerStep,
    LRSchedulerStep,
    training,
    testing,
    validation
)
from .utils import (
    load_dataloader,
    load_tokenizer,
    now
)


def main(config: dict = None) -> Path:
    """
    Entrypoint for model training and benchmarking script, accessed via the command line. Refer to `__init__.py` for
    variable definitions.
    """
    if config is None:
        path = environ['CONFIG_MAP']
        config_path = Path(path)
        config = srsly.read_yaml(config_path)

    model_conf = config.get('model', {})
    model_param = model_conf.get('param', {})
    # gets the model object we want; gives us more flexibility training different models
    model = getattr(mdl, model_conf.get('name'))(**model_param)
    model.to(DEVICE)

    label_map_path = environ['LABEL_MAP']
    label_map = srsly.read_json(label_map_path)

    token_map_path = environ.get('TOKEN_MAP')
    if token_map_path:
        tokenizer = load_tokenizer(token_map_path)
    else:
        if 'tokenizer' not in config:
            raise ValueError('tokenizer left undefined')
        identifier = config['tokenizer']['identifier']
        tokenizer = AutoTokenizer.from_pretrained(identifier)

    env = config.get('env', {})
    loader_map = {'tokenizer': tokenizer,
                  'label_map': label_map,
                  'batch_size': env.get('batch_size'),
                  'max_length': model_param.get('seq_len', 256)}

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

    lr_scheduler_param = config.get('lr_scheduler')
    if lr_scheduler_param is None:
        scheduler = type('StaticLR', (object,), {'step': lambda *args, **kwargs: None})
    else:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, **lr_scheduler_param)
    lr_scheduler_step = LRSchedulerStep(scheduler)
    step_per_batch = config.get('step_per_batch', True)

    wandb.login(key=environ.get('WANDB_API_KEY'))
    wandb.init(
        entity=environ.get('WANDB_ENTITY'),
        project=environ.get('WANDB_PROJECT'),
        config={
            'model_config': config,
            'data_config': {'train': environ['TRAIN_DATA'],
                            'dev': environ['DEV_DATA'],
                            'test': environ['TEST_DATA']}
        },
    )

    model_dir = Path(environ.get('SAVE_PATH')) / f'{model_conf.get("name")}-{now()}'
    model_dir.mkdir(parents=True, exist_ok=True)

    resume_from = environ.get('RESUME_FROM')
    if resume_from:
        prev_epochs = re.findall(r"epoch-([0-9]+)\.bin", resume_from)[0]
        prev_epochs = int(prev_epochs)
        state_dict = torch.load(resume_from, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        prev_epochs = 0

    epochs = env.get('epochs')
    for eps in range(1 + prev_epochs, epochs + 1):
        print(f'epoch: {eps}/{epochs}')
        model.train()
        model = training(model, train_loader, loss_fn, optimizer, lr_scheduler_step, step_per_batch)
        model.eval()
        model = validation(model, dev_loader, loss_fn)

        output_path = model_dir / f'epoch-{eps}.bin'
        torch.save(model.state_dict(), output_path)

    testing(model, test_loader)
    return model_dir


if __name__ == '__main__':
    main()

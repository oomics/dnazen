import logging

from torch.utils.data import DataLoader
from torch import nn

import deepspeed
from deepspeed.accelerator import get_accelerator

logger = logging.getLogger(__name__)


def train_epoch(
    model_engine: deepspeed.DeepSpeedEngine,
    optimizer,
    data_loader: DataLoader,
    checkpoint_dir: str,
    save_interval: int = 100,
):
    model_engine.train()
    if model_engine.device == get_accelerator().device(0):

        def _log_info(*args, **kwargs):
            logger.info(*args, **kwargs)
    else:

        def _log_info(*args, **kwargs):
            logger.info(*args, **kwargs)

    for i, batch in enumerate(data_loader):
        for k, v in batch.items():
            batch[k] = v.to(model_engine.device)

        loss = model_engine(**batch)
        model_engine.backward(loss)
        model_engine.step()

        if (i + 1) % save_interval == 0:
            model_engine.save_checkpoint(checkpoint_dir)
            _log_info(f"Model saved at step {i + 1}.")

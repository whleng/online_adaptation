import abc
import os
import pathlib
from typing import Dict, List, Literal, Protocol, Sequence, Union, cast

import lightning.pytorch as pl
import numpy as np
import torch
import torch.utils._pytree as pytree
import torch_geometric.data as tgd
from lightning.pytorch import Callback
from lightning.pytorch.loggers import WandbLogger

PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())


# This matching function
def match_fn(dirs: Sequence[str], extensions: Sequence[str], root: str = PROJECT_ROOT):
    def _match_fn(path: pathlib.Path):
        in_dir = any([str(path).startswith(os.path.join(root, d)) for d in dirs])

        if not in_dir:
            return False

        if not any([str(path).endswith(e) for e in extensions]):
            return False

        return True

    return _match_fn


TorchTree = Dict[str, Union[torch.Tensor, "TorchTree"]]


def flatten_outputs(outputs: List[TorchTree]) -> TorchTree:
    """Flatten a list of dictionaries into a single dictionary."""

    # Concatenate all leaf nodes in the trees.
    flattened_outputs = [pytree.tree_flatten(output) for output in outputs]
    flattened_list = [o[0] for o in flattened_outputs]
    flattened_spec = flattened_outputs[0][1]  # Spec definitely should be the same...
    cat_flat = [torch.cat(x) for x in list(zip(*flattened_list))]
    output_dict = pytree.tree_unflatten(cat_flat, flattened_spec)
    return cast(TorchTree, output_dict)


class CanMakePlots(Protocol):
    @staticmethod
    @abc.abstractmethod
    def make_plots(preds, batch: tgd.Batch):
        pass


class LightningModuleWithPlots(pl.LightningModule, CanMakePlots):
    pass


# TODO: Change
class LogPredictionSamplesCallback(Callback):
    def __init__(self, logger: WandbLogger, eval_per_n_epoch):
        self.logger = logger
        self.eval_per_n_epoch = eval_per_n_epoch

    @staticmethod
    def eval_log_random_sample(
        trainer: pl.Trainer,
        pl_module: LightningModuleWithPlots,
        outputs,
        batch,
        prefix: Literal["train", "val", "unseen"],
    ):
        preds = outputs["preds"]
        random_id = np.random.randint(0, len(batch))
        preds = preds.reshape(batch.num_graphs, -1, preds.shape[-2], preds.shape[-1])[
            random_id
        ]
        data = batch.get_example(random_id)
        plots = pl_module.make_plots(preds.cpu(), data.cpu())

        assert trainer.logger is not None and isinstance(trainer.logger, WandbLogger)
        trainer.logger.experiment.log(
            {
                **{f"{prefix}/{plot_name}": plot for plot_name, plot in plots.items()},
                "global_step": trainer.global_step,
            },
            step=trainer.global_step,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        if batch_idx != 0:
            return
        dataloader_names = ["train", "val", "unseen"]
        name = dataloader_names[dataloader_idx]
        if pl_module.current_epoch % self.eval_per_n_epoch == 0:
            self.eval_log_random_sample(trainer, pl_module, outputs, batch, name)

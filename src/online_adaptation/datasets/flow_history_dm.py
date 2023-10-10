import os

import lightning as L
import rpad.partnet_mobility_utils.dataset as rpd
import torch_geometric.loader as tgl
from rpad.pyg.dataset import CachedByKeyDataset

from online_adaptation.datasets.flow_history import FlowHistoryPyGDataset


class FlowHistoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size,
        num_workers,
        n_proc,
        randomize_camera: bool = True,
        seed=42,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed

        self.train_dset = CachedByKeyDataset(
            dset_cls=FlowHistoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-train-train",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS,
            root=root,
            processed_dirname=FlowHistoryPyGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=100,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        # For validation, we don't want to repeat the data.
        self.train_val_dset = CachedByKeyDataset(
            dset_cls=FlowHistoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-train-train",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TRAIN_TRAIN_OBJ_IDS,
            root=root,
            processed_dirname=FlowHistoryPyGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.val_dset = CachedByKeyDataset(
            dset_cls=FlowHistoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-train-test",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TRAIN_TEST_OBJ_IDS,
            root=root,
            processed_dirname=FlowHistoryPyGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

        self.unseen_dset = CachedByKeyDataset(
            dset_cls=FlowHistoryPyGDataset,
            dset_kwargs=dict(
                root=os.path.join(root, "raw"),
                split="umpnet-test",
                randomize_camera=randomize_camera,
            ),
            data_keys=rpd.UMPNET_TEST_OBJ_IDS,
            root=root,
            processed_dirname=FlowHistoryPyGDataset.get_processed_dir(
                True,
                randomize_camera,
            ),
            n_repeat=1,
            n_workers=num_workers,
            n_proc_per_worker=n_proc,
            seed=seed,
        )

    def train_dataloader(self, shuffle=True):
        if shuffle:
            L.seed_everything(self.seed)
        return tgl.DataLoader(
            self.train_dset, self.batch_size, shuffle=shuffle, num_workers=0
        )

    def train_val_dataloader(self):
        return tgl.DataLoader(
            self.train_val_dset, self.batch_size, shuffle=False, num_workers=0
        )

    def val_dataloader(self):
        return tgl.DataLoader(
            self.val_dset, self.batch_size, shuffle=False, num_workers=0
        )

    def unseen_dataloader(self):
        return tgl.DataLoader(
            self.unseen_dset, self.batch_size, shuffle=False, num_workers=0
        )

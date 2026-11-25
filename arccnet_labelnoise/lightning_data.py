"""
PyTorch Lightning DataModule for Hale classification.
"""

import pandas as pd

import arccnet_labelnoise.config as config
from arccnet.models.cutouts.hale.lightning_data import HaleDataModule as BaseDataModule
from arccnet_labelnoise.dataset import HaleDataset, get_fold_data


class HaleDataModule(BaseDataModule):
    """Lightning DataModule for Hale classification."""

    def __init__(self, df: pd.DataFrame = None, fold_num: int = 1, batch_size: int = None, num_workers: int = None):
        super().__init__(df, fold_num, batch_size, num_workers)
        self.batch_size = config.BATCH_SIZE if batch_size is None else batch_size
        self.num_workers = config.NUM_WORKERS if num_workers is None else num_workers

    def setup(self, stage: str = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            train_df, val_df, test_df = get_fold_data(self.df, self.fold_num)

            self.train_dataset = HaleDataset(train_df, data_type=config.DATA_TYPE, is_training=True)
            self.val_dataset = HaleDataset(val_df, data_type=config.DATA_TYPE, is_training=False)
            self.test_dataset = HaleDataset(test_df, data_type=config.DATA_TYPE, is_training=False)

        if stage == "test":
            if self.test_dataset is None:
                _, _, test_df = get_fold_data(self.df, self.fold_num)
                self.test_dataset = HaleDataset(test_df, data_type=config.DATA_TYPE, is_training=False)

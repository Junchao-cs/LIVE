import lightning.pytorch as pl
from torch.utils.data import DataLoader
from sgm.util import instantiate_from_config

class RealEstate10KDataModule(pl.LightningDataModule):
    """
    RealEstate10K data module compatible with the recorded LIVE configs.
    """
    def __init__(
        self,
        train=None,
        validation=None,
        test=None,
        predict=None,
        num_workers=None,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.train_config = train
        self.validation_config = validation
        self.test_config = test
        self.predict_config = predict
        self.num_workers = num_workers
        self.shuffle_val_dataloader = shuffle_val_dataloader

    def setup(self, stage=None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            if self.train_config:
                self.train_dataset = instantiate_from_config(self.train_config)
            if self.validation_config:
                self.val_dataset = instantiate_from_config(self.validation_config)

        if stage == "validate" or stage is None:
            if self.validation_config:
                self.val_dataset = instantiate_from_config(self.validation_config)

        if stage == "test" or stage is None:
            if self.test_config:
                self.test_dataset = instantiate_from_config(self.test_config)

        if stage == "predict" or stage is None:
            if self.predict_config:
                self.predict_dataset = instantiate_from_config(self.predict_config)

    def on_train_epoch_start(self):
            current_epoch = self.trainer.current_epoch
            # 检查 train_dataset 是否存在且有 set_current_epoch 方法
            if hasattr(self, 'train_dataset') and hasattr(self.train_dataset, 'set_current_epoch'):
                self.train_dataset.set_current_epoch(current_epoch)

    def train_dataloader(self):
        if hasattr(self, 'train_dataset'):
            # 从配置中获取loader参数
            loader_config = self.train_config.get('loader', {})
            return DataLoader(
                self.train_dataset,
                batch_size=loader_config.get('batch_size', 12),
                num_workers=loader_config.get('num_workers', self.num_workers or 4),
                shuffle=loader_config.get('shuffle', True),
                drop_last=True,
                persistent_workers=False,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            loader_config = self.validation_config.get('loader', {})
            return DataLoader(
                self.val_dataset,
                batch_size=loader_config.get('batch_size', 2),
                num_workers=loader_config.get('num_workers', self.num_workers or 4),
                shuffle=loader_config.get('shuffle', self.shuffle_val_dataloader),
                drop_last=False,
                persistent_workers=False,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            loader_config = self.test_config.get('loader', {})
            return DataLoader(
                self.test_dataset,
                batch_size=loader_config.get('batch_size', 1),
                num_workers=loader_config.get('num_workers', self.num_workers or 0),
                shuffle=False,
                drop_last=False,
                persistent_workers=False,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def predict_dataloader(self):
        if hasattr(self, 'predict_dataset'):
            loader_config = self.predict_config.get('loader', {})
            return DataLoader(
                self.predict_dataset,
                batch_size=loader_config.get('batch_size', 1),
                num_workers=loader_config.get('num_workers', self.num_workers or 0),
                shuffle=False,
                collate_fn=self._get_collate_fn(),
            )
        return None

    def _get_collate_fn(self):
        """获取collate函数"""
        from tvideo.mc.data.collectfn import dict_collation_fn
        return dict_collation_fn
